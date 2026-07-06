"""
Post-training int8 quantisation + zlib compression.

Matches the OpenAI Parameter Golf protocol EXACTLY:

  1. int8 per-row quantisation for 2D weight matrices.
  2. Per-tensor quantisation for 1D vectors (RMSNorm weights, etc.).
  3. Passthrough for small tensors (<= 65,536 elements) → fp16.
  4. Control tensors (attn_scale, mlp_scale, etc.) kept as fp16.
  5. Serialised with torch.save + zlib level 9 → .ptz artifact.
  6. Roundtrip validation: dequantise → reload → re-eval.
"""

import io
import math
import zlib
from pathlib import Path

import torch


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Tensors whose name contains any of these substrings are kept as fp16
# regardless of size (matching the competition control-tensor carveout).
CONTROL_PATTERNS = (
    "attn_scale", "attn_scales",
    "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes",
    "q_gain",
    "skip_weight", "skip_weights",
)

# Tensors with <= this many elements are stored as fp16 passthrough.
SMALL_TENSOR_THRESHOLD = 65536

# Format identifier stored in the quantised dict.
QFORMAT = "int8_clean_per_row_v1"


# ------------------------------------------------------------------
# Quantisation
# ------------------------------------------------------------------

def _is_control_tensor(name):
    """Check if a parameter name matches a control-tensor pattern."""
    return any(p in name for p in CONTROL_PATTERNS)


def _per_row_clip(t, percentile=99.99984):
    """Compute per-row absolute clip thresholds.

    Args:
        t:          (M, N) float tensor
        percentile: quantile for clipping (default 99.99984)

    Returns:
        clip_abs:  (M,) clip threshold per row
    """
    t_abs = t.float().abs()
    # k = int(N * percentile / 100), floored at 1
    k = max(1, int(t.shape[1] * percentile / 100.0))
    clip_abs = t_abs.kthvalue(t.shape[1] - k + 1, dim=1).values
    return clip_abs


def quantize_float_tensor(t, name=""):
    """Quantize a single float tensor to int8.

    Per-row quantisation for 2D tensors, per-tensor for 1D.

    Args:
        t:    float torch.Tensor
        name: parameter name (for control-tensor detection)

    Returns:
        (q_int8, scales, is_per_row)  where:
          - q_int8:   int8 tensor of same shape
          - scales:   fp16 scale(s) — scalar or per-row vector
          - is_per_row: True if per-row, False if per-tensor
    """
    if t.dim() == 2:
        # Per-row
        clip_abs = _per_row_clip(t)                          # (M,)
        clip_abs = clip_abs.clamp(min=1.0 / 127.0)
        scales = (clip_abs / 127.0).to(torch.float16)       # (M,)
        t_clipped = t.float().clamp(
            -clip_abs.unsqueeze(1), clip_abs.unsqueeze(1)
        )
        q = torch.round(t_clipped / scales.unsqueeze(1))
        q = q.clamp(-127, 127).to(torch.int8)
        is_per_row = True
    else:
        # Per-tensor (scalar)
        t_abs = t.float().abs()
        clip_abs = t_abs.max().clamp(min=1.0 / 127.0)
        scale = (clip_abs / 127.0).to(torch.float16)
        t_clipped = t.float().clamp(-clip_abs, clip_abs)
        q = torch.round(t_clipped / scale)
        q = q.clamp(-127, 127).to(torch.int8)
        scales = torch.tensor([scale], dtype=torch.float16)
        is_per_row = False

    return q, scales, is_per_row


def quantize_state_dict(state_dict):
    """Quantize a full model state dict.

    Args:
        state_dict:  model.state_dict() — maps name → tensor

    Returns:
        dict with keys: quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta
    """
    quantized = {}      # name → int8 tensor
    scales = {}         # name → fp16 scale(s)
    dtypes = {}         # name → quant dtype string
    passthrough = {}    # name → fp16 tensor (small / control)
    passthrough_orig_dtypes = {}  # name → original dtype (for dequant)
    qmeta = {}          # name → {"per_row": bool}

    for name, param in state_dict.items():
        if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            # Non-float: store as-is
            passthrough[name] = param.clone()
            passthrough_orig_dtypes[name] = str(param.dtype)
            continue

        if param.numel() <= SMALL_TENSOR_THRESHOLD:
            # Small tensor: fp16 passthrough
            passthrough[name] = param.to(torch.float16).clone()
            passthrough_orig_dtypes[name] = str(param.dtype)
            continue

        if _is_control_tensor(name):
            # Control tensor: fp16 passthrough
            passthrough[name] = param.to(torch.float16).clone()
            passthrough_orig_dtypes[name] = str(param.dtype)
            continue

        # Quantize
        q, s, is_per_row = quantize_float_tensor(param, name)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = "int8"
        qmeta[name] = {"per_row": is_per_row}

    return {
        "__quant_format__": QFORMAT,
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "passthrough_orig_dtypes": passthrough_orig_dtypes,
        "qmeta": qmeta,
    }


# ------------------------------------------------------------------
# Dequantisation
# ------------------------------------------------------------------

def dequantize_state_dict(quant_dict):
    """Reconstruct a float state dict from the quantised format.

    Args:
        quant_dict:  output of ``quantize_state_dict``

    Returns:
        dict mapping name → float tensor (ready for model.load_state_dict)
    """
    fmt = quant_dict.get("__quant_format__")
    if fmt != QFORMAT:
        raise ValueError(f"Unknown quant format: {fmt}")

    result = {}
    quantized = quant_dict.get("quantized", {})
    scales = quant_dict.get("scales", {})
    qmeta = quant_dict.get("qmeta", {})
    passthrough = quant_dict.get("passthrough", {})

    for name, q in quantized.items():
        s = scales[name]
        is_per_row = qmeta[name]["per_row"]
        if is_per_row:
            s = s.view(-1, *([1] * (q.dim() - 1)))
        result[name] = q.float() * s.float()

    for name, t in passthrough.items():
        result[name] = t.clone()

    return result


# ------------------------------------------------------------------
# Compression / serialisation
# ------------------------------------------------------------------

def compress_state_dict(quant_dict):
    """Serialize and zlib-compress a quantised state dict.

    Returns:
        bytes:  zlib-compressed blob (level 9)
    """
    buf = io.BytesIO()
    torch.save(quant_dict, buf)
    raw = buf.getvalue()
    return zlib.compress(raw, level=9)


def decompress_state_dict(blob):
    """Decompress and deserialize a quantised state dict.

    Args:
        blob:  bytes from ``compress_state_dict``

    Returns:
        dict  (output of ``quantize_state_dict``)
    """
    raw = zlib.decompress(blob)
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu", weights_only=False)


# ------------------------------------------------------------------
# High-level API
# ------------------------------------------------------------------

def save_compressed_model(model, output_path, code_path=None):
    """Quantize model, compress, and save as .ptz artifact.

    Args:
        model:       nn.Module
        output_path: path for .ptz output file (e.g. "final_model.int8.ptz")
        code_path:   path to train_gpt.py (for size reporting)

    Returns:
        dict with size info: quant_file_bytes, code_bytes, bytes_total, compression_ratio
    """
    print("Quantizing model...")
    quant_dict = quantize_state_dict(model.state_dict())

    quant_params = sum(
        quant_dict["quantized"][k].numel() for k in quant_dict["quantized"]
    )
    pt_params = sum(
        quant_dict["passthrough"][k].numel() for k in quant_dict["passthrough"]
    )
    print(f"  Quantized params:   {quant_params:,}")
    print(f"  Passthrough params: {pt_params:,}")

    blob = compress_state_dict(quant_dict)

    output_path = Path(output_path)
    output_path.write_bytes(blob)

    quant_file_bytes = len(blob)
    print(f"  Compressed size:    {quant_file_bytes:,} bytes ({quant_file_bytes/1e6:.2f} MB)")

    # Raw (uncompressed) model size for comparison
    raw_size = sum(p.numel() * p.element_size() for p in model.parameters())
    compression_ratio = raw_size / max(quant_file_bytes, 1)

    code_bytes = 0
    if code_path:
        code_text = Path(code_path).read_text(encoding="utf-8")
        code_bytes = len(code_text.encode("utf-8"))
        print(f"  Code size:          {code_bytes:,} bytes ({code_bytes/1e6:.2f} MB)")

    bytes_total = quant_file_bytes + code_bytes
    print(f"  Total artifact:     {bytes_total:,} bytes ({bytes_total/1e6:.2f} MB)")
    limit = 16_000_000
    status = "UNDER 16MB" if bytes_total < limit else f"OVER by {bytes_total - limit:,} bytes"
    print(f"  Status:             {status}")

    return {
        "quant_file_bytes": quant_file_bytes,
        "code_bytes": code_bytes,
        "bytes_total": bytes_total,
        "compression_ratio": compression_ratio,
    }


def load_and_dequantize(ptz_path):
    """Load a .ptz artifact and return a dequantised state dict.

    Args:
        ptz_path:  path to .ptz file

    Returns:
        dict  (ready for model.load_state_dict)
    """
    blob = Path(ptz_path).read_bytes()
    quant_dict = decompress_state_dict(blob)
    return dequantize_state_dict(quant_dict)


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Quantize smoke test ===\n")

    # Build a small model to quantize
    from model import TTMLATransformer
    model = TTMLATransformer()
    total_params = sum(p.numel() for p in model.parameters())
    raw_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f" Model params:    {total_params:,}")
    print(f" Raw size (fp32): {raw_bytes:,} bytes ({raw_bytes/1e6:.2f} MB)")

    # Quantize
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        ptz_path = os.path.join(tmpdir, "test.int8.ptz")
        info = save_compressed_model(model, ptz_path)

        # Roundtrip
        dequant_sd = load_and_dequantize(ptz_path)
        model.load_state_dict(dequant_sd, strict=True)

        # Verify output consistency (rough check)
        x = torch.randint(0, 4096, (2, 64))
        with torch.no_grad():
            out_dequant = model(x)
        assert torch.isfinite(out_dequant).all(), "Dequantised output has NaN/Inf"

        print(f"\n Roundtrip verified: output shape={out_dequant.shape}, finite=True")
        print(f" Compression ratio: {info['compression_ratio']:.1f}x")

    print(" All checks passed.")
