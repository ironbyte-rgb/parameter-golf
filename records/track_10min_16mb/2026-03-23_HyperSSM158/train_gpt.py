"""
HyperSSM-1.58: Hypernetwork-Generated Selective SSM with Ternary Quantization
for the OpenAI Parameter Golf Challenge.

Architecture: Selective State Space Model (Mamba-style) with weights generated
on-the-fly by a small hypernetwork seed (~230K BF16 params). Continuous depth
indexing via Euler integration with k=12 training / k=30 eval steps.
Native 1.58-bit ternary quantization {-1, 0, +1} with progressive STE training.

Stored params: ~1M (tiny artifact). Effective params: ~16M+ at eval depth.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 512))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # HyperSSM model shape
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    d_model = int(os.environ.get("D_MODEL", 384))
    d_inner = int(os.environ.get("D_INNER", 768))
    d_state = int(os.environ.get("D_STATE", 16))
    dt_rank = int(os.environ.get("DT_RANK", 24))  # d_model // 16
    conv_kernel = int(os.environ.get("CONV_KERNEL", 4))
    hyper_dim = int(os.environ.get("HYPER_DIM", 256))
    hyper_rank = int(os.environ.get("HYPER_RANK", 48))
    train_depth = int(os.environ.get("TRAIN_DEPTH", 12))
    eval_depth = int(os.environ.get("EVAL_DEPTH", 30))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer
    learning_rate = float(os.environ.get("LEARNING_RATE", 1e-2))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.1))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    # 3-phase timing (seconds)
    phase1_end = float(os.environ.get("PHASE1_END", 90.0))
    phase2_end = float(os.environ.get("PHASE2_END", 300.0))
    save_at = float(os.environ.get("SAVE_AT", 580.0))
    warmdown_start = float(os.environ.get("WARMDOWN_START", 520.0))

    # Test mode
    test_mode = bool(int(os.environ.get("TEST_MODE", "0")))


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "norm,scale,D_param,A_log",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# TERNARY QUANTIZATION (STE)
# -----------------------------

class TernaryQuantize(torch.autograd.Function):
    """Absmean ternary quantization with Straight-Through Estimator.

    Forward: W_ternary = alpha * round(clamp(W / alpha, -1, 1))
    where alpha = mean(|W|). Result is in {-alpha, 0, +alpha}.
    Backward: gradient passes through unchanged (STE).
    """

    @staticmethod
    def forward(ctx, w: Tensor, ternary_frac: float) -> Tensor:
        if ternary_frac <= 0.0:
            return w
        alpha = w.abs().mean().clamp_min(1e-8)
        w_ternary = alpha * torch.clamp(torch.round(w / alpha), -1.0, 1.0)
        if ternary_frac >= 1.0:
            return w_ternary
        # Progressive interpolation: stop gradient on ternary part to avoid double-counting
        return (1.0 - ternary_frac) * w + ternary_frac * w_ternary.detach()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


def ternarize(w: Tensor, ternary_frac: float) -> Tensor:
    return TernaryQuantize.apply(w, ternary_frac)


# -----------------------------
# MODEL MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class HyperNetWeightGen(nn.Module):
    """Hypernetwork that generates SSM block weights from a depth index t in [0,1].

    Architecture:
    - Sinusoidal encoding of t -> 64-dim
    - MLP: Linear(64->hyper_dim) -> SiLU -> Linear(hyper_dim->hyper_dim) -> SiLU
    - Output: Per-weight-type modulation vectors applied to stored low-rank factors
    - Weight W(t) = U @ diag(modulation(t)) @ V, then ternarized

    Stored parameters: U_i, V_i (low-rank bases) + MLP weights + modulation heads
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        d = args.d_model
        di = args.d_inner
        ds = args.d_state
        dr = args.dt_rank
        ck = args.conv_kernel
        r = args.hyper_rank  # low-rank dimension

        sinusoidal_dim = 64
        hd = args.hyper_dim

        # Sinusoidal encoding dimension
        self.sinusoidal_dim = sinusoidal_dim

        # MLP backbone
        self.mlp1 = nn.Linear(sinusoidal_dim, hd)
        self.mlp2 = nn.Linear(hd, hd)

        # Define weight specs: (name, out_dim, in_dim, rank)
        # in_proj: d_model -> 2*d_inner (x and gate z)
        # x_proj: d_inner -> dt_rank + 2*d_state
        # dt_proj: dt_rank -> d_inner
        # out_proj: d_inner -> d_model
        self.weight_specs = {
            'in_proj': (2 * di, d, r),          # (1536, 384, r)
            'x_proj': (dr + 2 * ds, di, r),     # (56, 768, r)
            'dt_proj': (di, dr, r),              # (768, 24, r)
            'out_proj': (d, di, r),              # (384, 768, r)
        }

        # Low-rank basis factors U_i, V_i for each weight type
        self.U_factors = nn.ParameterDict()
        self.V_factors = nn.ParameterDict()
        self.mod_heads = nn.ModuleDict()

        for name, (out_d, in_d, rank) in self.weight_specs.items():
            self.U_factors[name] = nn.Parameter(torch.empty(out_d, rank))
            self.V_factors[name] = nn.Parameter(torch.empty(rank, in_d))
            # Modulation head: hyper_dim -> rank
            self.mod_heads[name] = nn.Linear(hd, rank, bias=False)

        # Conv1d weights stored directly (small enough, 768*4=3072 params)
        self.conv1d_weight = nn.Parameter(torch.empty(di, 1, ck))
        # Conv1d modulation: just a per-channel scale from depth
        self.conv1d_mod_head = nn.Linear(hd, di, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Xavier for MLP
        nn.init.xavier_uniform_(self.mlp1.weight)
        nn.init.zeros_(self.mlp1.bias)
        nn.init.xavier_uniform_(self.mlp2.weight)
        nn.init.zeros_(self.mlp2.bias)

        r = self.args.hyper_rank
        for name, (out_d, in_d, rank) in self.weight_specs.items():
            # Initialize U, V so that product UV has Xavier-scale variance
            # Var[UV_ij] = r * std_U^2 * std_V^2 => want sqrt(2/(in_d+out_d))
            # => std = (2/(in_d+out_d))^0.25 / r^0.25
            target_std = (2.0 / (in_d + out_d)) ** 0.25 / (rank ** 0.25)
            nn.init.normal_(self.U_factors[name], std=target_std)
            nn.init.normal_(self.V_factors[name], std=target_std)
            nn.init.normal_(self.mod_heads[name].weight, std=0.01)  # small init for depth variation

        # Conv1d init
        nn.init.kaiming_uniform_(self.conv1d_weight, a=math.sqrt(5))
        nn.init.normal_(self.conv1d_mod_head.weight, std=0.01)

    def _sinusoidal_encoding(self, t: float, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Encode scalar t in [0,1] using sinusoidal positional encoding."""
        half = self.sinusoidal_dim // 2
        freqs = torch.arange(half, device=device, dtype=dtype)
        freqs = torch.exp(freqs * (-math.log(10000.0) / half))
        angles = t * freqs
        return torch.cat([angles.sin(), angles.cos()])  # (sinusoidal_dim,)

    def forward(self, t: float, ternary_frac: float) -> dict[str, Tensor]:
        """Generate all SSM block weights for depth index t.

        Args:
            t: depth index in [0, 1]
            ternary_frac: interpolation fraction for ternary quantization

        Returns:
            dict mapping weight names to weight tensors
        """
        device = self.mlp1.weight.device
        dtype = self.mlp1.weight.dtype

        # Encode depth
        enc = self._sinusoidal_encoding(t, device, dtype)  # (64,)

        # MLP backbone
        h = F.silu(self.mlp1(enc))
        h = F.silu(self.mlp2(h))  # (hyper_dim,)

        weights = {}

        # Generate each weight matrix via modulated low-rank factorization
        for name in self.weight_specs:
            U = self.U_factors[name]  # (out_d, r)
            V = self.V_factors[name]  # (r, in_d)
            mod = self.mod_heads[name](h)  # (r,)
            mod = torch.sigmoid(mod)  # scale to [0, 1] for stable modulation

            # W(t) = U @ diag(mod) @ V
            w = (U * mod.unsqueeze(0)) @ V  # (out_d, in_d)

            # Apply ternary quantization
            w = ternarize(w, ternary_frac)
            weights[name] = w

        # Conv1d weight with depth-dependent modulation
        conv_mod = torch.sigmoid(self.conv1d_mod_head(h))  # (d_inner,)
        conv_w = self.conv1d_weight * conv_mod.unsqueeze(-1).unsqueeze(-1)
        conv_w = ternarize(conv_w.reshape(-1, conv_w.shape[-1]), ternary_frac)
        weights['conv1d'] = conv_w.reshape(self.args.d_inner, 1, self.args.conv_kernel)

        return weights


class SelectiveSSMBlock(nn.Module):
    """Single Mamba-style Selective State Space block.

    Uses externally-provided weights (from hypernetwork).
    Only stores: RMSNorm params, A_log buffer, D skip-connection.
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.norm = RMSNorm(args.d_model)

        # A matrix: stored as log, initialized as small negative values for stable gradients
        # Using smaller values than standard Mamba to prevent vanishing gradients
        A = 0.5 + 0.5 * torch.arange(1, args.d_state + 1, dtype=torch.float32)  # [1, 1.5, ..., 8.5]
        A = A.unsqueeze(0).expand(args.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A), requires_grad=True)  # make learnable

        # D skip connection
        self.D_param = nn.Parameter(torch.ones(args.d_inner))

        # dt_proj bias: initialize so softplus(bias) gives reasonable delta values (~0.1-1.0)
        # softplus(x) ≈ x for x >> 0, so initialize near log(exp(target)-1)
        dt_init = torch.empty(args.d_inner).uniform_(0.001, 0.1)
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))  # inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)

    def forward(self, x: Tensor, weights: dict[str, Tensor], step_size: float) -> Tensor:
        """Forward pass with externally provided weights.

        Args:
            x: input (B, T, d_model)
            weights: dict of weight tensors from hypernetwork
            step_size: Euler step size (1/k)

        Returns:
            output (B, T, d_model) with residual connection
        """
        residual = x
        x = self.norm(x)

        B, T, D = x.shape
        d_inner = self.args.d_inner
        d_state = self.args.d_state
        dt_rank = self.args.dt_rank

        # in_proj: (B, T, d_model) -> (B, T, 2*d_inner)
        xz = F.linear(x, weights['in_proj'])  # (B, T, 2*d_inner)
        x_main, z = xz.split([d_inner, d_inner], dim=-1)

        # Short 1D convolution (causal, kernel_size=4)
        # Reshape for conv1d: (B, d_inner, T)
        x_conv = x_main.transpose(1, 2)
        x_conv = F.conv1d(
            x_conv,
            weights['conv1d'],
            bias=None,
            padding=self.args.conv_kernel - 1,
            groups=d_inner,
        )[..., :T]
        x_main = x_conv.transpose(1, 2)  # (B, T, d_inner)

        x_main = F.silu(x_main)

        # SSM parameter projection: x -> (delta, B, C)
        x_dbc = F.linear(x_main, weights['x_proj'])  # (B, T, dt_rank + 2*d_state)
        delta_proj, B_proj, C_proj = x_dbc.split([dt_rank, d_state, d_state], dim=-1)

        # delta = softplus(dt_proj(delta_proj) + dt_bias)
        delta = F.linear(delta_proj, weights['dt_proj']) + self.dt_bias  # (B, T, d_inner)
        delta = F.softplus(delta)  # (B, T, d_inner)

        # Selective scan
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        y = self._selective_scan(x_main, delta, A, B_proj, C_proj, self.D_param)

        # Gating
        y = y * F.silu(z)

        # Output projection
        out = F.linear(y, weights['out_proj'])  # (B, T, d_model)

        # Residual with Euler step scaling
        return residual + step_size * out

    def _selective_scan(
        self,
        x: Tensor,      # (B, T, d_inner)
        delta: Tensor,   # (B, T, d_inner)
        A: Tensor,       # (d_inner, d_state)
        B: Tensor,       # (B, T, d_state)
        C: Tensor,       # (B, T, d_state)
        D: Tensor,       # (d_inner,)
    ) -> Tensor:
        """Selective scan (sequential implementation).

        h_k = A_bar * h_{k-1} + B_bar * x_k
        y_k = C_k^T * h_k + D * x_k
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize: A_bar = exp(delta * A)
        # delta: (B, T, d_inner), A: (d_inner, d_state)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, T, d_inner, d_state)
        dB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, T, d_inner, d_state)

        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]
            h = h.clamp(-100, 100)  # stability clamp for ternary
            y_t = (h * C[:, t, None, :]).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        y = y + x * D.unsqueeze(0).unsqueeze(0)
        return y


class HyperSSM158(nn.Module):
    """HyperSSM-1.58: Full language model.

    Token embedding -> k Euler steps of SSM block (with hypernetwork weights) -> LM head.
    Tied embedding/LM head. Progressive ternary quantization.
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        self.hypernet = HyperNetWeightGen(args)
        self.ssm_block = SelectiveSSMBlock(args)
        self.final_norm = RMSNorm(args.d_model)
        self.logit_softcap = args.logit_softcap

        # Current ternary fraction (set externally during training)
        self._ternary_frac = 0.0
        self._depth_steps = args.train_depth

    def set_ternary_frac(self, frac: float):
        self._ternary_frac = frac

    def set_depth(self, k: int):
        self._depth_steps = k

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))

        k = self._depth_steps
        # Use constant step_size=1 for residual connections (like standard residual nets)
        # The depth variation comes from different hypernetwork-generated weights at each t
        step_size = 1.0

        # k Euler steps with hypernetwork-generated weights
        for i in range(k):
            t = (i + 0.5) / k  # depth index in (0, 1)
            weights = self.hypernet(t, self._ternary_frac)
            x = self.ssm_block(x, weights, step_size)

        x = self.final_norm(x)
        x = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        # Tied embedding as LM head
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    # Test mode overrides
    if args.test_mode:
        args.train_depth = 2
        args.eval_depth = 4
        args.iterations = 100
        args.train_batch_tokens = min(args.train_batch_tokens, 8192)
        args.val_batch_size = min(args.val_batch_size, 8192)
        args.max_wallclock_seconds = 120.0
        args.warmup_steps = 2
        args.val_loss_every = 50
        args.train_log_every = 10
        args.phase1_end = 15.0
        args.phase2_end = 45.0
        args.save_at = 100.0
        args.warmdown_start = 80.0

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    grad_accum_steps = max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        if not args.test_mode:
            raise RuntimeError("CUDA is required for training (use TEST_MODE=1 for CPU testing)")
        device = torch.device("cpu")

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    if use_cuda:
        log0(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
            console=False,
        )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER
    # -----------------------------

    base_model = HyperSSM158(args).to(device)
    if use_cuda:
        base_model = base_model.bfloat16()
        # Keep norms and small params in fp32
        for name, param in base_model.named_parameters():
            if param.ndim < 2 or any(p in name for p in ("norm", "D_param", "dt_bias")):
                param.data = param.data.float()
            if "A_log" in name:
                param.data = param.data.float()

    # Don't compile the full model due to hypernetwork dynamics
    # Compile just the SSM scan if possible
    if use_cuda:
        try:
            base_model.ssm_block._selective_scan = torch.compile(
                base_model.ssm_block._selective_scan,
                mode='reduce-overhead',
                dynamic=False,
            )
            log0("torch.compile: SSM scan compiled successfully")
        except Exception as e:
            log0(f"torch.compile: failed ({e}), running eager")

    model: nn.Module = (
        DDP(base_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        if distributed
        else base_model
    )

    # AdamW optimizer for all parameters
    param_groups = []
    embed_params = []
    other_params = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'tok_emb' in name:
            embed_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": args.learning_rate * 0.1, "base_lr": args.learning_rate * 0.1},
            {"params": other_params, "lr": args.learning_rate, "base_lr": args.learning_rate},
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=use_cuda,
    )

    n_params = sum(p.numel() for p in base_model.parameters())
    n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    log0(f"model_params:{n_params} trainable:{n_trainable}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"architecture:HyperSSM-1.58 d_model:{args.d_model} d_inner:{args.d_inner} d_state:{args.d_state}")
    log0(f"hyper_rank:{args.hyper_rank} hyper_dim:{args.hyper_dim}")
    log0(f"train_depth:{args.train_depth} eval_depth:{args.eval_depth}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # Verify artifact size
    if master_process:
        total_bytes_est = n_params * 1  # INT8 = 1 byte/param
        log0(f"estimated_artifact_size:{total_bytes_est} bytes ({total_bytes_est/1e6:.2f} MB)")

    # -----------------------------
    # DATA LOADER & WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    # Warmup primes compiled paths
    if args.warmup_steps > 0 and use_cuda:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
        model.train()
        for warmup_step in range(args.warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        optimizer.load_state_dict(initial_optimizer_state)
        optimizer.zero_grad(set_to_none=True)
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP (3-phase)
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    saved_checkpoint = False
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        # Validation
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            if use_cuda:
                torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Switch to eval depth for validation
            old_depth = base_model._depth_steps
            base_model.set_depth(args.eval_depth)
            base_model.set_ternary_frac(base_model._ternary_frac)  # keep current ternary setting
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            base_model.set_depth(old_depth)
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        # Compute elapsed time
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        elapsed_s = elapsed_ms / 1000.0

        # ---- 3-PHASE PROTOCOL ----
        if elapsed_s < args.phase1_end:
            # Phase 1: BF16 warmup, low depth, no ternary
            phase = 1
            ternary_frac = 0.0
            depth = max(4, args.train_depth // 3)
            # LR warmup: 0 -> base
            phase_progress = elapsed_s / args.phase1_end
            lr_scale = phase_progress
        elif elapsed_s < args.phase2_end:
            # Phase 2: Progressive ternarization, increasing depth
            phase = 2
            phase_progress = (elapsed_s - args.phase1_end) / (args.phase2_end - args.phase1_end)
            ternary_frac = phase_progress  # 0 -> 1
            depth = int(4 + (args.train_depth - 4) * phase_progress)
            depth = max(4, min(depth, args.train_depth))
            lr_scale = 1.0
        else:
            # Phase 3: Full ternary, full depth, WSD decay
            phase = 3
            ternary_frac = 1.0
            depth = args.train_depth
            if elapsed_s >= args.warmdown_start:
                remaining = max(args.save_at - elapsed_s, 0.0)
                warmdown_len = args.save_at - args.warmdown_start
                lr_scale = remaining / max(warmdown_len, 1e-9)
            else:
                lr_scale = 1.0

        base_model.set_ternary_frac(ternary_frac)
        base_model.set_depth(depth)

        # Update learning rates
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_scale

        # Training step
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda)
            with autocast_ctx:
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"phase:{phase} ternary:{ternary_frac:.2f} depth:{depth} "
                f"lr_scale:{lr_scale:.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Save checkpoint near deadline
        if not saved_checkpoint and elapsed_s >= args.save_at:
            if master_process:
                log0(f"saving checkpoint at {elapsed_s:.1f}s")
                torch.save(base_model.state_dict(), "final_model.pt")
                saved_checkpoint = True

        # Wallclock cap
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if use_cuda:
        log0(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
        )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP
    # -----------------------------

    if master_process:
        if not saved_checkpoint:
            torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    # Final eval with eval depth
    base_model.set_depth(args.eval_depth)
    base_model.set_ternary_frac(1.0)

    if use_cuda:
        torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    if use_cuda:
        torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
