"""
Multi-head Latent Attention (MLA) with Grouped-Query Attention (GQA).

Implements the MLA mechanism from DeepSeek-V2:
  1. Compress all KV representations to a low-dimensional latent vector (d_c).
  2. Decompress K and V per-head from the latent at attention time.
  3. Apply RoPE to Q and K after projection.
  4. GQA: broadcast the single KV head across all query heads.

Caches only the latent vector c^{KV} (d_c dims) per token, achieving ~4x
KV-cache reduction vs standard MHA at the same head dimension.

Reference: DeepSeek-V2 (arXiv:2405.04434)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLAAttention(nn.Module):
    """Multi-head Latent Attention with GQA.

    Q projection is done EXTERNALLY (by the transformer block, typically
    via a TTLinear).  This module handles KV compression/decompression
    and the attention computation.

    Args:
        d_model:    hidden dimension (256)
        n_heads:    number of query heads (8)
        n_kv_heads: number of key/value heads (1 — extreme GQA)
        d_head:     dimension per head (32)
        d_c:        KV latent bottleneck dimension (16)
        max_seq:    maximum sequence length for RoPE cache (512)
    """

    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_kv_heads=1,
        d_head=32,
        d_c=16,
        max_seq=512,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.d_c = d_c
        self.max_seq = max_seq
        self.heads_per_kv = n_heads // n_kv_heads  # 8

        # ---- KV compression (MLA core) ----
        # Down-projection:  d_model -> d_c  (shared across all KV heads)
        self.W_kv_down = nn.Linear(d_model, d_c, bias=False)

        # Up-projections:  d_c -> d_head  (for the single KV head)
        self.W_k_up = nn.Linear(d_c, n_kv_heads * d_head, bias=False)
        self.W_v_up = nn.Linear(d_c, n_kv_heads * d_head, bias=False)

        # ---- RoPE cache ----
        self._build_rope_cache()

        # ---- muP-style initialisation ----
        self._init_weights()

    # ------------------------------------------------------------------
    # RoPE
    # ------------------------------------------------------------------

    def _build_rope_cache(self):
        """Precompute cos/sin tables for RoPE (theta = 10 000)."""
        theta = 10000.0
        dim = self.d_head
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        t = torch.arange(self.max_seq, dtype=torch.float32)
        freqs = torch.outer(t, freqs)                     # (seq, dim/2)
        self.register_buffer("rope_cos", freqs.cos())      # (seq, dim/2)
        self.register_buffer("rope_sin", freqs.sin())      # (seq, dim/2)

    def _apply_rope(self, x):
        """Apply rotary position embedding to tensor x.

        Splits the last dimension into even/odd pairs and applies the
        2-D rotation:  (x_{2i}, x_{2i+1}) rotated by angle theta_i.

        Args:
            x:  (B, heads, T, d_head)

        Returns:
            x with RoPE applied (B, heads, T, d_head)
        """
        B, H, T, D = x.shape
        half = D // 2

        # Separate even (2i) and odd (2i+1) dimensions.
        x_reshaped = x.reshape(B, H, T, half, 2)
        x_even = x_reshaped[..., 0]          # (B, H, T, half)
        x_odd = x_reshaped[..., 1]           # (B, H, T, half)

        cos = self.rope_cos[:T].view(1, 1, T, half)
        sin = self.rope_sin[:T].view(1, 1, T, half)

        # Apply rotation to each pair.
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_odd * cos + x_even * sin

        # Interleave even/odd back: (even_0, odd_0, even_1, odd_1, ...)
        return torch.stack([rot_even, rot_odd], dim=-1).reshape(B, H, T, D)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        """muP-aware initialization for MLA's asymmetric projections.

        W_kv_down:  input-like (d_c is narrow, fixed) — std = 1/sqrt(d_model)
        W_k_up:     output-like (expands to d_model) — std = 1/sqrt(d_c)
        W_v_up:     same as W_k_up
        """
        # Down-projection: input-like (d_model is the "wide" input dim)
        nn.init.normal_(self.W_kv_down.weight, std=1.0 / math.sqrt(self.d_model))

        # Up-projections: output-like (d_c is the narrow input, d_model output)
        nn.init.normal_(self.W_k_up.weight, std=1.0 / math.sqrt(self.d_c))
        nn.init.normal_(self.W_v_up.weight, std=1.0 / math.sqrt(self.d_c))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, q):
        """Forward pass.

        Args:
            x:  hidden states  (B, T, d_model) — used for KV projection
            q:  pre-projected queries  (B, T, n_heads * d_head) — from TTLinear

        Returns:
            attention output  (B, T, n_heads * d_head) — to be projected
            back to d_model by the block's output TTLinear
        """
        B, T, _ = x.shape

        # --- Query: reshape pre-projected Q from external TTLinear ---
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # (B, n_heads, T, d_head)
        q = self._apply_rope(q)

        # --- KV latent compression (the MLA bit) ---
        kv_latent = self.W_kv_down(x)                       # (B, T, d_c)

        # Decompress K and V from the latent
        k = self.W_k_up(kv_latent)                           # (B, T, n_kv*d_head)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        # (B, n_kv_heads, T, d_head)
        k = self._apply_rope(k)

        v = self.W_v_up(kv_latent)                           # (B, T, n_kv*d_head)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        # (B, n_kv_heads, T, d_head)

        # --- GQA broadcast: expand KV heads to match Q heads ---
        if self.n_kv_heads < self.n_heads:
            k = k.expand(-1, self.heads_per_kv, -1, -1).reshape(
                B, self.n_heads, T, self.d_head
            )
            v = v.expand(-1, self.heads_per_kv, -1, -1).reshape(
                B, self.n_heads, T, self.d_head
            )

        # --- Flash Attention via PyTorch SDPA ---
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            scale=1.0 / math.sqrt(self.d_head),
        )

        # Merge heads back
        out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        # (B, T, n_heads * d_head)
        return out


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== MLAAttention smoke test ===")

    batch, seq, d_model = 2, 512, 256
    n_heads, n_kv, d_head, d_c = 8, 1, 32, 16

    layer = MLAAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        d_head=d_head,
        d_c=d_c,
        max_seq=seq,
    )

    x = torch.randn(batch, seq, d_model)
    q = torch.randn(batch, seq, n_heads * d_head)

    out = layer(x, q)

    print(f" Hidden shape:   {x.shape}")
    print(f" Q shape:        {q.shape}")
    print(f" Output shape:   {out.shape}")
    print(f" KV latent dim:  {d_c}")

    # Param count
    n_params = sum(p.numel() for p in layer.parameters())
    # KV params: W_down(256*16) + W_k_up(16*32) + W_v_up(16*32) = 4096+512+512
    print(f" Parameters:     {n_params:,}")
    assert n_params == 4096 + 512 + 512, f"Unexpected param count: {n_params}"
    assert out.shape == (batch, seq, n_heads * d_head)

    # Gradient check
    loss = out.sum()
    loss.backward()
    for name, p in layer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad for {name}"

    # KV cache size per token (what would be cached at inference)
    kv_cache_bytes_per_token = d_c * 2  # K + V latents, fp16
    standard_mha_cache = n_heads * d_head * 2 * 2  # K+V, fp16
    print(f" KV cache/tok (MLA):  {kv_cache_bytes_per_token} bytes")
    print(f" KV cache/tok (MHA):  {standard_mha_cache} bytes")
    print(f" Reduction:           {standard_mha_cache / kv_cache_bytes_per_token:.1f}x")
    print(" All checks passed.")
