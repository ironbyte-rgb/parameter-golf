"""
TT+MLA+GQA Transformer model.

Architecture (from Amon's Parameter Golf brief):

  Token Embedding (4096, 256)
  x8 TransformerBlock:
    RMSNorm -> TTLinear(Q) -> MLA Attention -> TTLinear(O) + residual
    RMSNorm -> SwiGLU FFN (256->512->256) + residual
  Final RMSNorm
  LM Head (tied with embedding)

~4.37M trainable parameters (~8.74 MB at BF16).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tt_layers import TTLinear
from mla_attention import MLAAttention


# ------------------------------------------------------------------
# RMS Normalisation
# ------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (Zhang & Sennrich, 2019).

    More efficient than LayerNorm: no mean subtraction, no bias.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms).to(dtype) * self.weight


# ------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU gated FFN (Shazeer, 2020).

    FFN(x) = W_down @ (SiLU(W_gate @ x) * W_up @ x)

    With d_model=256 and ffn_dim=512, the effective width is 1024
    (gate + up each contribute 512, multiplied elementwise).
    """

    def __init__(self, d_model=256, ffn_dim=512):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        # muP: gate/up as hidden weights, down as output-like
        std_hidden = 1.0 / math.sqrt(self.gate_proj.in_features)
        std_output = 1.0 / math.sqrt(self.down_proj.in_features)
        nn.init.normal_(self.gate_proj.weight, std=std_hidden)
        nn.init.normal_(self.up_proj.weight, std=std_hidden)
        nn.init.normal_(self.down_proj.weight, std=std_output)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ------------------------------------------------------------------
# Transformer Block
# ------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """One transformer block with TT-decomposed Q/O projections and MLA attention.

    Block structure:
        x = x + attn_scale * W_o(MLA(W_q(norm(x)), norm(x)))
        x = x + mlp_scale * SwiGLU_FFN(norm(x))
    """

    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_kv_heads=1,
        d_head=32,
        d_c=16,
        ffn_dim=1024,
        tt_rank=16,
        max_seq=512,
    ):
        super().__init__()
        # Attention sub-block
        self.norm1 = RMSNorm(d_model)
        self.W_q = TTLinear(d_model, d_model, tt_rank=tt_rank, bias=False)
        self.attention = MLAAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            d_c=d_c,
            max_seq=max_seq,
        )
        self.W_o = TTLinear(d_model, d_model, tt_rank=tt_rank, bias=False)
        self.attn_scale = nn.Parameter(torch.ones(1))

        # FFN sub-block
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model=d_model, ffn_dim=ffn_dim)
        self.mlp_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # --- Attention ---
        normed = self.norm1(x)
        q = self.W_q(normed)
        attn_out = self.attention(normed, q)
        attn_out = self.W_o(attn_out)
        x = x + attn_out * self.attn_scale

        # --- FFN ---
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out * self.mlp_scale

        return x


# ------------------------------------------------------------------
# Full Model
# ------------------------------------------------------------------

class TTMLATransformer(nn.Module):
    """TT+MLA+GQA Transformer language model.

    Args:
        vocab_size:  vocabulary size (4096)
        d_model:     hidden dimension (256)
        n_layers:    number of transformer blocks (8)
        n_heads:     number of query heads (8)
        n_kv_heads:  number of KV heads (1 — extreme GQA)
        d_head:      dimension per head (32)
        d_c:         MLA KV latent dimension (16)
        ffn_dim:     SwiGLU intermediate dimension (512)
        tt_rank:     TT decomposition rank (16)
        max_seq:     maximum sequence length (512)
    """

    def __init__(
        self,
        vocab_size=50257,
        d_model=400,
        n_layers=16,
        n_heads=8,
        n_kv_heads=1,
        d_head=50,
        d_c=16,
        ffn_dim=1536,
        tt_rank=16,
        max_seq=512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq = max_seq

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_head=d_head,
                d_c=d_c,
                ffn_dim=ffn_dim,
                tt_rank=tt_rank,
                max_seq=max_seq,
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm_final = RMSNorm(d_model)

        # LM head: tied with embedding weight
        # (weight stored as embedding.weight, shared for output projection)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        """muP-style initialisation.

        Embedding: std = 1.0  (muP input embedding rule)
        """
        nn.init.normal_(self.embedding.weight, std=1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, return_loss=False, targets=None):
        """Forward pass.

        Args:
            input_ids:   (B, T) token indices
            return_loss: if True, compute cross-entropy loss
            targets:     (B, T) target token indices (only needed if return_loss=True)

        Returns:
            logits:  (B, T, vocab_size)  if not return_loss
            loss:    scalar cross-entropy if return_loss
        """
        x = self.embedding(input_ids)                   # (B, T, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.norm_final(x)

        # Tied LM head
        logits = F.linear(x, self.embedding.weight)     # (B, T, vocab_size)

        if return_loss:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
            )
            return loss

        return logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def count_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def estimate_model_size_bytes(self, dtype_bytes=2):
        """Estimate model weight size in bytes at the given dtype width."""
        total, _ = self.count_parameters()
        return total * dtype_bytes


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== TTMLATransformer smoke test ===\n")

    model = TTMLATransformer()
    total, trainable = model.count_parameters()

    print(f" Vocabulary:       {model.vocab_size}")
    print(f" d_model:          {model.d_model}")
    print(f" Layers:           {model.n_layers}")
    print(f" Max seq length:   {model.max_seq}")
    print(f" Total params:     {total:,}")
    print(f" Trainable params: {trainable:,}")
    print(f" Model size (BF16): {model.estimate_model_size_bytes(2) / 1024 / 1024:.2f} MB")
    print(f" Model size (FP32): {model.estimate_model_size_bytes(4) / 1024 / 1024:.2f} MB")

    # Breakdown by component
    emb_params = model.embedding.weight.numel()
    attn_kv_params = sum(
        sum(p.numel() for p in blk.attention.parameters())
        for blk in model.blocks
    )
    tt_params = sum(
        sum(p.numel() for p in blk.W_q.parameters()) +
        sum(p.numel() for p in blk.W_o.parameters())
        for blk in model.blocks
    )
    ffn_params = sum(
        sum(p.numel() for p in blk.ffn.parameters())
        for blk in model.blocks
    )
    other_params = total - emb_params - attn_kv_params - tt_params - ffn_params

    print(f"\n Breakdown:")
    print(f"   Embedding:      {emb_params:,}")
    print(f"   TT Q+O (8 blk): {tt_params:,}")
    print(f"   MLA KV (8 blk): {attn_kv_params:,}")
    print(f"   SwiGLU (8 blk): {ffn_params:,}")
    print(f"   Other:           {other_params:,}")

    # Forward pass
    batch, seq = 4, 512
    input_ids = torch.randint(0, model.vocab_size, (batch, seq))
    targets = torch.randint(0, model.vocab_size, (batch, seq))

    with torch.no_grad():
        logits = model(input_ids)
        print(f"\n Forward pass:")
        print(f"   Input:   {input_ids.shape}")
        print(f"   Logits:  {logits.shape}")
        print(f"   logit range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

        loss = model(input_ids, return_loss=True, targets=targets)
        print(f"   Loss:    {loss.item():.4f}")
        # Random-init loss should be ~ln(vocab) = ln(4096) ≈ 8.32
        expected = math.log(model.vocab_size)
        print(f"   Expected ~{expected:.2f}  (ln(vocab))")

    # Gradient check
    loss = model(input_ids, return_loss=True, targets=targets)
    loss.backward()

    grad_norms = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad for {name}"
            grad_norms[name] = p.grad.norm().item()

    max_grad_name = max(grad_norms, key=grad_norms.get)
    print(f"\n Max grad norm: {grad_norms[max_grad_name]:.2f}  ({max_grad_name})")
    print(" All checks passed.")
