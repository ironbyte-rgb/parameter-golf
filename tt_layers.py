"""
Tensor Train (TT) decomposed linear layer.

Replaces a dense (d, d) weight matrix with a chain of small TT cores,
achieving ~8x parameter reduction for the Q and O projections in each
transformer block.  The decomposition is baked into the architecture at
initialization time — this is structural compression, not post-training.

For a 256x256 matrix:
  - Factor 256 = 16 x 16  (2-dimensional factorization)
  - 2 TT cores of shape (factor, factor, rank) and (rank, factor, factor)
  - At rank=16: 2 * 16*16*16 = 8,192 params  (vs 65,536 dense = 8x reduction)

Reference: Oseledets (2011), "Tensor-Train Decomposition"
"""

import math
import torch
import torch.nn as nn


class TTLinear(nn.Module):
    """Tensor Train decomposed linear layer.

    Decomposes a weight matrix W of shape (out_features, in_features) into
    a product of small TT cores.  Currently requires in_features == out_features
    and both must be perfect squares (the architecture brief only uses TT for
    the square Q and O projection matrices).

    Args:
        in_features:  input dimension (must equal out_features)
        out_features: output dimension (must equal in_features)
        tt_rank:      TT-rank connecting cores (default 16, gives ~8x compression)
        bias:         whether to include a bias term
    """

    def __init__(self, in_features, out_features, tt_rank=16, bias=True):
        super().__init__()
        assert in_features == out_features, (
            f"TTLinear requires square weight matrix; got "
            f"in={in_features}, out={out_features}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank

        # Find factorization into roughly square dimensions.
        # For d=256: factor=16, so we have 2 cores (d^2 = factor^4).
        self.factor = int(math.isqrt(in_features))
        if self.factor * self.factor != in_features:
            raise ValueError(
                f"in_features ({in_features}) must be a perfect square"
            )

        # TT cores.
        # Core 1: contracts with input dim 0  (j1, i1, r)
        # Core 2: contracts with input dim 1  (r,  j2, i2)
        self.core1 = nn.Parameter(torch.empty(self.factor, self.factor, tt_rank))
        self.core2 = nn.Parameter(torch.empty(tt_rank, self.factor, self.factor))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self):
        """muP-style initialisation: each core ~ N(0, 1/sqrt(rank * factor))."""
        std = 1.0 / math.sqrt(self.tt_rank * self.factor)
        nn.init.normal_(self.core1, std=std)
        nn.init.normal_(self.core2, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x):
        """Forward pass through the TT-decomposed weight matrix.

        Args:
            x:  (..., in_features)  input tensor

        Returns:
            y:  (..., out_features)  output tensor
        """
        # Save leading dims, then reshape to (batch, factor, factor).
        *batch, _ = x.shape
        x = x.reshape(-1, self.factor, self.factor)            # (B, i1, i2)

        # Contract through core 1:  (B, i, j) x (k, i, r) → (B, k, j, r)
        h = torch.einsum("b i j, k i r -> b k j r", x, self.core1)

        # Contract through core 2:  (B, k, j, r) x (r, l, j) → (B, k, l)
        y = torch.einsum("b k j r, r l j -> b k l", h, self.core2)

        y = y.reshape(*batch, self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def dense_weight(self):
        """Reconstruct the equivalent dense weight matrix (for inspection only).

        This materialises the full (out, in) matrix from the TT cores and is
        NOT used during training.  It is O(d^2) in memory.
        """
        with torch.no_grad():
            # G1: (j1, i1, r) → (j1, i1, r)
            # G2: (r,  j2, i2) → (r,  j2, i2)
            # W:  (j1*j2, i1*i2) = (out, in)
            w = torch.einsum(
                "k i r, r l j -> k l i j",
                self.core1, self.core2,
            )
            return w.reshape(self.out_features, self.in_features)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tt_rank={self.tt_rank}, factor={self.factor}"
        )


# ------------------------------------------------------------------
# Quick smoke-test (runs when the file is executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== TTLinear smoke test ===")

    batch, d, r = 4, 256, 16
    layer = TTLinear(d, d, tt_rank=r, bias=True)
    x = torch.randn(batch, 64, d)  # (batch, seq, d)

    y = layer(x)
    print(f" Input shape:  {x.shape}")
    print(f" Output shape: {y.shape}")
    print(f" Parameters:   {sum(p.numel() for p in layer.parameters()):,}")
    print(f" vs dense:     {d * d + d:,}")
    ratio = (d * d + d) / sum(p.numel() for p in layer.parameters())
    print(f" Compression:  {ratio:.1f}x")

    # Verify output is finite and non-zero.
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert torch.isfinite(y).all(), "Output contains NaN/Inf"
    assert not torch.allclose(y, torch.zeros_like(y)), "Output is all zeros"

    # Gradient check.
    loss = y.sum()
    loss.backward()
    for name, p in layer.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad for {name}"

    print(" All checks passed.")
