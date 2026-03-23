# HyperSSM-1.58: Hypernetwork-Generated Selective SSM with Ternary Quantization

## Architecture

HyperSSM-1.58 combines three non-standard techniques into a novel language model architecture:

1. **Selective State Space Model (Mamba-style SSM)** replaces attention with O(T) linear-time sequence processing instead of O(T^2) quadratic attention.

2. **Hypernetwork Weight Generation** — A small ~230K-param BF16 seed network generates all SSM block weights on-the-fly via continuous depth indexing t in [0,1]. Stored low-rank U,V basis factors are modulated by depth-dependent vectors from the hypernetwork.

3. **Native 1.58-bit Ternary Quantization (BitLinear)** — weights in {-1, 0, +1} via absmean quantization with Straight-Through Estimator gradients. Progressive ternarization during training.

### Key Innovation: Continuous Depth via Hypernetwork

Instead of storing separate weights for each layer, a single hypernetwork generates layer-specific weights from a depth index t in [0,1]:
- Store shared low-rank factors U, V for each weight type
- Hypernetwork generates depth-dependent modulation vectors
- W(t) = U @ diag(modulation(t)) @ V
- Apply ternary quantization to generated weights

This allows:
- **Training depth (k=12)**: 12 Euler integration steps through the continuous ODE
- **Eval depth (k=30)**: 30 steps for higher quality at inference, with NO additional stored parameters

### Parameter Budget

| Component | Stored Params | Description |
|-----------|--------------|-------------|
| Token Embedding | 393,216 | 1024 x 384, tied to LM head |
| Hypernetwork MLP | ~82,000 | Backbone: 64->256->256 |
| Modulation Heads | ~50,000 | Per-weight-type depth modulation |
| Low-rank U,V Factors | ~250,000 | Shared basis for weight generation |
| SSM Block (norm, A, D) | ~14,000 | Shared across all depth steps |
| **Total Stored** | **~790,000** | **~790KB at INT8** |
| **Effective (k=30)** | **~16.6M** | 30x weight generation per forward pass |

### Model Configuration

- `d_model = 384`, `d_inner = 768` (2x expansion)
- `d_state = 16`, `dt_rank = 24`, `conv_kernel = 4`
- `vocab_size = 1024` (SP-1024 tokenizer)
- `train_seq_len = 512`, `hyper_rank = 48`

## Training Protocol

3-phase training over 10 minutes on 8xH100:

| Phase | Time | Depth | Ternary | LR | Description |
|-------|------|-------|---------|----| ------------|
| 1 | 0-90s | k=4 | 0% | warmup 0->1e-2 | BF16 warmup, shallow depth |
| 2 | 90-300s | 4->12 | 0%->100% | stable 1e-2 | Progressive ternarization |
| 3 | 300-580s | k=12 | 100% | WSD decay->0 | Full ternary sprint |

- Optimizer: AdamW fused, beta1=0.9, beta2=0.95, weight_decay=0.1, grad_clip=1.0
- Checkpoint saved at 580s (20s before deadline)

## Results

- **val_bpb**: TBD (requires 8xH100 training run)
- **Artifact size**: ~500KB compressed (well under 16MB limit)
- **Training time**: ~9.5 min on 8xH100

## Comparison to Baseline

| Metric | Baseline | HyperSSM-1.58 |
|--------|----------|---------------|
| Architecture | Transformer + GQA | SSM + Hypernetwork |
| Stored Params | ~15M | ~790K |
| Effective Params | ~15M | ~16.6M (at k=30) |
| Artifact Size | ~15.8MB | ~500KB (est.) |
| val_bpb | 1.2244 | TBD |

## Running

```bash
# Local test (CPU)
TEST_MODE=1 python train_gpt.py

# Full training (8xH100)
RUN_ID=hyperssm158 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Approach Justification

This submission explores a radically different architecture paradigm:
- **SSMs for compression**: State space models offer O(T) sequence processing, allowing longer effective context within the training time budget.
- **Hypernetwork for parameter efficiency**: By generating weights from a tiny seed, we achieve massive parameter sharing while maintaining depth-specific specialization.
- **Ternary quantization for compression**: Native 1.58-bit weights make the artifact extremely small, well under the 16MB limit.

The combination is novel and may not beat the highly-optimized transformer baselines, but it demonstrates an interesting point in the architecture design space for parameter-efficient language models.
