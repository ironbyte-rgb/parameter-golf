# Parameter Golf — TT+MLA+GQA

Structural compression language model for the OpenAI Parameter Golf Challenge.

**Architecture:** 8-layer transformer with Tensor Train Q/O projections,
Multi-head Latent Attention (MLA) KV compression, Grouped-Query Attention
(GQA, 8:1 ratio), and SwiGLU FFN. ~4.37M parameters (~8.74 MB at BF16).

## Quick start (Colab Pro A100)

1. Open `run_colab.ipynb` in Google Colab
2. Run all cells
3. Training takes 2-4 hours on A100
4. Output: `final_model.int8.ptz` (<2 MB compressed)

## Local testing

```bash
pip install -r requirements.txt
python tt_layers.py      # verify TT decomposition
python mla_attention.py  # verify MLA attention
python model.py          # verify full model
python tokenizer_.py     # verify tokenizer
python eval.py           # verify evaluation
python quantize.py       # verify quantization
```

## Architecture

| Component | Detail |
|---|---|
| Vocab | 4096 (BPE) |
| d_model | 256 |
| Layers | 8 |
| Q heads / KV heads | 8 / 1 (GQA 8:1) |
| MLA latent | 16 |
| FFN | SwiGLU, 512 dim |
| TT rank | 16 (8x compression on Q/O) |
| Total params | 4,370,704 |

## Files

- `model.py` — TTMLATransformer model definition
- `tt_layers.py` — Tensor Train decomposed linear layer
- `mla_attention.py` — Multi-head Latent Attention
- `tokenizer_.py` — BPE tokenizer + byte LUT builder
- `data_pipeline.py` — FineWeb-Edu streaming
- `eval.py` — BPB evaluation protocol
- `quantize.py` — int8 + zlib compression
- `train_gpt.py` — Training script
- `run_colab.ipynb` — Colab notebook
