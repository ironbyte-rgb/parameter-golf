"""
Parameter Golf — TT+MLA+GQA Training Script.

Trains a sub-16MB language model on FineWeb-Edu with structural
TT + MLA + GQA compression baked in at initialisation time.

Single-GPU Colab A100 path.  Trains for a configurable number of steps.

Usage:
    python train_gpt.py

Environment variables (all optional):
    BATCH_SIZE       — sequences per step (default 64)
    SEQ_LEN          — sequence length (default 512)
    LR               — peak learning rate (default 6e-3)
    WARMUP_STEPS     — LR warmup steps (default 500)
    TRAIN_STEPS      — total training steps (default 10000)
    DECAY_STEPS      — LR decay steps at end (default 1000)
    EVAL_EVERY       — evaluate every N steps (default 1000)
    CHECKPOINT_EVERY — save checkpoint every N steps (default 2000)
    MAX_VAL_TOKENS   — validation tokens to use (default 200000)
    COMPILE_MODEL    — use torch.compile (default 1)
    OUTPUT_DIR       — checkpoint / artifact directory (default ./output)
"""

import math
import os
import time
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Local modules
from model import TTMLATransformer
from tokenizer_ import train_bpe_tokenizer, build_byte_luts, get_gpt2_tokenizer
from data_pipeline import create_dataloader
from eval import evaluate_model, evaluate_sliding_window, load_validation_tokens
from quantize import save_compressed_model, load_and_dequantize


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

def get_config():
    """Build config from environment variables with sensible defaults."""
    return {
        # Model
        "vocab_size": 50257,
        "d_model": 576,
        "n_layers": 20,
        "n_heads": 8,
        "n_kv_heads": 1,
        "d_head": 72,
        "d_c": 16,
        "ffn_dim": 2048,
        "tt_rank": 16,
        "max_seq": 512,

        # Training
        "batch_size": int(os.getenv("BATCH_SIZE", "1024")),
        "seq_len": int(os.getenv("SEQ_LEN", "512")),
        "lr": float(os.getenv("LR", "6e-3")),
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "warmup_steps": int(os.getenv("WARMUP_STEPS", "500")),
        "train_steps": int(os.getenv("TRAIN_STEPS", "10000")),
        "decay_steps": int(os.getenv("DECAY_STEPS", "1000")),

        # Evaluation
        "eval_every": int(os.getenv("EVAL_EVERY", "1000")),
        "checkpoint_every": int(os.getenv("CHECKPOINT_EVERY", "2000")),
        "max_val_tokens": int(os.getenv("MAX_VAL_TOKENS", "200000")),
        "sliding_window_stride": 64,

        # Performance
        "compile_model": bool(int(os.getenv("COMPILE_MODEL", "1"))),
        "dtype": torch.bfloat16,

        # Paths
        "data_dir": os.getenv("DATA_DIR", "./data/tokens"),
        "output_dir": Path(os.getenv("OUTPUT_DIR", "./output")),
        "tokenizer_path": Path("./tokenizer.json"),
    }


# ------------------------------------------------------------------
# Learning rate schedule (trapezoidal)
# ------------------------------------------------------------------

def get_lr(step, cfg):
    """Trapezoidal LR: warmup -> constant peak -> linear decay."""
    warmup = cfg["warmup_steps"]
    total = cfg["train_steps"]
    decay = cfg["decay_steps"]
    peak = cfg["lr"]
    min_lr = peak * 0.1

    if step < warmup:
        return peak * step / max(warmup, 1)
    elif step < total - decay:
        return peak
    else:
        frac = (step - (total - decay)) / max(decay, 1)
        return peak - (peak - min_lr) * frac


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(cfg):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    output_dir = cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    print("\n=== Step 1: Tokenizer ===")
    tokenizer_path = cfg["tokenizer_path"]
    if tokenizer_path.exists():
        print(f"Loading existing tokenizer from {tokenizer_path}")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        # Default: GPT-2 tokenizer (50K vocab, zero training cost)
        tokenizer = get_gpt2_tokenizer(save_path=tokenizer_path)

    actual_vocab = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {actual_vocab}")
    luts = build_byte_luts(tokenizer, vocab_size=max(cfg["vocab_size"], actual_vocab))
    print("Byte LUTs built.")

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    print("\n=== Step 2: Model ===")
    model = TTMLATransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        d_head=cfg["d_head"],
        d_c=cfg["d_c"],
        ffn_dim=cfg["ffn_dim"],
        tt_rank=cfg["tt_rank"],
        max_seq=cfg["max_seq"],
    )
    model = model.to(device)

    total, _ = model.count_parameters()
    model_size_mb = total * 2 / 1e6
    print(f"Parameters: {total:,}  ({model_size_mb:.2f} MB at BF16)")

    model_raw = model  # keep uncompiled ref for roundtrip validation
    if cfg["compile_model"] and device.type == "cuda":
        print("Compiling model with torch.compile (reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")

    # ------------------------------------------------------------------
    # 3. Optimizer
    # ------------------------------------------------------------------
    print("\n=== Step 3: Optimizer ===")
    decay_params = []
    nodecay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.dim() < 2
            or "norm" in name
            or "scale" in name
            or "embedding" in name
        ):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": cfg["weight_decay"]},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    print(f"Optimizer: AdamW, lr={cfg['lr']}, wd={cfg['weight_decay']}")
    print(f"  Decay params:    {sum(p.numel() for p in decay_params):,}")
    print(f"  No-decay params: {sum(p.numel() for p in nodecay_params):,}")

    # ------------------------------------------------------------------
    # 4. Data pipeline
    # ------------------------------------------------------------------
    print("\n=== Step 4: Data pipeline ===")
    train_loader = create_dataloader(
        tokenizer=tokenizer,
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        data_dir=cfg["data_dir"],
        num_workers=4,
        prefetch_factor=4,
    )
    print(f"Batch: {cfg['batch_size']}, Seq: {cfg['seq_len']}, "
          f"Tokens/step: {cfg['batch_size'] * cfg['seq_len']:,}")

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    print(f"\n=== Step 5: Training ({cfg['train_steps']} steps) ===\n")

    model.train()
    step = 0
    train_loss_accum = 0.0
    train_steps_accum = 0
    tokens_seen = 0
    best_val_bpb = float("inf")
    t_start = time.time()

    train_iter = iter(train_loader)

    while step < cfg["train_steps"]:
        current_lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        try:
            input_ids, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, targets = next(train_iter)

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.amp.autocast("cuda", dtype=cfg["dtype"]):
            loss = model(input_ids, return_loss=True, targets=targets)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["grad_clip"]
        )
        optimizer.step()

        train_loss_accum += loss.item()
        train_steps_accum += 1
        tokens_seen += input_ids.numel()

        if step % 100 == 0 and step > 0:
            avg_loss = train_loss_accum / train_steps_accum
            elapsed = time.time() - t_start
            tok_per_sec = tokens_seen / max(elapsed, 0.001)
            print(
                f"  step {step:>6d}/{cfg['train_steps']} | "
                f"loss {avg_loss:.4f} | "
                f"lr {current_lr:.2e} | "
                f"grad {grad_norm:.2f} | "
                f"{tok_per_sec/1e6:.1f}M tok/s | "
                f"{elapsed:.0f}s"
            )
            train_loss_accum = 0.0
            train_steps_accum = 0

        if step > 0 and step % cfg["eval_every"] == 0:
            val_loss, val_bpb = run_eval(
                model, tokenizer, luts, device, cfg
            )
            print(
                f"  >>> EVAL step {step}: "
                f"val_loss={val_loss:.4f}  val_bpb={val_bpb:.4f} <<<"
            )
            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                ckpt_path = output_dir / "best_model.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  >>> New best val_bpb: {val_bpb:.4f} (saved) <<<")
            model.train()

        if step > 0 and step % cfg["checkpoint_every"] == 0:
            ckpt_path = output_dir / f"checkpoint_step{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_bpb": best_val_bpb,
                "tokens_seen": tokens_seen,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        step += 1

    # ------------------------------------------------------------------
    # 6. Final evaluation
    # ------------------------------------------------------------------
    print(f"\n=== Step 6: Final evaluation ===")
    total_time = time.time() - t_start
    print(f"Total training time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total tokens seen: {tokens_seen:,}")

    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.exists():
        print(f"Loading best checkpoint: {best_ckpt}")
        model_raw.load_state_dict(torch.load(best_ckpt, map_location=device))

    eval_model = model_raw  # use uncompiled for reliable weight loading
    val_loss, val_bpb = run_eval(
        eval_model, tokenizer, luts, device, cfg, sliding=False
    )
    print(f"\nFinal val_loss: {val_loss:.4f}")
    print(f"Final val_bpb:  {val_bpb:.4f}")

    sw_loss, sw_bpb = run_eval(
        eval_model, tokenizer, luts, device, cfg, sliding=True
    )
    print(f"Sliding-window val_bpb (stride={cfg['sliding_window_stride']}): {sw_bpb:.4f}")

    # ------------------------------------------------------------------
    # 7. Quantize & compress
    # ------------------------------------------------------------------
    print(f"\n=== Step 7: Quantize & compress ===")
    ptz_path = output_dir / "final_model.int8.ptz"
    info = save_compressed_model(
        model, ptz_path, code_path=Path(__file__)
    )

    print("\nRoundtrip validation...")
    dequant_sd = load_and_dequantize(ptz_path)
    # Must use uncompiled model — torch.compile caches weights in the graph
    model_raw.load_state_dict(dequant_sd, strict=True)

    rt_loss, rt_bpb = run_eval(
        model_raw, tokenizer, luts, device, cfg, sliding=False
    )
    print(f"Roundtrip val_loss: {rt_loss:.4f}")
    print(f"Roundtrip val_bpb:  {rt_bpb:.4f}")
    print(f"Roundtrip exact:    {rt_loss:.8f}  {rt_bpb:.8f}")

    limit = 16_000_000
    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best val_bpb:      {best_val_bpb:.4f}")
    print(f"  Roundtrip val_bpb: {rt_bpb:.4f}")
    print(f"  Artifact size:     {info['bytes_total']:,} bytes "
          f"({info['bytes_total']/1e6:.2f} MB)")
    print(f"  Under 16MB:        {info['bytes_total'] < limit}")
    print(f"  Training time:     {total_time:.0f}s")
    print(f"  Tokens seen:       {tokens_seen:,}")
    print(f"  Output dir:        {output_dir}")
    print(f"{'='*60}")


# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------

@torch.no_grad()
def run_eval(model, tokenizer, luts, device, cfg, sliding=False):
    """Run evaluation and return (val_loss, val_bpb)."""
    val_tokens = load_validation_tokens(tokenizer, cfg["max_val_tokens"])

    if sliding:
        return evaluate_sliding_window(
            model, val_tokens, luts, cfg["seq_len"],
            cfg["sliding_window_stride"], device,
        )
    else:
        return evaluate_model(
            model, val_tokens, luts, cfg["seq_len"], device,
        )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    cfg = get_config()
    print("=" * 60)
    print("Parameter Golf — TT+MLA+GQA Training")
    print("=" * 60)
    for k, v in cfg.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")
    print("=" * 60)
    train(cfg)
