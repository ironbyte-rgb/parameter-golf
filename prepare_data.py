"""
Pre-tokenize FineWeb-Edu for fast training.

Downloads a chunk of FineWeb-Edu, tokenises in bulk, and saves as
uint16 .pt shards (~100M tokens each).  The training loop then loads
these with zero I/O overhead — no streaming, no on-the-fly tokenisation.

Usage:
    python prepare_data.py                    # default: ~1B tokens
    python prepare_data.py --num_tokens 5e9   # ~5B tokens
    python prepare_data.py --tokenizer ./tokenizer.json
"""

import argparse
import os
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize FineWeb-Edu")
    parser.add_argument("--num_tokens", type=float, default=1e9,
                        help="Target number of tokens (default 1e9 = 1B)")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer.json",
                        help="Path to trained BPE tokenizer")
    parser.add_argument("--output_dir", type=str, default="./data/tokens",
                        help="Output directory for .pt shards")
    parser.add_argument("--shard_size", type=int, default=100_000_000,
                        help="Tokens per shard (default 100M)")
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="Documents per batch for encoding")
    args = parser.parse_args()

    num_tokens = int(args.num_tokens)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}.  Train it first with "
            f"tokenizer_.py or run the notebook cell 4."
        )
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()
    bos_id = tokenizer.token_to_id("<s>") or 0
    eos_id = tokenizer.token_to_id("</s>") or 2
    print(f"Tokenizer loaded: vocab={vocab_size}, bos={bos_id}, eos={eos_id}")

    # ------------------------------------------------------------------
    # Stream & tokenise
    # ------------------------------------------------------------------
    print(f"Streaming FineWeb-Edu (target: {num_tokens:,} tokens)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        streaming=True,
        split="train",
    )

    all_tokens = []
    total = 0
    batch_texts = []
    t_start = time.time()
    shard_idx = 0

    def save_shard(tokens_list):
        nonlocal shard_idx
        flat = torch.tensor(tokens_list, dtype=torch.uint16)
        path = output_dir / f"fineweb_edu_{shard_idx:04d}.pt"
        torch.save(flat, path)
        size_mb = path.stat().st_size / 1e6
        print(f"  Saved {path} — {flat.numel():,} tokens ({size_mb:.1f} MB)")
        shard_idx += 1
        return []

    for row in ds:
        text = row["text"]
        if not text or len(text.strip()) < 10:
            continue

        batch_texts.append(text)

        if len(batch_texts) >= args.batch_size:
            # Batch-encode for speed
            encoded = tokenizer.encode_batch(batch_texts)
            for doc in encoded:
                ids = doc.ids
                if len(ids) < 4:
                    continue
                all_tokens.append(bos_id)
                all_tokens.extend(ids)
                all_tokens.append(eos_id)
                total += len(ids) + 2

            batch_texts = []

            # Save shard when we hit the limit
            while len(all_tokens) >= args.shard_size:
                shard_tokens = all_tokens[: args.shard_size]
                all_tokens = all_tokens[args.shard_size :]
                all_tokens = save_shard(shard_tokens)

            # Progress
            if total >= num_tokens:
                break

            if total % 10_000_000 < args.batch_size * 100:
                elapsed = time.time() - t_start
                rate = total / max(elapsed, 0.001)
                print(f"  {total:,} tokens  ({rate/1e6:.1f}M tok/s)  [{elapsed:.0f}s]")

    # Save remaining
    if all_tokens:
        save_shard(all_tokens)

    elapsed = time.time() - t_start
    print(f"\nDone: {total:,} tokens in {elapsed:.0f}s ({total/max(elapsed,0.001)/1e6:.1f}M tok/s)")
    print(f"Shards saved to {output_dir}/")
    print(f"Total shards: {shard_idx}")


if __name__ == "__main__":
    main()
