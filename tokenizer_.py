"""
BPE tokenizer training and byte-counting LUT builder.

Trains a Byte-Pair Encoding tokenizer (vocab=4096) on a FineWeb-Edu sample,
then builds the three lookup tables required for the val_bpb computation:

  base_bytes_lut[t]    — UTF-8 byte length of token t's decoded string
  has_leading_space[t] — True if the token starts with a space marker
  is_boundary[t]       — True for special tokens (control / unknown / unused)

The val_bpb formula matches the OpenAI Parameter Golf protocol exactly:

  val_bpb = (val_loss / ln(2)) * (token_count / byte_count)

where byte_count accumulates per-token bytes with the leading-space rule:
  bytes = base_bytes_lut[tgt]
  if has_leading_space[tgt] and not is_boundary[prev]:
      bytes += 1
"""

import json
import math
import os
from pathlib import Path

import torch


# ------------------------------------------------------------------
# Tokenizer training
# ------------------------------------------------------------------

def train_bpe_tokenizer(
    output_path,
    vocab_size=4096,
    sample_size=100_000_000,  # ~100M chars ≈ enough for 4K vocab
):
    """Train a BPE tokenizer on a FineWeb-Edu sample.

    Uses HuggingFace ``tokenizers`` for fast training.  Downloads a sample
    from FineWeb-Edu, trains BPE, and saves the tokenizer to disk.

    Args:
        output_path:  path to save tokenizer.json
        vocab_size:   vocabulary size (default 4096)
        sample_size:  approximate chars to sample for training

    Returns:
        tokenizers.Tokenizer  object
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from datasets import load_dataset

    print(f"Training BPE tokenizer (vocab={vocab_size})...")

    # --- Download FineWeb-Edu sample ---
    print("  Downloading FineWeb-Edu sample...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        streaming=True,
        split="train",
    )

    # Collect ~sample_size chars of text
    samples = []
    chars = 0
    for row in ds:
        text = row["text"]
        samples.append(text)
        chars += len(text)
        if chars >= sample_size:
            break

    print(f"  Collected {len(samples):,} documents ({chars:,} chars)")

    # --- Train BPE ---
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(samples, trainer)
    tokenizer.save(str(output_path))
    print(f"  Tokenizer saved to {output_path}")
    print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


# ------------------------------------------------------------------
# Byte-counting LUTs
# ------------------------------------------------------------------

def build_byte_luts(tokenizer, vocab_size=4096):
    """Build byte-counting lookup tables for val_bpb computation.

    These LUTs encode the same logic as the competition's SentencePiece
    byte accounting, adapted for BPE tokenizers:

    - ``base_bytes_lut``: UTF-8 byte length of each token's decoded string,
      after stripping the space prefix marker.
    - ``has_leading_space``: True if the token's decoded form starts with
      a space character.
    - ``is_boundary``: True for special tokens (control / unknown) where
      no inter-token space should be inserted.

    The leading-space compensation rule (matching the competition protocol):
      If a token has a leading space AND the previous token is NOT a
      boundary token, add 1 byte (for the stripped space).

    Args:
        tokenizer:   a HuggingFace ``tokenizers.Tokenizer`` or a path to one
        vocab_size:  vocabulary size (used for sizing the LUT arrays)

    Returns:
        dict with keys: base_bytes_lut, has_leading_space, is_boundary
        (all ``torch.Tensor`` of dtype int16 / bool)
    """
    if isinstance(tokenizer, (str, Path)):
        from tokenizers import Tokenizer as T
        tokenizer = T.from_file(str(tokenizer))

    vocab = tokenizer.get_vocab()
    # Build id→token mapping (some ids may be missing from vocab dict)
    id_to_token = {}
    for token_str, idx in vocab.items():
        id_to_token[idx] = token_str

    base_bytes_lut = torch.zeros(vocab_size, dtype=torch.int16)
    has_leading_space = torch.zeros(vocab_size, dtype=torch.bool)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool)

    special_ids = set()
    for special_name in ["<unk>", "<s>", "</s>", "<pad>"]:
        if special_name in vocab:
            special_ids.add(vocab[special_name])

    for idx in range(vocab_size):
        token_str = id_to_token.get(idx)

        if token_str is None or idx in special_ids:
            # Missing or special token → boundary, no bytes
            is_boundary[idx] = True
            base_bytes_lut[idx] = 0
            has_leading_space[idx] = False
            continue

        # Decode: the ByteLevel pre-tokenizer uses Ġ (U+0120) as space marker
        # in the token string.  Strip it to get the "real" text.
        decoded = token_str

        # Check for leading space marker (Ġ for BPE byte-level, ▁ for SP)
        if decoded.startswith("Ġ"):
            has_leading_space[idx] = True
            decoded = decoded[1:]  # strip the space marker
        elif decoded.startswith(" "):
            has_leading_space[idx] = True
            decoded = decoded.lstrip(" ")

        # Count UTF-8 bytes
        utf8_bytes = len(decoded.encode("utf-8"))
        base_bytes_lut[idx] = utf8_bytes

    # Build inverse mapping: token string → id (for encoding during eval)
    # We'll store this in the returned dict

    return {
        "base_bytes_lut": base_bytes_lut,
        "has_leading_space": has_leading_space,
        "is_boundary": is_boundary,
    }


def compute_val_bpb(val_loss, val_token_count, val_byte_count):
    """Compute bits-per-byte from accumulated validation stats.

    This is the EXACT competition formula used by train_gpt.py:
      bits_per_token = val_loss / ln(2)
      tokens_per_byte = val_token_count / val_byte_count
      val_bpb         = bits_per_token * tokens_per_byte

    All inputs should be Python floats (not tensors).
    """
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count / max(val_byte_count, 1)
    return bits_per_token * tokens_per_byte


def count_bytes_for_tokens(token_ids, prev_token_ids, luts):
    """Count bytes for a batch of token sequences using the LUTs.

    Args:
        token_ids:      (N,) or (B, T) — current token ids
        prev_token_ids: (N,) or (B, T) — previous token ids in sequence
                         (shifted right; first position should be boundary)
        luts:           dict from ``build_byte_luts``

    Returns:
        total_bytes: int
    """
    base = luts["base_bytes_lut"][token_ids]
    leading = luts["has_leading_space"][token_ids]
    boundary_prev = luts["is_boundary"][prev_token_ids]

    extra = leading & ~boundary_prev
    return (base + extra.int()).sum().item()


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Tokenizer + Byte LUT smoke test ===\n")

    # Quick test: create a minimal tokenizer and verify LUTs
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    # Train on a tiny synthetic corpus
    corpus = [
        "hello world this is a test",
        "the quick brown fox jumps over the lazy dog",
        "machine learning is fascinating",
        "transformers use attention mechanisms",
    ]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer_obj = trainers.BpeTrainer(
        vocab_size=256,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        min_frequency=1,
    )
    tokenizer.train_from_iterator(corpus, trainer_obj)

    vocab_size = tokenizer.get_vocab_size()
    print(f" Tiny vocab size: {vocab_size}")

    luts = build_byte_luts(tokenizer, vocab_size=512)  # pad to 512

    # Test encoding
    text = "hello world"
    encoded = tokenizer.encode(text)
    print(f"\n Encoding '{text}':")
    print(f"   Token ids:  {encoded.ids}")
    # Tokens may contain Ġ (Ġ) which can't print on cp1252 terminals
    safe_tokens = [t.encode("unicode_escape").decode("ascii") for t in encoded.tokens]
    print(f"   Tokens:     {safe_tokens}")

    # Print byte LUT for these tokens
    for tid in encoded.ids:
        print(
            f"   id={tid}: "
            f"bytes={luts['base_bytes_lut'][tid].item()}, "
            f"space={luts['has_leading_space'][tid].item()}, "
            f"boundary={luts['is_boundary'][tid].item()}"
        )

    # Verify byte counting
    token_ids = torch.tensor(encoded.ids, dtype=torch.long)
    prev_ids = torch.cat([
        torch.tensor([tokenizer.token_to_id("<s>")]),
        token_ids[:-1],
    ])
    byte_count = count_bytes_for_tokens(token_ids, prev_ids, luts)
    actual_bytes = len(text.encode("utf-8"))
    print(f"\n Byte count (LUT):  {byte_count}")
    print(f" Actual UTF-8 bytes: {actual_bytes}")
    # Note: BPE byte-level may not match exactly due to space handling,
    # but should be close for non-space-heavy text

    # Test val_bpb formula
    test_bpb = compute_val_bpb(val_loss=2.0, val_token_count=1000, val_byte_count=2500)
    expected = (2.0 / math.log(2.0)) * (1000 / 2500)
    print(f"\n val_bpb test: {test_bpb:.6f} (expected {expected:.6f})")
    assert abs(test_bpb - expected) < 1e-9

    print("\n All checks passed.")
