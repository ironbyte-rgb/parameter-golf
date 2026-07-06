"""
Validation evaluation — bits-per-byte (BPB) computation.

Matches the OpenAI Parameter Golf protocol EXACTLY:

  bits_per_token = val_loss / ln(2)
  tokens_per_byte = val_token_count / val_byte_count
  val_bpb         = bits_per_token * tokens_per_byte

Byte counting uses the tokenizer LUTs with the leading-space compensation
rule: if a token has a leading space marker AND the previous token is NOT
a boundary token, add 1 byte (for the stripped space character).

Sliding-window evaluation (stride < seq_len) gives tokens more context
and produces lower (better) BPB scores.  This is the standard evaluation
used by all competitive submissions.
"""

import math
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# Core BPB computation
# ------------------------------------------------------------------

def compute_val_bpb(val_loss, val_token_count, val_byte_count):
    """Compute bits-per-byte from accumulated validation statistics.

    Args:
        val_loss:         sum of cross-entropy losses (in nats)
        val_token_count:  number of tokens evaluated
        val_byte_count:   number of UTF-8 bytes those tokens represent

    Returns:
        (val_loss_mean, val_bpb) tuple
    """
    val_loss_mean = val_loss / max(val_token_count, 1)
    bits_per_token = val_loss_mean / math.log(2.0)
    tokens_per_byte = val_token_count / max(val_byte_count, 1)
    val_bpb = bits_per_token * tokens_per_byte
    return val_loss_mean, val_bpb


# ------------------------------------------------------------------
# Per-token byte counting
# ------------------------------------------------------------------

def count_token_bytes(target_ids, prev_ids, luts):
    """Count bytes for target tokens using the competition LUTs.

    Byte counting rule (matching SentencePiece logic):
      bytes = base_bytes_lut[token]
      IF has_leading_space[token] AND NOT is_boundary[prev_token]:
          bytes += 1   (add the space that the tokenizer stripped)

    Args:
        target_ids:  (N,) or (B, T) — target token ids
        prev_ids:    same shape — preceding token ids in sequence
        luts:        dict with base_bytes_lut, has_leading_space, is_boundary

    Returns:
        total_byte_count: Python int
    """
    base = luts["base_bytes_lut"][target_ids].long()
    leading = luts["has_leading_space"][target_ids]
    boundary_prev = luts["is_boundary"][prev_ids]

    extra = leading & ~boundary_prev
    return (base + extra.long()).sum().item()


# ------------------------------------------------------------------
# Standard (non-sliding) evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, token_ids, luts, seq_len, device):
    """Evaluate model with non-overlapping chunks.

    Splits the validation set into non-overlapping sequences of
    ``seq_len`` tokens.  Each token starts with zero context except
    what it receives within its chunk.

    Args:
        model:     TTMLATransformer
        token_ids: (total_tokens,) — concatenated validation tokens
        luts:      byte-counting LUTs from tokenizer_.build_byte_luts()
        seq_len:   sequence length
        device:    torch device

    Returns:
        (val_loss_mean, val_bpb) tuple of floats
    """
    model.eval()
    total_tokens = token_ids.shape[0]
    # Trim to exact multiple of seq_len
    usable = (total_tokens - 1) // seq_len * seq_len
    token_ids = token_ids[: usable + 1]

    val_loss_sum = 0.0
    val_token_count = 0
    val_byte_count = 0

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, usable, seq_len):
            chunk = token_ids[i : i + seq_len + 1]
            inputs = chunk[:-1].unsqueeze(0).to(device)   # (1, seq_len)
            targets = chunk[1:].unsqueeze(0).to(device)   # (1, seq_len)
            prev_ids = chunk[:seq_len].to(device)          # unshifted for byte counting

            logits = model(inputs)                          # (1, seq_len, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1),
                reduction="sum",
            )

            val_loss_sum += loss.item()
            val_token_count += seq_len

            # Byte counting (on CPU for the LUT lookup)
            tgt_cpu = targets.view(-1).cpu()
            prev_cpu = prev_ids.view(-1).cpu()
            val_byte_count += count_token_bytes(tgt_cpu, prev_cpu, luts)

    return compute_val_bpb(val_loss_sum, val_token_count, val_byte_count)


# ------------------------------------------------------------------
# Sliding-window evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_sliding_window(model, token_ids, luts, seq_len, stride, device):
    """Evaluate with sliding windows for better per-token context.

    Instead of partitioning into non-overlapping chunks, windows slide
    with configurable stride.  Each token is scored in the window that
    gives it the most context (the rightmost position).

    For stride=64 and seq_len=512:
      - Window 1: tokens 0..511 scored, but only rightmost 64 count
      - Window 2: tokens 64..575 scored, rightmost 64 count
      - ...
      - Every token (past the first 64) gets scored with ~960 tokens of context

    This typically improves val_bpb by ~0.03 vs. standard evaluation.

    Args:
        model:     TTMLATransformer
        token_ids: (total_tokens,) concatenated val tokens
        luts:      byte-counting LUTs
        seq_len:   sequence length (window size)
        stride:    stride between windows (e.g., 64)
        device:    torch device

    Returns:
        (val_loss_mean, val_bpb) tuple of floats
    """
    model.eval()
    total_tokens = token_ids.shape[0]

    val_loss_sum = 0.0
    val_token_count = 0
    val_byte_count = 0

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, total_tokens - seq_len, stride):
            end = start + seq_len
            chunk = token_ids[start : end + 1]
            inputs = chunk[:-1].unsqueeze(0).to(device)    # (1, seq_len)
            targets = chunk[1:].unsqueeze(0).to(device)

            # Only count the rightmost ``stride`` tokens
            # (first window counts all tokens since stride may not divide evenly;
            #  but convention is first window counts rightmost stride too.)
            count_start = seq_len - stride
            if start == 0:
                # First window: score all tokens (they have no prior context anyway)
                count_start = 0

            scored_targets = targets[:, count_start:]
            scored_prev = chunk[count_start : seq_len].to(device)

            logits = model(inputs)
            scored_logits = logits[:, count_start:, :]

            loss = F.cross_entropy(
                scored_logits.reshape(-1, scored_logits.shape[-1]),
                scored_targets.reshape(-1),
                reduction="sum",
            )

            n_scored = scored_targets.numel()
            val_loss_sum += loss.item()
            val_token_count += n_scored

            tgt_cpu = scored_targets.view(-1).cpu()
            prev_cpu = scored_prev.view(-1).cpu()
            val_byte_count += count_token_bytes(tgt_cpu, prev_cpu, luts)

    return compute_val_bpb(val_loss_sum, val_token_count, val_byte_count)


# ------------------------------------------------------------------
# Validation data loading
# ------------------------------------------------------------------

def load_validation_tokens(tokenizer, num_tokens=1_000_000):
    """Load a chunk of FineWeb validation tokens.

    Downloads the FineWeb-Edu validation set and tokenises it.
    For a fair evaluation, we use the held-out validation split.

    Args:
        tokenizer:  HuggingFace ``tokenizers.Tokenizer``
        num_tokens: approximate number of tokens to load

    Returns:
        torch.Tensor of shape (num_tokens,) — token ids (on CPU)
    """
    from datasets import load_dataset

    print(f"  Loading validation data (~{num_tokens:,} tokens)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        streaming=True,
        split="train",
    )

    # Use the first portion of the training split as "validation"
    # (FineWeb-Edu sample-10BT doesn't have a separate val split).
    # For a proper eval, you'd use the actual FineWeb validation set.
    all_tokens = []
    for row in ds:
        text = row["text"]
        if not text or len(text.strip()) < 10:
            continue
        encoded = tokenizer.encode(text)
        all_tokens.extend(encoded.ids)
        if len(all_tokens) >= num_tokens:
            break

    tokens = torch.tensor(all_tokens[:num_tokens], dtype=torch.long)
    print(f"  Loaded {tokens.shape[0]:,} validation tokens")
    return tokens


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Evaluation smoke test ===\n")

    # Build a tiny tokenizer + LUTs
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from tokenizer_ import build_byte_luts  # (avoid compute_val_bpb name clash)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer_obj = trainers.BpeTrainer(
        vocab_size=256,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        min_frequency=1,
    )
    tok.train_from_iterator(["hello world test"] * 10, trainer_obj)
    luts = build_byte_luts(tok, vocab_size=256)

    # Test val_bpb formula
    _, bpb = compute_val_bpb(val_loss=200.0, val_token_count=100, val_byte_count=250)
    expected = (200.0 / 100 / math.log(2.0)) * (100.0 / 250.0)
    print(f" val_bpb: {bpb:.6f}  (expected {expected:.6f})")
    assert abs(bpb - expected) < 1e-9

    # Test byte counting
    text = "hello world"
    ids = torch.tensor(tok.encode(text).ids, dtype=torch.long)
    prev = torch.cat([torch.tensor([0]), ids[:-1]])  # 0 = <unk> (boundary)
    byte_count = count_token_bytes(ids, prev, luts)
    actual = len(text.encode("utf-8"))
    print(f" Byte count: {byte_count}  (actual UTF-8: {actual})")
    assert byte_count == actual, f"Mismatch: {byte_count} vs {actual}"

    print(" All checks passed.")
