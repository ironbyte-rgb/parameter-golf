"""
FineWeb-Edu streaming data pipeline.

Streams FineWeb-Edu from HuggingFace datasets, tokenises on-the-fly with
our BPE tokenizer, and yields packed sequences for training.  Built for
Colab A100 — needs to sustain >1M tokens/sec to keep the GPU fed.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader


class FineWebDataset(IterableDataset):
    """Streaming FineWeb-Edu dataset with on-the-fly tokenisation.

    Yields (input_ids, targets) pairs where:
      - input_ids: (seq_len,)   — token sequence
      - targets:   (seq_len,)   — same sequence shifted right by 1

    Documents are concatenated with an <s> separator.  Sequences never
    cross a document boundary without the separator, preventing the
    model from attending across unrelated documents.

    Args:
        tokenizer:       a HuggingFace ``tokenizers.Tokenizer``
        seq_len:         sequence length (512)
        split:           HF dataset split (default "train")
        buffer_size:     tokens to buffer before yielding batches
        max_docs:        max documents to stream (None = unlimited)
    """

    def __init__(
        self,
        tokenizer,
        seq_len=512,
        split="train",
        buffer_size=1_000_000,
        max_docs=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.buffer_size = buffer_size
        self.max_docs = max_docs

        # Special token ids
        self.bos_id = tokenizer.token_to_id("<s>") or 0
        self.eos_id = tokenizer.token_to_id("</s>") or 2
        self.pad_id = tokenizer.token_to_id("<pad>") or 3

    # ------------------------------------------------------------------
    # Internal: token buffer management
    # ------------------------------------------------------------------

    def _tokenize_doc(self, text):
        """Tokenize a single document, return list of token ids."""
        encoded = self.tokenizer.encode(text)
        return encoded.ids

    # ------------------------------------------------------------------

    def __iter__(self):
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            streaming=True,
            split=self.split,
        )

        buffer = []          # flat list of token ids
        doc_count = 0

        for row in ds:
            text = row["text"]
            if not text or len(text.strip()) < 10:
                continue

            tokens = self._tokenize_doc(text)
            if len(tokens) < 4:
                continue

            # Add document tokens with boundary markers
            buffer.append(self.bos_id)
            buffer.extend(tokens)
            buffer.append(self.eos_id)

            doc_count += 1
            if self.max_docs and doc_count >= self.max_docs:
                break

            # Yield sequences once buffer has enough tokens
            while len(buffer) >= self.seq_len + 1:
                # Take seq_len + 1 tokens (for input + shifted target)
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]  # shift by seq_len

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, targets

        # Yield any remaining tokens at end of dataset
        while len(buffer) >= self.seq_len + 1:
            chunk = buffer[: self.seq_len + 1]
            buffer = buffer[self.seq_len :]
            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            targets = torch.tensor(chunk[1:], dtype=torch.long)
            yield input_ids, targets


def create_dataloader(
    tokenizer,
    seq_len=512,
    batch_size=64,
    split="train",
    num_workers=2,
    prefetch_factor=4,
    max_docs=None,
):
    """Create a DataLoader for FineWeb-Edu streaming.

    Args:
        tokenizer:         HuggingFace ``tokenizers.Tokenizer``
        seq_len:           sequence length
        batch_size:        sequences per batch
        split:             "train" or validation subset
        num_workers:       DataLoader worker processes
        prefetch_factor:   batches to prefetch per worker
        max_docs:          max documents (None = unlimited)

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = FineWebDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        split=split,
        max_docs=max_docs,
    )

    # Collate function: stack individual sequences into batches
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return input_ids, targets

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
    )

    return loader


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Data pipeline smoke test ===\n")

    from tokenizers import Tokenizer

    # Quick test with a pre-built tiny tokenizer
    # (same as tokenizer_.py smoke test)
    tokenizer_path = "/tmp/_test_tokenizer.json"
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception:
        # Build one on-the-fly
        from tokenizers import models, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=True
        )
        trainer_obj = trainers.BpeTrainer(
            vocab_size=512,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            min_frequency=1,
        )
        corpus = [
            "hello world this is a test corpus for training",
            "machine learning transformers attention neural networks",
            "deep learning models use gradient descent optimization",
        ] * 100
        tokenizer.train_from_iterator(corpus, trainer_obj)
        tokenizer.save(tokenizer_path)

    print(f" Vocab size: {tokenizer.get_vocab_size()}")

    # Test with a small number of real FineWeb documents
    loader = create_dataloader(
        tokenizer=tokenizer,
        seq_len=512,
        batch_size=4,
        num_workers=0,   # must be 0 in __main__ on Windows
        max_docs=200,
    )

    print(" Streaming FineWeb-Edu (200 docs max)...")
    batch_count = 0
    tokens_seen = 0

    import time
    t0 = time.time()

    for input_ids, targets in loader:
        batch_count += 1
        tokens_seen += input_ids.numel()
        assert input_ids.shape == targets.shape
        assert input_ids.shape[1] == 512
        if batch_count <= 2:
            print(f"   Batch {batch_count}: shape={input_ids.shape}, "
                  f"ids range=[{input_ids.min().item()}, {input_ids.max().item()}]")
        if batch_count >= 50:
            break

    elapsed = time.time() - t0
    tok_per_sec = tokens_seen / elapsed
    print(f"\n Batches:     {batch_count}")
    print(f" Tokens:      {tokens_seen:,}")
    print(f" Time:        {elapsed:.1f}s")
    print(f" Throughput:  {tok_per_sec:,.0f} tok/sec")
    print(" All checks passed.")
