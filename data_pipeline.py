"""
Data pipelines for training.

Two modes:
  1. PreTokenizedDataset  — loads pre-tokenized .pt shards (FAST, 10M+ tok/s)
  2. FineWebDataset       — streams from HF + tokenises on-the-fly (fallback)

Always prefer mode 1.  Run prepare_data.py first to build the shards.
"""

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pathlib import Path


# ------------------------------------------------------------------
# FAST PATH: pre-tokenized .pt shards
# ------------------------------------------------------------------

class PreTokenizedDataset(Dataset):
    """Map-style dataset over pre-tokenized uint16 .pt shards.

    Loads all shards into a single contiguous buffer in RAM.  For datasets
    larger than available RAM, set ``mmap=True`` to use memory-mapped files
    instead (slightly slower but handles 10B+ tokens).

    Yields (input_ids, targets) pairs of shape (seq_len,).
    """

    def __init__(self, data_dir, seq_len=512, mmap=False):
        self.seq_len = seq_len
        data_dir = Path(data_dir)

        shards = sorted(data_dir.glob("fineweb_edu_*.pt"))
        if not shards:
            raise FileNotFoundError(
                f"No fineweb_edu_*.pt shards found in {data_dir}.  "
                f"Run prepare_data.py first."
            )

        print(f"Loading {len(shards)} pre-tokenized shard(s)...")

        if mmap:
            # Memory-map each shard (zero-copy, virtual memory backed by disk)
            self._tensors = [
                torch.load(str(s), map_location="cpu", weights_only=True)
                for s in shards
            ]
            # Build index for fast lookup without concatenating
            self._cumsum = [0]
            for t in self._tensors:
                self._cumsum.append(self._cumsum[-1] + t.numel())
            self._total = self._cumsum[-1]
            self._mmap = True
        else:
            # Load everything into one contiguous buffer (fastest)
            tensors = [
                torch.load(str(s), map_location="cpu", weights_only=True)
                for s in shards
            ]
            total_elems = sum(t.numel() for t in tensors)
            self._data = torch.empty(total_elems, dtype=torch.uint16)
            offset = 0
            for t in tensors:
                self._data[offset : offset + t.numel()] = t
                offset += t.numel()
            self._total = total_elems
            self._mmap = False

        self.n_sequences = (self._total - 1) // seq_len
        total_mb = self._total * 2 / 1e6
        print(f"  {self._total:,} tokens ({total_mb:.0f} MB), "
              f"{self.n_sequences:,} sequences")

    def _get_token(self, idx):
        """O(1) token lookup for both mmap and contiguous modes."""
        if self._mmap:
            # Binary search for the right shard
            import bisect
            shard = bisect.bisect_right(self._cumsum, idx) - 1
            offset = idx - self._cumsum[shard]
            return self._tensors[shard][offset]
        else:
            return self._data[idx]

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        if self._mmap:
            # Slower path: gather token by token
            chunk = torch.tensor(
                [int(self._get_token(i)) for i in range(start, end)],
                dtype=torch.long,
            )
        else:
            chunk = self._data[start:end].long()
        return chunk[:-1], chunk[1:]


# ------------------------------------------------------------------
# FALLBACK: HF streaming (slow — use only if pre-tokenized data unavailable)
# ------------------------------------------------------------------

class FineWebDataset(IterableDataset):
    """Streaming FineWeb-Edu with on-the-fly tokenisation (FALLBACK)."""

    def __init__(self, tokenizer, seq_len=512, split="train", max_docs=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.max_docs = max_docs
        self.bos_id = tokenizer.token_to_id("<s>") or 0
        self.eos_id = tokenizer.token_to_id("</s>") or 2

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", "sample-10BT",
            streaming=True, split=self.split,
        )
        buffer, doc_count = [], 0
        for row in ds:
            text = row["text"]
            if not text or len(text.strip()) < 10:
                continue
            tokens = self.tokenizer.encode(text).ids
            if len(tokens) < 4:
                continue
            buffer.append(self.bos_id)
            buffer.extend(tokens)
            buffer.append(self.eos_id)
            doc_count += 1
            if self.max_docs and doc_count >= self.max_docs:
                break
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:], dtype=torch.long))
        while len(buffer) >= self.seq_len + 1:
            chunk = buffer[: self.seq_len + 1]
            buffer = buffer[self.seq_len :]
            yield (torch.tensor(chunk[:-1], dtype=torch.long),
                   torch.tensor(chunk[1:], dtype=torch.long))


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_dataloader(
    tokenizer=None,
    seq_len=512,
    batch_size=64,
    data_dir="./data/tokens",
    num_workers=4,
    prefetch_factor=4,
    max_docs=None,
    mmap=False,
):
    """Create the best available DataLoader.

    If ``data_dir`` contains pre-tokenized .pt shards, uses the fast
    PreTokenizedDataset.  Otherwise falls back to HF streaming.

    Args:
        tokenizer:       HuggingFace tokenizer (only needed for fallback)
        seq_len:         sequence length
        batch_size:      sequences per batch
        data_dir:        path to pre-tokenized .pt shards
        num_workers:     DataLoader workers (0 = main process only)
        prefetch_factor: batches to prefetch per worker
        max_docs:        max documents for fallback (None = unlimited)
        mmap:            use memory-mapped files for fast loader

    Returns:
        torch.utils.data.DataLoader
    """
    data_dir = Path(data_dir)

    if list(data_dir.glob("fineweb_edu_*.pt")):
        print(f"Using pre-tokenized data from {data_dir}")
        dataset = PreTokenizedDataset(data_dir, seq_len=seq_len, mmap=mmap)
    elif tokenizer is not None:
        print("No pre-tokenized data found, using HF streaming (SLOW).")
        print("Run prepare_data.py for 10-30x faster training.")
        dataset = FineWebDataset(
            tokenizer=tokenizer, seq_len=seq_len, max_docs=max_docs,
        )
    else:
        raise RuntimeError(
            "No pre-tokenized data found and no tokenizer provided.  "
            "Either run prepare_data.py or pass a tokenizer."
        )

    def collate_fn(batch):
        return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=True,
    )


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import time, sys

    print("=== Data pipeline smoke test ===\n")
    data_dir = Path("./data/tokens")

    # Check if pre-tokenized data exists
    shards = list(data_dir.glob("fineweb_edu_*.pt"))
    if shards:
        print(f"Found {len(shards)} pre-tokenized shard(s)")
        loader = create_dataloader(
            seq_len=512, batch_size=128, data_dir=data_dir,
            num_workers=0, mmap=False,
        )
        print(f"Dataset: {len(loader.dataset):,} sequences")

        t0 = time.time()
        tokens_seen = 0
        for i, (x, y) in enumerate(loader):
            tokens_seen += x.numel()
            if i >= 50:
                break
        elapsed = time.time() - t0
        print(f"  {tokens_seen:,} tokens in {elapsed:.1f}s "
              f"({tokens_seen / max(elapsed, 0.001) / 1e6:.1f}M tok/s)")
        print("  Pre-tokenized pipeline OK!")
    else:
        print("No pre-tokenized data found.  Testing fallback...")
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        trainer_obj = trainers.BpeTrainer(
            vocab_size=512,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            min_frequency=1,
        )
        corpus = ["hello world test corpus for training"] * 100
        tok.train_from_iterator(corpus, trainer_obj)

        loader = create_dataloader(
            tokenizer=tok, seq_len=512, batch_size=4,
            num_workers=0, max_docs=200,
        )
        t0 = time.time()
        tokens_seen = 0
        for i, (x, y) in enumerate(loader):
            tokens_seen += x.numel()
            if i >= 50:
                break
        elapsed = time.time() - t0
        print(f"  {tokens_seen:,} tokens in {elapsed:.1f}s "
              f"({tokens_seen / max(elapsed, 0.001) / 1e6:.1f}M tok/s)")
        print("  Fallback pipeline OK!")
