from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from tokenizer.bpe import BPETokenizer


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/corpus.txt"
OUT = ROOT / "data/processed"
VOCAB = OUT / "vocab.json"
DATASET = OUT / "dataset.pt"
META = OUT / "meta.json"

VOCAB_SIZE = 32000
TRAIN_RATIO = 0.9
MIN_BPE_FREQUENCY = 2


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        raise FileNotFoundError(
            "Run python -m dataset.collect first."
        )

    print(f"Reading corpus: {RAW}")
    text = RAW.read_text(encoding="utf-8", errors="ignore")

    if len(text) < 10000:
        raise ValueError("Dataset is too small. Add more text.")

    print(f"Characters: {len(text):,}")
    print("Training tokenizer...")

    tokenizer = BPETokenizer()
    tokenizer.train(
        text,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_BPE_FREQUENCY,
        verbose=True,
    )

    print("Encoding corpus...")
    encoded = tokenizer.encode(text)
    ids = torch.tensor(encoded, dtype=torch.int32)

    split = int(len(ids) * TRAIN_RATIO)
    if split <= 0 or split >= len(ids):
        raise ValueError("Invalid train/validation split.")

    # int32 halves dataset RAM/disk compared with int64 while remaining large
    # enough for a 32k-token vocabulary. Training converts batches to int64 on
    # the selected device when needed by nn.Embedding.
    torch.save(
        {
            "train": ids[:split].contiguous(),
            "val": ids[split:].contiguous(),
        },
        DATASET,
        _use_new_zipfile_serialization=True,
    )

    tokenizer.save(VOCAB)

    META.write_text(
        json.dumps(
            {
                "vocab_size": tokenizer.vocab_size,
                "merges": len(tokenizer.merges),
                "tokens": len(ids),
                "train_tokens": split,
                "validation_tokens": len(ids) - split,
                "train_ratio": TRAIN_RATIO,
                "bpe_min_frequency": MIN_BPE_FREQUENCY,
                "corpus_sha256": fingerprint(RAW),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Vocabulary: {tokenizer.vocab_size:,}")
    print(f"BPE merges: {len(tokenizer.merges):,}")
    print(f"Tokens: {len(ids):,}")
    print(f"Train tokens: {split:,}")
    print(f"Validation tokens: {len(ids) - split:,}")
    print(f"Saved dataset: {DATASET}")
    print(f"Saved tokenizer: {VOCAB}")


if __name__ == "__main__":
    main()
