from __future__ import annotations

import argparse
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


def load_or_train_tokenizer(reuse: bool):
    if reuse:
        if not VOCAB.exists():
            raise FileNotFoundError(
                "--reuse-tokenizer was requested, but "
                f"{VOCAB} does not exist. Prepare a fresh dataset first."
            )
        print(f"Loading existing tokenizer: {VOCAB}")
        return BPETokenizer.load(VOCAB)

    print("Training tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.train(
        RAW.read_text(encoding="utf-8", errors="ignore"),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_BPE_FREQUENCY,
        verbose=True,
    )
    tokenizer.save(VOCAB)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize and prepare the Small-GPT corpus."
    )
    parser.add_argument(
        "--reuse-tokenizer",
        action="store_true",
        help=(
            "Reuse data/processed/vocab.json instead of retraining BPE. "
            "Use this for continual-learning/update runs."
        ),
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        raise FileNotFoundError(
            "Run python -m dataset.collect first."
        )

    print(f"Reading corpus: {RAW}")
    text = RAW.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if len(text) < 10000:
        raise ValueError("Dataset is too small. Add more text.")

    print(f"Characters: {len(text):,}")

    tokenizer = load_or_train_tokenizer(
        reuse=args.reuse_tokenizer
    )

    print("Encoding corpus...")
    encoded = tokenizer.encode(text)

    if tokenizer.vocab_size > 65535:
        raise ValueError(
            "Vocabulary is larger than uint16 can represent. "
            "Reduce VOCAB_SIZE before preparing the dataset."
        )

    # uint16 cuts processed token RAM/disk usage in half versus int32.
    # Training converts only sampled batches to int64 on the target device.
    ids = torch.tensor(encoded, dtype=torch.uint16)

    split = int(len(ids) * TRAIN_RATIO)
    if split <= 0 or split >= len(ids):
        raise ValueError("Invalid train/validation split.")

    torch.save(
        {
            "train": ids[:split].contiguous(),
            "val": ids[split:].contiguous(),
        },
        DATASET,
        _use_new_zipfile_serialization=True,
    )

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
                "token_dtype": "uint16",
                "corpus_sha256": fingerprint(RAW),
                "tokenizer_sha256": fingerprint(VOCAB),
                "reused_tokenizer": args.reuse_tokenizer,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"Vocabulary: {tokenizer.vocab_size:,}")
    print(f"BPE merges: {len(tokenizer.merges):,}")
    print(f"Tokens: {len(ids):,}")
    print(f"Train tokens: {split:,}")
    print(f"Validation tokens: {len(ids) - split:,}")
    print("Token dtype: uint16")
    print(f"Saved dataset: {DATASET}")
    print(f"Saved tokenizer: {VOCAB}")


if __name__ == "__main__":
    main()
