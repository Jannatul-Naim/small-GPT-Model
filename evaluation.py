"""Repeatable held-out next-token evaluation for a saved Small-GPT checkpoint.

Example: python evaluation.py --checkpoint checkpoints/gpt-best.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from model.modern_gpt import ModernGPT

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/gpt-best.pt")
    parser.add_argument("--batches", type=int, default=100, help="Fixed number of held-out windows to score")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error("--batches and --batch-size must be positive")

    vocab_path = DATA / "vocab.json"
    val_path = DATA / "val.bin"
    meta_path = DATA / "meta.json"
    split_path = DATA / "split.json"
    for path in (args.checkpoint, vocab_path, val_path, meta_path, split_path):
        if not path.exists():
            raise FileNotFoundError(path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer_hash = sha256(vocab_path)
    if checkpoint.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Checkpoint tokenizer hash does not match data/processed/vocab.json")
    tokens = np.memmap(val_path, dtype=np.uint16, mode="r")
    cfg = checkpoint["config"]
    block = int(cfg["block_size"])
    if len(tokens) <= block + 1:
        raise ValueError("Validation set is too short for one context window")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    model = ModernGPT(**cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for _ in range(args.batches):
            starts = torch.randint(0, len(tokens) - block - 1, (args.batch_size,), generator=generator).tolist()
            x = torch.stack([torch.from_numpy(np.asarray(tokens[i:i + block], dtype=np.int64)) for i in starts]).to(device)
            y = torch.stack([torch.from_numpy(np.asarray(tokens[i + 1:i + block + 1], dtype=np.int64)) for i in starts]).to(device)
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            losses.append(float(loss))

    mean_loss = sum(losses) / len(losses)
    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "device": device,
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "context_length": block,
        "validation_loss_nats_per_token": mean_loss,
        "validation_perplexity": math.exp(mean_loss),
        "validation_binary_sha256": sha256(val_path),
        "tokenizer_sha256": tokenizer_hash,
        "dataset_metadata": json.loads(meta_path.read_text(encoding="utf-8")),
        "split": json.loads(split_path.read_text(encoding="utf-8")),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
