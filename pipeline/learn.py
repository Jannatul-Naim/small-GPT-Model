"""Rebuild topic data with the fixed tokenizer and run a replay update."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Paths
ROOT = Path(__file__).resolve().parents[1]

TOPICS = ROOT / "data/topics"
CK = ROOT / "checkpoints/gpt.pt"
VOCAB = ROOT / "data/processed/vocab.json"


def run(*args):
    """Run a subprocess from the project root."""
    print(">>>", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def slug(s):
    """Convert a topic name into a filesystem-friendly slug."""
    import re

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        s.lower(),
    ).strip("_") or "general"


def main():
    # Parse command-line arguments
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--topic",
        required=True,
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=1500,
    )

    args = ap.parse_args()

    # Verify that the base model and tokenizer exist
    if not CK.exists() or not VOCAB.exists():
        raise SystemExit(
            "Create a base model first: build data with dataset.engine, "
            "then train with --mode new."
        )

    # Prepare the topic directory
    d = TOPICS / slug(args.topic)
    d.mkdir(parents=True, exist_ok=True)

    print(f"Put new permitted .txt material into: {d}")

    # Rebuild the dataset using the existing tokenizer
    run(
        sys.executable,
        "-m",
        "dataset.engine",
        "--input",
        str(TOPICS),
        "--no-web",
        "--reuse-tokenizer",
    )

    # Check whether new source versions were accepted
    new_ids = json.loads(
        (ROOT / "data/raw/new_sources.json").read_text(
            encoding="utf-8"
        )
    )

    if not new_ids:
        raise SystemExit(
            "No new source versions were accepted; "
            "add new material before running an update."
        )

    # Run the replay training update
    run(
        sys.executable,
        "-m",
        "training.train",
        "--mode",
        "update",
        "--steps",
        str(args.steps),
    )

    # Evaluate the updated checkpoint
    report = (
        ROOT
        / "reports"
        / f"{slug(args.topic)}-update-validation.json"
    )

    run(
        sys.executable,
        "evaluation.py",
        "--checkpoint",
        "checkpoints/gpt-best.pt",
        "--output",
        str(report.relative_to(ROOT)),
    )

    print("\nDAILY LEARNING COMPLETE")


if __name__ == "__main__":
    main()
