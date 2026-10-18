
"""Run repeatable validation perplexity and report training metadata."""

import json
import math
from pathlib import Path


# Paths
ROOT = Path(__file__).resolve().parents[1]

meta = ROOT / "data/processed/meta.json"
ck = ROOT / "checkpoints/gpt.pt"


# Dataset metadata
print("=== Small-GPT v3 benchmark metadata ===")

if meta.exists():
    print(json.dumps(json.loads(meta.read_text()), indent=2))
else:
    print("No prepared dataset")


# Checkpoint metadata and approximate perplexity
if ck.exists():
    import torch

    x = torch.load(
        ck,
        map_location="cpu",
        weights_only=False,
    )

    print("checkpoint step:", x.get("step"))
    print("best validation loss:", x.get("best_val"))

    print(
        "approx perplexity:",
        math.exp(min(20, float(x.get("best_val", 20)))),
    )
else:
    print("No checkpoint yet.")
