"""Score story checkpoints in bits per character on held-out text, each with its own tokenizer. Read-only."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/naim/Projects/LLM/small-GPT-Model")
sys.path.insert(0, str(ROOT))
from dataset.large_text import read_documents
from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [
    ("v2", "checkpoints/tinystories-v2/gpt-best.pt", "data/tinystories"),
    ("v3", "checkpoints/stories-v3/gpt-best.pt", "data/stories-v3"),
]
TEXTS = [
    ("TinyStories valid", "data/tinystories/raw/TinyStoriesV2-GPT4-valid.txt"),
    ("SimpleStories test", "data/stories-v3/raw/simple-test.txt"),
]


@torch.inference_mode()
def full_sweep(model, data, block, bs=8):
    total, n = 0.0, 0
    starts = list(range(0, len(data) - block - 1, block))
    for b in range(0, len(starts), bs):
        st = starts[b:b + bs]
        x = torch.stack([torch.from_numpy(data[i:i + block]) for i in st]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + block + 1]) for i in st]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            _, l = model(x, y)
        total += float(l) * x.numel()
        n += x.numel()
    return total / n


# Only counts and scores are printed, never the text.
for label, path, data_dir in MODELS:
    ck = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    model = ModernGPT(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["model_state"])
    block = ck["config"]["block_size"]
    tok = BPETokenizer.load(ROOT / data_dir / "vocab.json")
    print(f"\n=== {label}: step {ck['step']}, block {block}, vocab {tok.vocab_size} ===", flush=True)
    del ck
    for text_name, text_path in TEXTS:
        tokens, chars, decoded_chars, docs = [], 0, 0, 0
        for doc in read_documents(ROOT / text_path):
            ids = tok.encode(doc)
            chars += len(doc)
            decoded_chars += len(tok.decode(ids))
            tokens.extend(ids)
            tokens.append(tok.EOS)
            docs += 1
        data = np.asarray(tokens, dtype=np.int64)
        nats = full_sweep(model, data, block)
        per_token = chars / len(data)
        print(
            f"{label} on {text_name}: docs {docs:,} | tokens {len(data):,} | chars {chars:,} "
            f"(decoded {decoded_chars:,}) | chars/token {per_token:.3f} | nats/token {nats:.4f} | "
            f"perplexity {math.exp(nats):.2f} | bits/char {nats / per_token / math.log(2):.4f}",
            flush=True,
        )
    del model
    torch.cuda.empty_cache()
