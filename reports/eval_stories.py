"""Score a story model: losses, bigram baseline, copying, and fixed prompts. Read-only.

Usage: python reports/eval_stories.py checkpoints/stories-v3 data/stories-v3 [best] [last]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/naim/Projects/LLM/small-GPT-Model")
sys.path.insert(0, str(ROOT))
from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer
from training import config

CK = ROOT / sys.argv[1]
D = ROOT / sys.argv[2]
device = "cuda" if torch.cuda.is_available() else "cpu"
train = np.memmap(D / "train.bin", dtype=np.uint16, mode="r")
val = np.memmap(D / "val.bin", dtype=np.uint16, mode="r")
tok = BPETokenizer.load(D / "vocab.json")
V = tok.vocab_size

val_chars = sum(len(tok.decode(val[i:i + 100_000].tolist())) for i in range(0, len(val), 100_000))
chars_per_token = val_chars / len(val)
print(f"train tokens {len(train):,} | val tokens {len(val):,} | vocab {V} | val chars/token {chars_per_token:.2f}")
print(f"EOS in train {int((train == tok.EOS).sum()):,} | EOS in val {int((val == tok.EOS).sum()):,}")
freq = np.bincount(train, minlength=V)
print(f"vocab never in train {int((freq == 0).sum())} | median token frequency {int(np.median(freq)):,}")
print(f"tokens spanning 2+ words: {sum(len(tok.decode([i]).split()) >= 2 for i in range(260, V))}")

# Interpolated bigram baseline, counted in chunks so memory stays flat.
counts = np.zeros(V * V, dtype=np.int64)
for start in range(0, len(train) - 1, 50_000_000):
    end = min(start + 50_000_001, len(train))
    a = np.asarray(train[start:end - 1], dtype=np.int64)
    b = np.asarray(train[start + 1:end], dtype=np.int64)
    counts += np.bincount(a * V + b, minlength=V * V)
context = counts.reshape(V, V).sum(axis=1)
unigram = (freq + 1) / (freq.sum() + V)
a = np.asarray(val[:-1], dtype=np.int64)
b = np.asarray(val[1:], dtype=np.int64)
pair_counts = counts[a * V + b]
bigram = np.divide(pair_counts, context[a], out=np.zeros(len(a)), where=context[a] > 0)
baseline = float(-np.log(0.5 * bigram + 0.5 * unigram[b]).mean())
print(f"bigram baseline val loss: {baseline:.4f} | perplexity {math.exp(baseline):.1f}")
del counts, context, pair_counts, bigram, a, b


@torch.inference_mode()
def loss_on(model, data, block, windows=2000, seed=2026, bs=4):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(windows // bs):
        st = torch.randint(0, len(data) - block - 1, (bs,), generator=g).tolist()
        x = torch.stack([torch.from_numpy(np.asarray(data[i:i + block], dtype=np.int64)) for i in st]).to(device)
        y = torch.stack([torch.from_numpy(np.asarray(data[i + 1:i + block + 1], dtype=np.int64)) for i in st]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            _, l = model(x, y)
        out.append(float(l))
    return float(np.mean(out))


@torch.inference_mode()
def full_sweep(model, data, block, bs=8):
    total, n = 0.0, 0
    starts = list(range(0, len(data) - block - 1, block))
    for b in range(0, len(starts), bs):
        st = starts[b:b + bs]
        x = torch.stack([torch.from_numpy(np.asarray(data[i:i + block], dtype=np.int64)) for i in st]).to(device)
        y = torch.stack([torch.from_numpy(np.asarray(data[i + 1:i + block + 1], dtype=np.int64)) for i in st]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            _, l = model(x, y)
        total += float(l) * x.numel()
        n += x.numel()
    return total / n


GRAM = 16
MULT = np.uint64(1_000_003)


def gram_hashes(tokens):
    """Hash every GRAM-token run. The window shrinks with wrap-around uint64 math."""
    tokens = np.asarray(tokens, dtype=np.uint64)
    h = np.zeros(len(tokens) - GRAM + 1, dtype=np.uint64)
    for j in range(GRAM):
        h = h * MULT + tokens[j:len(tokens) - GRAM + 1 + j]
    return h


def copied_share(generated):
    """Share of generated GRAM-token runs that also appear in the training text."""
    runs = [gram_hashes(g) for g in generated if len(g) >= GRAM]
    if not runs:
        return 0.0
    all_runs = np.concatenate(runs)
    unique = np.unique(all_runs)
    found = np.zeros(len(unique), dtype=bool)
    for start in range(0, len(train) - GRAM, 50_000_000):
        chunk = np.asarray(train[start:start + 50_000_000 + GRAM - 1])
        found |= np.isin(unique, gram_hashes(chunk))
    return float(found[np.searchsorted(unique, all_runs)].mean())


PROMPTS = [
    "Once upon a time, there was a little girl named Lily. She loved to",
    "Tom and his dog went to the park. They saw a big",
    "The little bird was sad because",
    "One day, a boy found a red ball in the garden.",
    "Mia wanted to bake a cake for her mom, but",
    "What is the capital of France?",
]

CHECKPOINTS = {"best": CK / "gpt-best.pt", "last": CK / "gpt.pt"}
for label in sys.argv[3:] or CHECKPOINTS:
    path = CHECKPOINTS[label]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ModernGPT(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["model_state"])
    block = ck["config"]["block_size"]
    params = sum(q.numel() for q in model.parameters())
    tl = loss_on(model, train, block)
    vw = loss_on(model, val, block)
    vf = full_sweep(model, val, block)
    print(f"\n=== {label}: step {ck['step']}, {params:,} params, {len(train) / params:.1f} train tokens per param ===")
    print(f"train {tl:.4f} | val (2000 windows) {vw:.4f} | val (full pass) {vf:.4f} | perplexity {math.exp(vf):.1f} | gap {vw - tl:.4f} | bits/char {vf / chars_per_token / math.log(2):.3f}")
    del ck

    generated = []
    for prompt in PROMPTS:
        torch.manual_seed(7)
        idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model.generate(idx, 220, 0.75, 40, 0.92, 1.08, tok.EOS, config.min_new_tokens)[0].tolist()
        gen = out[idx.shape[1]:]
        generated.append(gen)
        print(f"\n[{label}] PROMPT: {prompt}\n{tok.decode(out)}\n-- generated {len(gen)} tokens | stopped at EOS: {tok.EOS in gen}")

    print(f"\n[{label}] share of generated {GRAM}-token runs found in the training text: {copied_share(generated):.1%}")
    del model
    torch.cuda.empty_cache()
