"""Compare story checkpoints on many generated stories, using the app's settings. Read-only.

Each argument is checkpoint_dir=data_dir. The last run's tokenizer and training text are the shared yardstick.
Usage: python reports/compare_stories.py checkpoints/tinystories-v2=data/tinystories checkpoints/stories-v3=data/stories-v3
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/naim/Projects/LLM/small-GPT-Model")
sys.path.insert(0, str(ROOT))
from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer
from training import config

device = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = [arg.split("=") for arg in sys.argv[1:]]


def open_data(name):
    d = ROOT / name
    train = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
    return train, val, BPETokenizer.load(d / "vocab.json")


# None of these were used to choose or score either model.
PROMPTS = [
    "Ben had a small boat. He took it to the lake and",
    "Anna found a lost kitten under the bed. She",
    "It was raining and Sam could not go outside, so",
    "The old man lived in a little house near the woods. Every morning,",
    "Sara and her brother wanted the same toy. They",
    "A little mouse was very hungry. It looked for",
    "On her birthday, Emma got a big box from her grandma. Inside",
    "The sun was hot, and the frog could not find any water.",
]
SEEDS = range(5)
GRAM = 16
MULT = np.uint64(1_000_003)


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def gram_hashes(tokens):
    tokens = np.asarray(tokens, dtype=np.uint64)
    h = np.zeros(len(tokens) - GRAM + 1, dtype=np.uint64)
    for j in range(GRAM):
        h = h * MULT + tokens[j:len(tokens) - GRAM + 1 + j]
    return h


def copied_share(generated, train):
    """Share of generated GRAM-token runs that also appear in the training text."""
    all_runs = np.concatenate([gram_hashes(g) for g in generated if len(g) >= GRAM])
    unique = np.unique(all_runs)
    found = np.zeros(len(unique), dtype=bool)
    for start in range(0, len(train) - GRAM, 50_000_000):
        chunk = np.asarray(train[start:start + 50_000_000 + GRAM - 1])
        found |= np.isin(unique, gram_hashes(chunk))
    return float(found[np.searchsorted(unique, all_runs)].mean())


def repeated_share(story):
    """Share of 4-word runs in a story that already appeared earlier in it."""
    w = words(story)
    runs = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(runs)) / len(runs) if runs else 0.0


# Known words: all validation text plus the first and last 20M training tokens of every run's data.
# Only counts are printed.
known = set()
for _, data_dir in RUNS:
    train, val, tok = open_data(data_dir)
    for part in (val, train[:20_000_000], train[-20_000_000:]):
        for i in range(0, len(part), 100_000):
            known.update(words(tok.decode(part[i:i + 100_000].tolist())))
print(f"known words: {len(known):,}", flush=True)

all_stories = {}
for name, data_dir in RUNS:
    train, _, tok = open_data(data_dir)
    ck = torch.load(ROOT / name / "gpt-best.pt", map_location="cpu", weights_only=False)
    model = ModernGPT(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["model_state"])
    print(f"\n=== {name}: step {ck['step']} ===", flush=True)
    del ck

    generated, stories = [], []
    stopped = 0
    for p, prompt in enumerate(PROMPTS):
        for s in SEEDS:
            torch.manual_seed(1000 * p + s)
            idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                out = model.generate(idx, 400, 0.75, 40, 0.92, 1.08, tok.EOS, config.min_new_tokens)[0].tolist()
            gen = out[idx.shape[1]:]
            generated.append(gen)
            stopped += tok.EOS in gen
            story = tok.decode(gen)
            stories.append(story)
            print(f"\n[{name} p{p} s{s}] {prompt}{story}", flush=True)

    made_up = [w for story in stories for w in words(story) if w not in known]
    total_words = sum(len(words(story)) for story in stories)
    print(f"\n[{name}] stories {len(stories)} | stopped at EOS {stopped} | mean tokens {np.mean([len(g) for g in generated]):.0f}")
    print(f"[{name}] repeated 4-word runs {np.mean([repeated_share(s) for s in stories]):.1%}")
    print(f"[{name}] unknown words {len(made_up)} of {total_words} ({len(made_up) / total_words:.2%}): {sorted(set(made_up))[:40]}")
    print(f"[{name}] share of generated {GRAM}-token runs found in its own training text: {copied_share(generated, train):.1%}", flush=True)
    all_stories[name] = stories
    del model
    torch.cuda.empty_cache()

train, _, tok = open_data(RUNS[-1][1])
for name, stories in all_stories.items():
    share = copied_share([tok.encode(s) for s in stories], train)
    print(f"[{name}] share of {GRAM}-token runs found, using {RUNS[-1][1]} as the shared yardstick: {share:.1%}", flush=True)
