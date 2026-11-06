"""Combine TinyStories and SimpleStories into one training and one validation text file.

Prints counts only. Every output is swapped in whole, so rerun it after a power cut.
Run with python -I, because it reads downloaded files.
"""
import hashlib
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/home/naim/Projects/LLM/small-GPT-Model")
TINY = ROOT / "data/tinystories/raw"
SIMPLE = ROOT / "data/simplestories/raw"
OUT = ROOT / "data/stories-v3/raw"
SEP = "<|endoftext|>"


def digest(text):
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest()


def stories(files):
    """Yield story text one record batch at a time, reading only that column."""
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000, columns=["story"]):
            yield from batch.column(0).to_pylist()


def swap_in(temp, target):
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(temp, target)


OUT.mkdir(parents=True, exist_ok=True)

# The test split comes first, so stories that also appear in it can leave training.
test_temp = OUT / "._simple-test.txt"
seen = set()
counts = {"test": 0, "test_duplicates": 0, "test_blank": 0, "test_chars": 0, "separator_inside": 0}
with test_temp.open("w", encoding="utf-8", errors="replace") as out:
    for story in stories(sorted(SIMPLE.glob("test-*.parquet"))):
        text = (story or "").strip()
        if not text:
            counts["test_blank"] += 1
            continue
        key = digest(text)
        if key in seen:
            counts["test_duplicates"] += 1
            continue
        seen.add(key)
        counts["separator_inside"] += SEP in text
        counts["test"] += 1
        counts["test_chars"] += len(text)
        out.write(text + "\n" + SEP + "\n")
swap_in(test_temp, OUT / "simple-test.txt")
test_keys = set(seen)

train_temp = OUT / "._train.txt"
counts.update(train=0, train_blank=0, train_duplicates=0, train_in_test=0, train_chars=0)
with train_temp.open("wb") as out:
    with (TINY / "TinyStoriesV2-GPT4-train.txt").open("rb") as f:
        shutil.copyfileobj(f, out, 1 << 24)
    out.write(("\n" + SEP + "\n").encode())
    for story in stories(sorted(SIMPLE.glob("train-*.parquet"))):
        text = (story or "").strip()
        if not text:
            counts["train_blank"] += 1
            continue
        key = digest(text)
        if key in test_keys:
            counts["train_in_test"] += 1
            continue
        if key in seen:
            counts["train_duplicates"] += 1
            continue
        seen.add(key)
        counts["separator_inside"] += SEP in text
        counts["train"] += 1
        counts["train_chars"] += len(text)
        out.write((text + "\n" + SEP + "\n").encode("utf-8", "replace"))
swap_in(train_temp, OUT / "train.txt")

val_temp = OUT / "._val.txt"
with val_temp.open("wb") as out:
    with (TINY / "TinyStoriesV2-GPT4-valid.txt").open("rb") as f:
        shutil.copyfileobj(f, out, 1 << 24)
    out.write(("\n" + SEP + "\n").encode())
    with (OUT / "simple-test.txt").open("rb") as f:
        shutil.copyfileobj(f, out, 1 << 24)
swap_in(val_temp, OUT / "val.txt")

print(counts)
for name in ("train.txt", "val.txt", "simple-test.txt"):
    print(name, f"{(OUT / name).stat().st_size / 1e9:.3f} GB")
