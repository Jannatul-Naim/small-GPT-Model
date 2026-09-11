# Small-GPT v3 — Research-Grade Continual Domain Expert

Designed for the supplied **RTX 3060 12GB + 32GB RAM + Ryzen 5 7600**.

## Research objective

Build a local decoder-only language model that can be trained from scratch,
then updated day by day on a focused subject while keeping previous knowledge,
using retrieval for immediately available current information.

## Training flow

```text
PHASE 0  Environment/GPU check
        ↓
PHASE 1  Ingest permitted PDF + TXT + HTML + web sources
        ↓
PHASE 2  Normalize + quality gate + exact/version deduplication
        ↓
PHASE 3  Chunk knowledge for retrieval + source provenance
        ↓
PHASE 4  Train tokenizer ONCE (16,384 vocab)
        ↓
PHASE 5  Build uint16 train/validation/replay binaries
        ↓
PHASE 6  Pretrain Small-GPT from scratch
        ↓
PHASE 7  Evaluate validation loss/perplexity
        ↓
PHASE 8  Add new topic material
        ↓
PHASE 9  Retrieval becomes current immediately
        ↓
PHASE 10 Continual update = new data + replay old data
        ↓
PHASE 11 Re-evaluate against fixed questions
        ↓
PHASE 12 Stream answers through local web UI
```

## v3 model

```text
vocab       16,384
context        768
hidden dim     768
layers          14
query heads     12
KV heads         4
FFN           2048
GQA             yes
RoPE            yes
RMSNorm         yes
SwiGLU          yes
weight tying    yes
```

This is approximately **100M parameters**, rather than the earlier ~44M profile.
The configuration is intentionally below the 12GB limit so there is room for
training overhead.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Build the first corpus

The recommended v3 data engine accepts local PDF/TXT/HTML files recursively. Put permitted material in:

```text
data/ingest/
    papers/
    notes/
    web/
```

You can also keep URLs in `data/links.txt`. Then run the complete pipeline in one command:

```bash
python -m dataset.engine
```

For a later run, keep the tokenizer fixed:

```bash
python -m dataset.engine --reuse-tokenizer
```

The engine extracts text, normalizes it, applies deterministic quality gates,
deduplicates exact content, tracks changed-source versions, creates metadata-rich
retrieval chunks, and writes the same `train.bin` / `val.bin` / `new.bin` /
`replay.bin` files consumed by the existing trainer.

The older three-stage commands (`dataset.collect`, `dataset.chunk`,
`dataset.prepare`) remain available for compatibility.

**Do not use random web dumps.** Your corpus should be relevant, clean, and
legally usable for your experiment.

## 3. First training

For a real experiment, don't repeat the previous 88k-token run. Start with
at least several hundred thousand tokens and preferably millions.

```bash
python -m training.train --mode new --steps 5000
```

Watch validation loss. A training loss collapsing toward zero while validation
loss rises is overfitting; add data or reduce training rather than blindly
running more steps.

## 4. Daily domain learning

Example:

```text
data/topics/robotics/
    day01_kinematics.txt
    day01_vision.txt
    day02_inverse_kinematics.txt
```

Run:

```bash
python -m pipeline.learn --topic "robotics" --steps 1500
```

The pipeline collects everything, detects new sources, reuses the existing
tokenizer, rebuilds retrieval chunks, and runs a continual update.

### Why replay matters

An update samples roughly:

```text
65% new material
35% old/replay material
```

The exact ratio is in `training/config.py`.

This reduces catastrophic forgetting compared with training only on today's
new files.

## 5. Resume interrupted training

```bash
python -m training.train --mode resume --steps 5000
```

This restores model + optimizer + step.

## 6. Real-time knowledge

Start:

```bash
python -m web.server
```

Open:

```text
http://127.0.0.1:8000
```

Retrieval uses the newest indexed chunks. Therefore a new source does not have
to wait for a complete neural training cycle before it can be searched.

## 7. Generate from terminal

```bash
python -m generation.generate --prompt "Explain inverse kinematics" --max-new-tokens 200 --temperature 0.75 --top-k 40 --top-p 0.92 --repetition-penalty 1.08
```

## 8. Benchmark

```bash
python experiments/benchmark.py
```

Keep a fixed evaluation set in `data/eval_questions.txt`. Compare the same
questions after each update.

## 9. Research measurements to record

For each run, record:

- corpus tokens
- new tokens
- replay tokens
- parameter count
- GPU memory
- GPU utilization
- training loss
- validation loss
- validation perplexity
- tokens/second
- update steps
- domain benchmark score
- hallucination/error examples

A research project should show these numbers instead of only saying "the model
trained successfully."

## 10. Recommended research progression

### Experiment A — baseline

General corpus → pretraining → fixed evaluation.

### Experiment B — domain adaptation

General model → 1 focused domain → evaluation.

### Experiment C — continual learning

Day 1 → Day 7 → Day 14 → replay updates → forgetting measurement.

### Experiment D — retrieval vs training

Compare:

```text
model only
model + retrieval
model + continual training
model + continual training + retrieval
```

This can become a strong university research experiment.

## 11. Next upgrades after v3

The v3 data engine now includes PDF/HTML/TXT extraction with provenance. Add the remaining upgrades one at a time:

1. Sentence-transformer embeddings + FAISS.
2. Hybrid BM25 + vector retrieval.
4. Topic router.
5. Instruction-tuning dataset.
6. LoRA adapters for multiple specialist domains.
7. Automatic evaluation with held-out QA.
8. TensorBoard/W&B-style experiment logging.
9. Quantized inference.
10. Knowledge graphs for structured domain facts.

Do not add all ten at once. First establish a reproducible baseline.
