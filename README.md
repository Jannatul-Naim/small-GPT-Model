# Small-GPT: an educational research project

Small-GPT is a decoder-only language model and data pipeline for learning how
to build, train, evaluate, and update a small model locally. The repository is
designed to support controlled experiments on a single consumer GPU (the
default profile targets a 12 GB RTX 3060), while keeping the code compact enough
to study.

The architecture and pipeline are research tools; the project is not a
validated domain expert until a specific dataset, training run, and evaluation
show that it performs well. Use the experiment procedure below to make that
claim responsibly.

## What is included

- A roughly 100M-parameter decoder-only Transformer with RMSNorm, RoPE,
  grouped-query attention, SwiGLU, and tied input/output embeddings.
- A source-aware pipeline for local PDF, TXT, and HTML files, with optional
  URLs, normalization, filtering, exact deduplication, chunking, and a BPE
  tokenizer.
- Pretraining, resume, and replay-based continual-update modes.
- Local lexical retrieval and a small web interface.
- Optional Ollama-based synthetic educational text generation, saved separately
  from the primary source corpus.
- Fixed-window held-out loss/perplexity reporting with data and tokenizer
  hashes (`evaluation.py`).

## Quick start

### 1. Prepare an environment

Use Python 3.10 or newer. Create an isolated environment and install the
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Confirm PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The model can run on CPU for code study, but training the default model on CPU
is not a practical full experiment.

### 2. Choose and document a small domain

Pick a subject narrow enough to evaluate, such as introductory control
systems, a particular biology unit, or a software library version. Collect
material you have permission to use: your own notes, open textbooks, public
technical documentation, or datasets whose terms explicitly allow your
intended use. Record each source's title, author/organization, URL or file,
access date, license/terms, and any modifications. Do not use an arbitrary web
dump or assume that publicly reachable text is reusable.

For the first meaningful run, aim for at least several hundred thousand
training tokens and preferably millions. Include multiple independent sources
and enough held-out sources for evaluation. The engine needs at least two
distinct sources with 769 or more tokens after tokenization to create both
training and validation sets. A tiny corpus is useful for a smoke run, not for
claims about general performance.

### Optional: generate a synthetic lesson

Install and start Ollama separately, then run:

```bash
python generate_text.py "Neural Networks"
```

The default model is `gpt-oss:120b-cloud`; change it with `--model` if needed.
Control the requested output limit with `--num-predict` (default: 20,000):

```bash
python generate_text.py --num-predict 3000 "Neural Networks"
```

Text is saved under `data/text/`, with a JSON sidecar labeling it synthetic
and recording the model and generation time. Repeated runs get a new filename
instead of replacing the earlier result. These files are **not** ingested by
the training pipeline by default. Review factual claims against sources and
keep the sidecar; only copy reviewed files into `data/ingest/synthetic/` when
you intentionally want to study synthetic-data augmentation. Preserve the
original source corpus and evaluate on held-out material that was not used in
the generation prompt.

Place files in the following layout:

```text
data/ingest/
  books/
  course-notes/
  documentation/
```

Optionally add permitted URLs, one per line, to `data/links.txt`. A URL fetch
may fail because of site rules, access controls, or format; prefer downloading
permitted material yourself and retaining its provenance.

### 3. Build the dataset and tokenizer

For the first run, train the tokenizer and build the files consumed by the
trainer:

```bash
python -m dataset.engine --input data/ingest --no-web
```

To fetch the permitted URLs from `data/links.txt`, omit `--no-web`:

```bash
python -m dataset.engine --input data/ingest
```

Inspect these outputs before training:

```text
data/raw/sources.jsonl       extracted source text and provenance
data/raw/chunks.jsonl        retrieval chunks with source metadata
data/raw/stats.json          accepted, skipped, duplicate, and source counts
data/processed/meta.json     token counts and content hashes
data/processed/split.json    source IDs assigned to train and validation
data/processed/vocab.json    fixed tokenizer vocabulary
```

Check that the sources were extracted cleanly and that the validation sources
are genuinely held out. A source-level split prevents a source appearing on
both sides of the split; it does not prevent leakage if another source repeats
the same content or if evaluation questions/answers are copied into training
material. Clean and deduplicate those cases yourself.

### 4. Run a short smoke training

This catches setup and data problems without committing to a long run:

```bash
python -m training.train --mode new --steps 10
```

If it completes, start the planned training run. Save the terminal output and
the exact command in your experiment notes:

```bash
python -m training.train --mode new --steps 5000
```

The trainer prints validation loss at intervals and writes checkpoints under
`checkpoints/`. Watch for training instability or validation loss rising while
training loss falls. The step count is an example budget, not a guarantee of
convergence; choose it based on validation results and available compute.

To continue an interrupted run, use the matching processed data and tokenizer:

```bash
python -m training.train --mode resume --steps 1000
```

Resume and update now stop if the loaded checkpoint tokenizer hash differs
from the current vocabulary. Keep a copy of `vocab.json` with each checkpoint.

### 5. Evaluate the checkpoint

Run a fixed set of held-out validation windows and save the machine-readable
report:

```bash
python evaluation.py \
  --checkpoint checkpoints/gpt-best.pt \
  --batches 100 \
  --seed 2026 \
  --output reports/baseline-validation.json
```

The report includes validation loss in nats/token, perplexity, checkpoint
step, tokenizer and validation-data hashes, and the recorded source split. The
same seed and batch settings select the same windows. Compare checkpoints only
when they use the same tokenizer, validation data, and evaluation settings.

Perplexity measures next-token prediction on this validation text. It is not a
score for factual accuracy, reasoning, or usefulness. Also keep a separate
fixed set of domain questions with reference answers. Have a person score each
answer with a written rubric (for example: correctness, evidence, completeness,
and unsupported claims), and retain both good and failed examples. Do not
silently change the question set or scoring rubric between runs.

Generate a sample from the trained model with:

```bash
python -m generation.generate \
  --prompt "Explain the key idea in this topic:" \
  --max-new-tokens 160
```

Generation is qualitative inspection; it does not replace the held-out
evaluation.

### 6. Run a controlled continual-learning experiment

Formulate the question first. A useful starter question is:

> Does replay data reduce loss of old-topic performance when the model is
> updated on new-topic material?

Create fixed old-topic and new-topic evaluation sets that are kept out of all
training sources. Save the pretrained checkpoint and make a copy for each
condition. Compare at least:

| Condition | Update data | Purpose |
|---|---|---|
| No update | None | Baseline drift/control |
| New only | New-topic sources | Measures adaptation and forgetting without replay |
| New + replay | New-topic and old-topic sources | Tests the replay hypothesis |

Put new permitted source files under `data/topics/`, retaining topic folders
and a source log. Rebuild while keeping the original tokenizer fixed:

```bash
python -m dataset.engine --input data/topics --no-web --reuse-tokenizer
python -m training.train --mode update --steps 1500
```

The current update mode samples replay batches with the configured probability
in `training/config.py`; this is a batch-level sampling probability, not a
guaranteed exact token ratio for a short run. Run the new-only condition by
setting `replay_ratio = 0.0` for that run, then restore the setting and run
new-plus-replay from the same original checkpoint. Keep each condition's
checkpoint and report separate. Do not compare sequentially updated models as
if each were an independent condition.

Evaluate both old- and new-topic question sets for every condition. Report the
change from the no-update baseline: new-topic gain and old-topic retention.
Repeat with at least three random training seeds for stronger evidence, and
report every run or the mean and spread; do not report only the best run.

### 7. Optional retrieval comparison

The local retrieval index uses lexical matching over `data/raw/chunks.jsonl`.
Start the web interface with:

```bash
python -m web.server
```

Then visit `http://127.0.0.1:8000`. To study retrieval, compare model-only
answers with retrieval-assisted answers on the same held-out questions. Record
the retrieved source IDs and whether the answer is supported by them. Retrieval
can expose current source text to an answer flow, but it does not change model
weights or establish that the model learned that information.

## What to save for every experiment

Create a folder per experiment and preserve:

- Research question, hypotheses, conditions, and scoring rubric.
- Git commit or source snapshot, exact commands, config, and random seeds.
- Hardware, operating system, Python, PyTorch, and CUDA versions.
- Source manifest, licenses/terms, extraction decisions, and dataset hashes.
- Tokenizer hash, source split, train/validation token counts, and checkpoint.
- Training and validation curves, evaluation JSON, question-level scores, and
  representative failures.
- Any deviations from the planned experiment.

Do not publish source text unless the license permits redistribution. Publishing
source IDs, hashes, collection methods, and transformation scripts can still
make a study easier to reproduce.

## Educational path through the code

1. Follow a source through `dataset/engine.py`: extraction, normalization,
   splitting, tokenization, and binary output.
2. Trace `training/train.py`: random context windows, next-token targets,
   cross-entropy, gradient accumulation, optimizer step, and checkpoint.
3. Study `model/modern_gpt.py`: one token batch through embeddings, attention,
   residual blocks, logits, and loss.
4. Change one experimental variable at a time (for example, replay ratio or
   model size) and explain the prediction before running it.
5. Compare quantitative metrics with generated examples and discuss where the
   metrics fail to capture answer quality.

## Current limits and honest claims

- The default model is randomly initialized and trained from scratch; it needs
  much more high-quality text than a tiny demo corpus.
- The tokenizer is a compact project implementation, not a claim of parity
  with production tokenizers.
- Validation loss/perplexity only reflects the selected held-out text.
- The question-based answer score is human-rated unless you build and validate
  a separate automatic evaluator.
- Replay may help retention, but its effect must be measured against matched
  controls. It is not guaranteed by the implementation.
- Results on one small corpus and one GPU do not establish broad language
  ability or general research conclusions.

## Suggested report outline

1. Research question and hypothesis
2. Data sources, licenses, filtering, and split
3. Model, tokenizer, optimizer, and compute budget
4. Experimental conditions and seeds
5. Validation loss/perplexity and domain-question results
6. Old-topic retention, new-topic gain, and failure examples
7. Limitations, deviations, and what the evidence supports
