# Small-GPT

An educational decoder-only LLM project implemented in PyTorch.

## Important hardware note

Target architecture: approximately 1.5–2B parameters.

Your RTX 3060 12GB is NOT suitable for comfortable full 1.8B AdamW training from scratch. This project therefore includes:

- `3060` configuration: practical learning/training configuration
- `1.8b` configuration: approximately 1.8B parameter architecture
- tokenizer
- website extraction with BeautifulSoup
- TXT ingestion
- persistent corpus
- new/resume/update training
- checkpoint + optimizer state
- text generation
- real-time browser streaming

The code is intentionally educational rather than production-optimized.

---

## Architecture

The 1.8B configuration is:

```text
Vocabulary       32,000
Hidden size       2,048
Layers               24
Attention heads      16
KV heads              4
Head dimension      128
FFN                  5,504
Context             2,048
RoPE                 yes
RMSNorm              yes
SwiGLU               yes
GQA                  yes
Weight tying         yes
```

This is an approximately 1.5–2B-class decoder architecture.

---

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. Add training data

URLs:

```text
data/links.txt
```

TXT files:

```text
data/text/
```

Then collect:

```bash
python -m dataset.collect
```

Prepare/tokenize:

```bash
python -m dataset.prepare
```

The corpus is kept in:

```text
data/raw/corpus.txt
```

---

## 3. Configuration

Edit:

```text
training/config.py
```

Default:

```python
MODEL_SIZE = "3060"
```

This is the configuration intended for your RTX 3060.

The 1.8B configuration is:

```python
MODEL_SIZE = "1.8b"
```

Do not expect the 1.8B configuration to fit comfortably for full training on a 12GB GPU.

---

## 4. Train a new model

```bash
python -m training.train --mode new
```

This initializes weights randomly.

---

## 5. Resume training

If training stopped:

```bash
python -m training.train --mode resume
```

The checkpoint contains:

- model weights
- optimizer state
- step
- configuration
- tokenizer fingerprint
- best validation loss

Training continues from the saved step.

---

## 6. Add new knowledge

Add more URLs:

```text
data/links.txt
```

Add TXT files:

```text
data/text/
```

Then:

```bash
python -m dataset.collect
python -m dataset.prepare
```

Finally:

```bash
python -m training.train --mode update
```

### Important

`update` uses the existing checkpoint and the newly prepared combined corpus.

The tokenizer is deliberately kept fixed during resume/update. If the vocabulary changes, the training script refuses to load the old checkpoint rather than silently corrupting the embedding layer.

For continual learning, keeping old data in the corpus is important because training only on new data can cause catastrophic forgetting.

---

## 7. Generate text

```bash
python -m generation.generate \
  --prompt "Artificial intelligence is" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

---

## 8. Start web interface

After training:

```bash
python -m web.server
```

Open:

```text
http://127.0.0.1:8000
```

Generated tokens are streamed to the browser.

---


## 9. What this project is NOT

This project does not contain pretrained GPT/Llama weights.

It is not ChatGPT.

It is not an instruction-tuned assistant.

If trained on a small corpus, output quality will be limited.

A useful LLM requires a large, clean, diverse dataset and substantial computewith huge feeding .

---

## 11. Security / scraping

Only scrape websites where you have permission and follow the site's robots.txt and terms of service.

The collector is a basic educational extractor, not a web-scale crawler.


## Optimized tokenizer

The byte-level BPE tokenizer was upgraded from a full-corpus rescan on every
merge to an incremental heap + linked-list implementation. Pair frequencies
are updated only around affected tokens, making tokenizer training much more
scalable for large corpora.

The tokenizer also now:

- Handles PAD/UNK/BOS/EOS IDs correctly.
- Uses a compact serialized vocabulary.
- Encodes using a merge-rank heap instead of repeatedly scanning every merge.
- Keeps UTF-8 round-trip behavior for arbitrary text.
- Supports a configurable minimum BPE frequency.

Dataset preparation stores token IDs as `int32` and converts batches to
`int64` only when they are moved to the model device, reducing processed
dataset memory and disk usage.

Run:

```bash
python -m dataset.prepare
```

If a previous tokenizer/dataset was created with an older version, rebuild the
processed files before training.
