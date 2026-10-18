# Data Engine

The data engine is the reproducible ingestion layer for Small-GPT. It accepts
local `PDF`, `TXT`, `HTML`/`HTM` files recursively and can also keep the
existing `data/links.txt` web-source workflow.

## Directory layout

```text
data/ingest/
    papers/
        paper1.pdf
    notes/
        notes.txt
    web/
        article.html
```

## One-command build

From the project root:

```bash
python -m dataset.engine
```

The engine performs:

1. PDF/HTML/TXT extraction
2. Unicode/newline normalization
3. deterministic quality filtering
4. exact content deduplication
5. source-version tracking when a file changes
6. retrieval chunk generation with source metadata
7. BPE tokenizer training or reuse
8. source-level train/validation splitting
9. `uint16` token binary generation
10. provenance/statistics files

## Reuse the tokenizer

After the first build:

```bash
python -m dataset.engine --reuse-tokenizer
```

This keeps the existing tokenizer vocabulary fixed while incorporating new
material, which is required for stable continual-learning updates.

## Useful outputs

```text
data/raw/sources.jsonl   # canonical source manifest + extracted text
data/raw/chunks.jsonl    # retrieval chunks + provenance
/data/raw/corpus.txt     # flattened corpus for inspection
/data/raw/new_sources.json
/data/raw/state.json
/data/raw/stats.json

data/processed/vocab.json
/data/processed/train.bin
/data/processed/val.bin
/data/processed/new.bin
/data/processed/replay.bin
/data/processed/meta.json
```

The leading `/` in the output examples above is only visual; paths are relative
to the project root.

## Tune ingestion

```bash
python -m dataset.engine \
  --input data/ingest \
  --min-chars 800 \
  --chunk-chars 1800 \
  --overlap 250
```

For a fully local build that ignores `links.txt`:

```bash
python -m dataset.engine --no-web
```
