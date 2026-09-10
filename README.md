# Small-GPT

An educational decoder-only Transformer language model implemented from scratch with PyTorch.

## RTX 3060 12GB profile

The default `3060` profile is deliberately conservative for a single RTX 3060 12GB:

| Setting | Value |
|---|---:|
| Vocabulary | 32,000 |
| Context | 384 |
| Hidden size | 512 |
| Layers | 8 |
| Query heads | 8 |
| KV heads | 2 |
| FFN | 1,536 |
| Parameters | ~40.5M |
| Batch size | 2 |
| Gradient accumulation | 16 |
| Effective tokens / optimizer step | 12,288 |
| Mixed precision | FP16 |
| Gradient checkpointing | Enabled |

This architecture is much more practical for training from scratch on a 12GB RTX 3060 than a billion-parameter model with AdamW.

### Why this is memory efficient

1. **Smaller model**: ~40.5M parameters.
2. **384-token context**: greatly reduces activation memory versus 1024/2048.
3. **Batch 2 + accumulation 16**: keeps per-step GPU memory low while retaining a useful effective batch.
4. **FP16 autocasting + GradScaler**: reduces activation/temporary memory on CUDA.
5. **Gradient checkpointing**: recomputes Transformer blocks during backward instead of storing all block activations.
6. **uint16 processed tokens**: halves host RAM and disk usage versus int32 for a 32k vocabulary.
7. **Atomic checkpoints**: writes to a temporary file and replaces the old checkpoint only after the save succeeds.

## Install

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

Expected on the target PC:

```text
True
NVIDIA GeForce RTX 3060
```

## Prepare data

Put local text files in:

```text
data/text/
```

Add permitted URLs to:

```text
data/links.txt
```

Collect:

```bash
python -m dataset.collect
```

Prepare a fresh tokenizer and dataset:

```bash
python -m dataset.prepare
```

The processed dataset is saved as `uint16`. Batches are converted to `int64` only when moved to the GPU.

## Train from scratch

```bash
python -m training.train --mode new
```

The default run is **5,000 optimizer steps**.

The training loop reports:

```text
step | train loss | validation loss
```

Checkpoints:

```text
checkpoints/gpt.pt       # latest resumable checkpoint
checkpoints/gpt-best.pt  # best validation checkpoint
```

## Resume training

`resume` restores:

- model weights
- optimizer state
- training step
- model configuration
- tokenizer fingerprint
- best validation loss

Run:

```bash
python -m training.train --mode resume
```

Important: `max_steps` means **additional optimizer steps**, not an absolute final step. For example, resuming from step 5,000 runs another 5,000 steps.

## Update with new data

The `update` mode is intentionally different from `resume`.

1. Add new URLs/files.
2. Recollect the combined corpus:

```bash
python -m dataset.collect
```

3. Re-encode the combined corpus **without changing the tokenizer**:

```bash
python -m dataset.prepare --reuse-tokenizer
```

4. Continue from the previous model weights:

```bash
python -m training.train --mode update
```

In `update` mode:

- model weights are restored
- the optimizer is reset
- the step counter starts from 0
- the learning rate is reduced to 50% of the normal training rate

Keeping the old corpus together with the new data helps reduce catastrophic forgetting.

Do **not** run plain `python -m dataset.prepare` for an update, because that would create a new tokenizer and make the old checkpoint incompatible.

## Generate text

```bash
python -m generation.generate \
  --prompt "Artificial intelligence is" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

Generation automatically uses the context window stored in the checkpoint.

## Web UI

```bash
python -m web.server
```

Open:

```text
http://127.0.0.1:8000
```

## Project structure

```text
small-GPT-Model/
├── data/
│   ├── links.txt
│   ├── text/
│   ├── raw/
│   │   └── corpus.txt
│   └── processed/
├── dataset/
│   ├── collect.py
│   └── prepare.py
├── generation/
│   └── generate.py
├── model/
│   └── modern_gpt.py
├── tokenizer/
│   └── bpe.py
├── training/
│   ├── config.py
│   └── train.py
├── web/
├── checkpoints/
├── requirements.txt
└── README.md
```

## Important limitation

This project trains a language model from random initialization. With a small corpus, output can be repetitive, incoherent, or factually wrong. It is an educational/research project, not a ChatGPT replacement.

Only collect data you are permitted to use, and respect website terms, rate limits, copyright, and privacy requirements.

## Short Cut Commands

For a fresh run:
```
cd small-GPT-Model

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m dataset.collect
python -m dataset.prepare

python -m training.train --mode new

```
Then generate:

```
python -m generation.generate \
  --prompt "Artificial intelligence is" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

## Text Generation

Generate large text datasets using Ollama's `gpt-oss:120b-cloud` model.

```bash
python generate_text.py "Machine Learning"
```

Generated files are saved to:

```text
/data/text/
```


## License

This project is created for **educational purposes**.

You are responsible for ensuring that your use of this project, collected data, and trained models complies with:

* Applicable laws
* Copyright regulations
* Website Terms of Service
* Data licensing requirements
* Privacy regulations