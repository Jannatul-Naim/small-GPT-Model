# 🧠 Small-GPT

An educational decoder-only Large Language Model (LLM) implemented from scratch with PyTorch.


> ⚠️ **This is an educational project.** It is not intended to compete with production-scale models such as ChatGPT, GPT, Llama, or other commercially trained LLMs.

---

# ⚠️ Hardware Requirements

Small-GPT includes an architecture targeting approximately **1.5–2 billion parameters**.

However, full training of a ~1.8B parameter model with AdamW requires significantly more GPU memory and compute than a typical **RTX 3060 12GB** can comfortably provide.

For this reason, the project includes two model configurations:

| Configuration | Purpose                                                                          |
| ------------- | -------------------------------------------------------------------------------- |
| `3060`        | Practical learning and experimentation on an RTX 3060 12GB                       |
| `1.8b`        | Large architecture for experimentation, distributed training, or future hardware |

> ⚠️ The `1.8b` configuration is **not recommended for full AdamW training from scratch on a 12GB RTX 3060**.

The smaller configuration is intended for:

* Learning
* Debugging
* Architecture experiments
* Tokenizer development
* Dataset testing
* Small-scale training
* Fine-tuning experiments

---

# 🏗️ Model Architecture

The `1.8b` configuration uses a modern decoder-only Transformer architecture.

| Component           |   Value |
| ------------------- | ------: |
| Vocabulary Size     |  32,000 |
| Hidden Size         |   2,048 |
| Transformer Layers  |      24 |
| Attention Heads     |      16 |
| KV Heads            |       4 |
| Head Dimension      |     128 |
| FFN Dimension       |   5,504 |
| Context Length      |   2,048 |
| Positional Encoding |    RoPE |
| Normalization       | RMSNorm |
| Activation          |  SwiGLU |
| Attention           |     GQA |
| Weight Tying        |     Yes |

This configuration belongs approximately to the **1.5–2B parameter class**, depending on implementation details and exact parameter accounting.

---

# 📁 Project Structure

```text
Small-GPT/
│
├── data/
│   ├── links.txt              # Website URLs for collection
│   ├── text/                  # User-provided TXT files
│   ├── raw/
│   │   └── corpus.txt         # Persistent collected corpus
│   └── processed/             # Tokenized training data
│
├── dataset/
│   ├── collect.py             # Website and TXT collection
│   └── prepare.py             # Tokenizer training and dataset preparation
│
├── model/
│   └── ...                    # Transformer implementation
│
├── tokenizer/
│   └── ...                    # Byte-level BPE tokenizer
│
├── training/
│   ├── config.py              # Model and training configuration
│   └── train.py               # Training pipeline
│
├── generation/
│   └── generate.py            # Text generation
│
├── web/
│   └── server.py              # Browser streaming interface
│
├── checkpoints/               # Saved checkpoints
│
├── requirements.txt
│
└── README.md
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <https://github.com/Jannatul-Naim/small-GPT-Model>
cd Small-GPT
```

## 2. Create a Virtual Environment (Optional)

```bash
python3 -m venv .venv
```

Activate it:

### Linux / Ubuntu

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

Upgrade `pip`:

```bash
pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

# 🖥️ Check GPU Support

Verify that PyTorch can detect your CUDA GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output:

```text
True
NVIDIA GeForce RTX 3060
```

If CUDA is not available:

```text
False
CPU
```

Make sure your installed PyTorch version supports your CUDA environment.

---

# 📚 Preparing Training Data

Small-GPT supports two primary data sources:

1. 🌐 Websites
2. 📄 Local TXT files

---

## 🌐 Add Website URLs

Add URLs to:

```text
data/links.txt
```

Example:

```text
https://example.com/article1
https://example.com/article2
https://example.com/article3
```

---

## 📄 Add TXT Files

Place your text files inside:

```text
data/text/
```

Example:

```text
data/text/
├── book.txt
├── notes.txt
├── articles.txt
└── dataset.txt
```

---

# 🌐 Collect the Dataset

Run:

```bash
python -m dataset.collect
```

The collector will extract text from supported sources and maintain a persistent corpus.

The combined corpus is stored in:

```text
data/raw/corpus.txt
```

> ⚠️ Only collect data from websites where you have permission. Always respect `robots.txt`, copyright restrictions, rate limits, and the website's Terms of Service.

This collector is intended for **educational use and small-scale experiments**. It is **not a production web crawler**.

---

# 🔤 Tokenize and Prepare the Dataset

After collecting your data, run:

```bash
python -m dataset.prepare
```

This process:

1. Reads the persistent corpus
2. Trains or loads the tokenizer
3. Tokenizes the dataset
4. Stores token IDs in processed form
5. Prepares the data for training

Processed token IDs are stored using:

```text
int32
```

They are converted to:

```text
int64
```

only when moved into the PyTorch model.

This reduces storage and memory usage for large datasets.

---

# 🔤 Optimized Tokenizer

Small-GPT uses a custom **byte-level BPE tokenizer**.

The tokenizer has been optimized to avoid repeatedly rescanning the entire corpus for every BPE merge.

Instead, it uses an incremental approach based on:

* Pair frequency tracking
* Heap-based merge selection
* Linked-list-style token updates
* Local pair frequency updates

Only token pairs affected by a merge need to be updated, making tokenizer training significantly more scalable than a naive full-corpus rescan implementation.

## Tokenizer Features

* Byte-level UTF-8 processing
* BPE tokenization
* Incremental merge updates
* Heap-based merge selection
* Merge-rank encoding
* Compact serialized vocabulary
* UTF-8 round-trip support
* Configurable minimum BPE frequency
* PAD token support
* UNK token support
* BOS token support
* EOS token support

Special token IDs are handled explicitly:

```text
PAD
UNK
BOS
EOS
```

> ⚠️ If you previously generated a tokenizer or processed dataset using an older version of the project, rebuild the processed files before training.

```bash
python -m dataset.prepare
```

---

# ⚙️ Model Configuration

Model settings are located in:

```text
training/config.py
```

The default configuration is:

```python
MODEL_SIZE = "3060"
```

This configuration is intended for practical experimentation on an **RTX 3060 12GB**.

The larger architecture can be selected with:

```python
MODEL_SIZE = "1.8b"
```

> ⚠️ Do not expect the `1.8b` configuration to comfortably fit for full training with AdamW on a 12GB GPU.

---

# 🏋️ Training

Small-GPT supports three training modes:

| Mode     | Description                                      |
| -------- | ------------------------------------------------ |
| `new`    | Start training from randomly initialized weights |
| `resume` | Continue training from the latest checkpoint     |
| `update` | Continue training using an updated dataset       |

---

## 🆕 Start New Training

```bash
python -m training.train --mode new
```

This will:

1. Initialize a new model
2. Create the optimizer
3. Load the prepared dataset
4. Begin training
5. Save checkpoints

---

## 🔄 Resume Training

If training stops or your system restarts:

```bash
python -m training.train --mode resume
```

The training system restores:

* Model weights
* Optimizer state
* Training step
* Model configuration
* Tokenizer fingerprint
* Best validation loss

Training then continues from the saved state.

---

# 🧠 Add New Knowledge

To expand the training corpus:

### 1. Add new URLs

```text
data/links.txt
```

### 2. Add new TXT files

```text
data/text/
```

### 3. Collect the new data

```bash
python -m dataset.collect
```

### 4. Prepare the updated dataset

```bash
python -m dataset.prepare
```

### 5. Continue training

```bash
python -m training.train --mode update
```

---

## ⚠️ Important: Tokenizer Compatibility

During `resume` and `update` training, the tokenizer is deliberately kept fixed.

The checkpoint stores a tokenizer fingerprint.

If the tokenizer vocabulary changes, the training script will refuse to load the checkpoint.

This prevents accidental corruption or incompatibility with the embedding layer.

For example:

```text
Old tokenizer vocabulary
        ↓
Checkpoint embeddings
        ↓
Vocabulary changes
        ↓
Embedding indices no longer match
        ↓
❌ Training is stopped safely
```

This behavior is intentional.

---

# 🧠 Continual Learning and Catastrophic Forgetting

When adding new data, avoid training exclusively on the new dataset.

Training only on new information can cause the model to forget previously learned patterns. This phenomenon is known as:

> **Catastrophic Forgetting**

For continual learning, it is recommended to keep older training data in the corpus.

The training flow should look like:

```text
Old Data
   +
New Data
   ↓
Combined Corpus
   ↓
Tokenizer
   ↓
Training / Update
```

---

# ✍️ Generate Text

After training, generate text using:

```bash
python -m generation.generate \
  --prompt "Artificial intelligence is" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40
```

## Generation Parameters

| Parameter          | Description                         |
| ------------------ | ----------------------------------- |
| `--prompt`         | Starting text for generation        |
| `--max-new-tokens` | Maximum number of generated tokens  |
| `--temperature`    | Controls randomness                 |
| `--top-k`          | Limits sampling to the top K tokens |

Example:

```bash
python -m generation.generate \
  --prompt "The future of artificial intelligence" \
  --max-new-tokens 300 \
  --temperature 0.7 \
  --top-k 50
```

---

# 🌐 Web Interface

Start the local web server:

```bash
python -m web.server
```

Then open:

`http://127.0.0.1:8000`

The interface supports **real-time token streaming**, allowing generated text to appear in the browser as the model produces it.

---

# 🔄 Typical Workflow

A typical Small-GPT workflow looks like this:

```text
           ┌─────────────────┐
           │   Add Data      │
           │ URLs / TXT      │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ dataset.collect │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  corpus.txt     │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ dataset.prepare │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   Tokenizer     │
           │ + Token Dataset │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ Training        │
           │ new / resume /  │
           │ update          │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   Checkpoint    │
           └────────┬────────┘
                    │
           ┌────────┴────────┐
           ▼                 ▼
    ┌─────────────┐   ┌─────────────┐
    │ Text        │   │ Web         │
    │ Generation  │   │ Interface   │
    └─────────────┘   └─────────────┘
```

---

# 💾 Checkpoints

Checkpoints store the information required to safely continue training.

A checkpoint includes:

```text
✓ Model weights
✓ Optimizer state
✓ Training step
✓ Model configuration
✓ Tokenizer fingerprint
✓ Best validation loss
```

This allows training to continue without restarting from scratch.

---

# ❌ What Small-GPT Is Not

Small-GPT does **not** include:

* Pre-trained GPT weights
* Pre-trained Llama weights
* ChatGPT weights
* Instruction tuning
* RLHF
* Production-scale distributed training
* Web-scale data processing
* Enterprise inference optimization

The model starts from **randomly initialized weights**.

If trained on a small dataset, the generated output will likely be:

* Repetitive
* Grammatically inconsistent
* Factually unreliable
* Limited in vocabulary
* Highly dependent on the training corpus

Training a useful language model requires:

```text
Large Dataset
      +
High-Quality Data
      +
Diverse Domains
      +
Large Compute
      +
Long Training Time
      +
Careful Hyperparameter Tuning
```

---

# 🖥️ Hardware Recommendations

For the `3060` configuration:

```text
GPU: NVIDIA RTX 3060 12GB
RAM: 16GB minimum
Recommended RAM: 32GB+
Storage: SSD/NVMe
CUDA: Supported PyTorch CUDA version
```

For larger model training:

```text
Multiple GPUs
or
High-memory GPU
or
Cloud GPU infrastructure
```

Possible techniques for scaling include:

* Mixed precision training
* Gradient accumulation
* Gradient checkpointing
* Distributed Data Parallel
* Fully Sharded Data Parallel
* CPU/NVMe offloading
* Activation checkpointing

These techniques are outside the main educational scope of this project unless explicitly implemented.

---

# 🔒 Security and Responsible Data Collection

When collecting website data:

* Only scrape websites where you have permission
* Respect `robots.txt`
* Follow Terms of Service
* Respect copyright and licensing
* Avoid aggressive request rates
* Do not collect private or restricted content
* Do not use the collector for large-scale crawling

The website collector is intentionally simple and designed for educational experimentation.

# 📊 Example Development Workflow

```bash
## Not mandatory 
# 1. Activate environment
source .venv/bin/activate

# 2. Add URLs and TXT files

# 3. Collect the corpus
python -m dataset.collect

# 4. Train tokenizer and prepare dataset
python -m dataset.prepare

# 5. Start training
python -m training.train --mode new

# 6. Resume later if needed
python -m training.train --mode resume

# 7. Generate text
python -m generation.generate \
  --prompt "Artificial intelligence is" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40

# 8. Start the web interface
python -m web.server


```

---

# ⚠️ Disclaimer

Small-GPT is an educational project intended for learning, experimentation, and research.

It is not production-ready and does not provide guarantees regarding:

* Accuracy
* Safety
* Reliability
* Security
* Factual correctness

Generated text may contain incorrect, biased, repetitive, or nonsensical information.

Always evaluate model output before using it in important applications.

---

# 📜 License

This project is created for **educational purposes**.

You are responsible for ensuring that your use of this project, collected data, and trained models complies with:

* Applicable laws
* Copyright regulations
* Website Terms of Service
* Data licensing requirements
* Privacy regulations
