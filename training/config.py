"""Small-GPT configuration tuned for an NVIDIA RTX 3060 12GB."""

# Practical default. The larger configuration is intentionally retained only
# as a reference for future hardware.
MODEL_SIZE = "small"

CONFIGS = {
    # ~40.5M parameters with the current tied-embedding architecture.
    # This is intentionally conservative for a 12GB RTX 3060.
    "3060": {
        "vocab_size": 32000,
        "block_size": 384,
        "dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 2,
        "ffn_dim": 1536,
        "dropout": 0.0,
    },

    "1.8b": {
        "vocab_size": 32000,
        "block_size": 2048,
        "dim": 2048,
        "n_layers": 24,
        "n_heads": 16,
        "n_kv_heads": 4,
        "ffn_dim": 5504,
        "dropout": 0.0,
    },
}

# RTX 3060-safe defaults:
# 2 samples × 16 accumulation × 384 tokens = 12,288 tokens/optimizer step.
batch_size = 2
gradient_accumulation_steps = 16

learning_rate = 3e-4
weight_decay = 0.1
max_steps = 5000
eval_interval = 250
eval_iters = 10
grad_clip = 1.0

# Saves activation memory by checkpointing Transformer blocks during training.
gradient_checkpointing = True

# More robust checkpointing: gpt.pt is the latest resumable checkpoint;
# gpt-best.pt stores the best validation loss seen so far.
checkpoint_path = "checkpoints/gpt.pt"
best_checkpoint_path = "checkpoints/gpt-best.pt"

seed = 1337
