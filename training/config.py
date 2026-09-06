# Use "3060" for actual experimentation on an RTX 3060.
# Use "1.8b" only when you have enough memory/compute.

MODEL_SIZE = "3060"

CONFIGS = {
    "3060": {
        "vocab_size": 32000,
        "block_size": 512,
        "dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 4,
        "ffn_dim": 2048,
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

batch_size = 4
gradient_accumulation_steps = 8

learning_rate = 3e-4
weight_decay = 0.1

max_steps = 5000
eval_interval = 250
eval_iters = 10

grad_clip = 1.0

seed = 1337

checkpoint_path = (
    "checkpoints/gpt.pt"
)
