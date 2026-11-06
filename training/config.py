"""Research-grade Small-GPT configuration for RTX 3060 12GB."""

MODEL_SIZE = "small"

CONFIGS = {
    # About 16M parameters, or 11 per token of the 1.4M-token training set.
    "small": {
        "vocab_size": 16384,
        "block_size": 768,
        "dim": 384,
        "n_layers": 6,
        "n_heads": 6,
        "n_kv_heads": 2,
        "ffn_dim": 1024,
        "dropout": 0.1,
    },
    # About 24M parameters with a 4096-token vocabulary. Made for TinyStories
    # (556M tokens); it would memorize a small corpus. Select it with
    # --model-size medium.
    "medium": {
        "vocab_size": 16384,
        "block_size": 768,
        "dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 2,
        "ffn_dim": 1365,
        "dropout": 0.1,
    },
    # About 185M parameters with an 8192-token vocabulary, using about 9.6 GiB
    # of GPU memory in training. Made for a billion tokens of stories. Select
    # it with --model-size large.
    "large": {
        "vocab_size": 16384,
        "block_size": 768,
        "dim": 1024,
        "n_layers": 16,
        "n_heads": 16,
        "n_kv_heads": 4,
        "ffn_dim": 2730,
        "dropout": 0.1,
    },
    # About 100M parameters. It memorized the 1.4M-token training set, so use
    # it only with far more data. It also needs batch_size = 2,
    # gradient_accumulation_steps = 16, and gradient_checkpointing = True.
    "3060": {
        "vocab_size": 16384,
        "block_size": 768,
        "dim": 768,
        "n_layers": 14,
        "n_heads": 12,
        "n_kv_heads": 4,
        "ffn_dim": 2048,
        "dropout": 0.05,
    },
}

# Optimizer-step settings. Effective batch = batch_size * accumulation.
batch_size = 8
gradient_accumulation_steps = 4

learning_rate = 3e-4
update_learning_rate = 6e-5
weight_decay = 0.1
betas = (0.9, 0.95)
max_steps = 1500
warmup_steps = 150
eval_interval = 100
eval_iters = 20
grad_clip = 1.0
gradient_checkpointing = False
min_lr_ratio = 0.1
seed = 1337

# Continual-learning replay. During --mode update, each optimizer step mixes
# a fraction of old tokens with new-domain tokens.
replay_ratio = 0.35

# Tokens to generate before EOS may end a reply.
min_new_tokens = 20

checkpoint_path = "checkpoints/gpt.pt"
best_checkpoint_path = "checkpoints/gpt-best.pt"
