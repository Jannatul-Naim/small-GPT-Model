"""Research-grade Small-GPT configuration for RTX 3060 12GB."""

MODEL_SIZE = "3060"

CONFIGS = {
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
batch_size = 2
gradient_accumulation_steps = 16

learning_rate = 3e-4
update_learning_rate = 6e-5
weight_decay = 0.1
betas = (0.9, 0.95)
max_steps = 10000
warmup_steps = 500
eval_interval = 250
eval_iters = 20
grad_clip = 1.0
gradient_checkpointing = True
min_lr_ratio = 0.1
seed = 1337

# Continual-learning replay. During --mode update, each optimizer step mixes
# a fraction of old tokens with new-domain tokens.
replay_ratio = 0.35

checkpoint_path = "checkpoints/gpt.pt"
best_checkpoint_path = "checkpoints/gpt-best.pt"
