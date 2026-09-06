from model.modern_gpt import ModernGPT


def count(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def main():
    configs = {
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

    for name, cfg in configs.items():
        model = ModernGPT(**cfg)

        params = count(model)

        print(
            f"{name:8} "
            f"{params / 1e9:.3f}B parameters "
            f"({params:,})"
        )

        del model


if __name__ == "__main__":
    main()
