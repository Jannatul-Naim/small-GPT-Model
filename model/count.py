"""Model parameter counting utility for Small-GPT.

This module instantiates each supported Small-GPT architecture and reports
the total number of parameters.

It is useful for verifying the actual parameter count of each configuration
before starting training.
"""

from model.modern_gpt import ModernGPT


def count_parameters(model: ModernGPT) -> int:
    """Return the total number of parameters in the model.

    Args:
        model: Small-GPT model whose parameters should be counted.

    Returns:
        Total number of parameters registered by the model.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def main() -> None:
    """Create each supported model configuration and report its size.

    The project provides two primary configurations:

    - ``3060``: Smaller configuration intended for practical experimentation
      on an RTX 3060 12GB.
    - ``1.8b``: Larger architecture intended for experimentation with
      higher-memory or distributed training systems.
    """
    configs = {
        # Practical configuration for learning, debugging, and
        # experimentation on an RTX 3060 12GB.
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

        # Larger architecture in approximately the 1.5–2B parameter class.
        # Full AdamW training from scratch is not recommended on a 12GB GPU.
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

    # Evaluate each configuration independently so that only one model
    # needs to remain in memory at a time.
    for name, config in configs.items():
        # Instantiate the model using the exact architecture configuration.
        model = ModernGPT(**config)

        # Calculate the model's total parameter count.
        parameters = count_parameters(model)

        # Display both the approximate parameter count in billions and
        # the exact number of parameters.
        print(
            f"{name:8} "
            f"{parameters / 1e9:.3f}B parameters "
            f"({parameters:,})"
        )

        # Release the current model before processing the next configuration.
        del model


if __name__ == "__main__":
    main()

