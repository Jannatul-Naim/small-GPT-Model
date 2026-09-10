"""
Text generation entry point for the Small-GPT language model.

This module loads a trained Small-GPT checkpoint and its corresponding
Byte-Pair Encoding (BPE) tokenizer, then generates text from a user-provided
prompt.

The model automatically uses CUDA when a compatible GPU is available;
otherwise, generation falls back to the CPU.

Example:
    python -m generation.generate \
        --prompt "Artificial intelligence is" \
        --max-new-tokens 200 \
        --temperature 0.8 \
        --top-k 40
"""

from pathlib import Path
import argparse

import torch

from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer


# Resolve the project root relative to this source file.
# This keeps paths independent of the directory from which the command
# is executed.
ROOT = Path(__file__).resolve().parents[1]

# Path to the latest trained Small-GPT checkpoint.
CHECKPOINT = ROOT / "checkpoints/gpt.pt"

# Path to the serialized BPE tokenizer vocabulary.
TOKENIZER = ROOT / "data/processed/vocab.json"


def main() -> None:
    """Load the trained model and generate text from a prompt.

    The generation pipeline consists of the following stages:

    1. Parse command-line generation parameters.
    2. Select CUDA when available, otherwise use the CPU.
    3. Load the trained model checkpoint.
    4. Reconstruct the model using the saved configuration.
    5. Restore the trained model parameters.
    6. Load the tokenizer used by the model.
    7. Encode the input prompt into token IDs.
    8. Generate new tokens using the trained model.
    9. Decode the generated token IDs back into human-readable text.
    """
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Small-GPT model."
    )

    parser.add_argument(
        "--prompt",
        default="Artificial intelligence is",
        help="Initial text prompt used to start generation.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help=(
            "Sampling temperature. Higher values produce more random "
            "output, while lower values produce more deterministic output."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help=(
            "Restrict sampling to the top K most probable tokens at "
            "each generation step."
        ),
    )

    args = parser.parse_args()

    # Prefer GPU acceleration when CUDA is available.
    # Otherwise, use the CPU so the program remains portable.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the checkpoint onto the selected device.
    # map_location prevents device-mismatch errors when, for example,
    # a checkpoint created on a GPU is loaded on a CPU-only system.
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "No checkpoint found. Train the model first with "
            "python -m training.train --mode new"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    # Reconstruct the exact model architecture used during training.
    # The configuration is stored inside the checkpoint so that the
    # architecture does not need to be manually recreated here.
    model = ModernGPT(
        **checkpoint["config"]
    ).to(device)

    # Restore the trained model parameters from the checkpoint.
    model.load_state_dict(
        checkpoint["model_state"]
    )

    # Switch the model to evaluation mode.
    # This disables training-specific behavior such as dropout.
    model.eval()

    # Load the same BPE tokenizer used to prepare the training data.
    # Keeping the tokenizer consistent is essential because token IDs
    # must correspond to the model's vocabulary.
    tokenizer = BPETokenizer.load(
        TOKENIZER
    )

    # Convert the text prompt into integer token IDs.
    ids = tokenizer.encode(args.prompt)

    if not ids:
        raise ValueError("Prompt produced no tokens.")

    # The model only supports its configured context length.
    ids = ids[-checkpoint["config"]["block_size"]:]

    # Create a batch containing one prompt sequence.
    # Shape: [batch_size, sequence_length]
    idx = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    # Generate new tokens autoregressively.
    #
    # max_new_tokens controls the maximum generation length.
    # temperature controls sampling randomness.
    # top_k limits sampling to the most probable K tokens.
    result = model.generate(
        idx,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )

    # Convert the generated token IDs back into readable text.
    # result[0] selects the first (and only) sequence in the batch.
    generated_text = tokenizer.decode(
        result[0].tolist()
    )

    print(generated_text)


if __name__ == "__main__":
    main()

