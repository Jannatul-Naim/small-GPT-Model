from pathlib import Path
import argparse

import torch

from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT / "checkpoints/gpt.pt"
)

TOKENIZER = (
    ROOT / "data/processed/vocab.json"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        default=(
            "Artificial intelligence is"
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model = ModernGPT(
        **checkpoint["config"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    tokenizer = BPETokenizer.load(
        TOKENIZER
    )

    ids = tokenizer.encode(
        args.prompt
    )

    idx = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    result = model.generate(
        idx,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
    )

    print(
        tokenizer.decode(
            result[0].tolist()
        )
    )


if __name__ == "__main__":
    main()
