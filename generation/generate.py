
from pathlib import Path

import argparse
import torch

from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer
from training import config


# Paths
ROOT = Path(__file__).resolve().parents[1]

CKPT = ROOT / config.best_checkpoint_path
VOCAB = ROOT / "data/processed/vocab.json"


def main():
    # Parse command-line arguments
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=CKPT,
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=VOCAB.parent,
        help="Directory holding the checkpoint's vocab.json",
    )
    ap.add_argument(
        "--prompt",
        default="Artificial intelligence is",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
    )
    ap.add_argument(
        "--min-new-tokens",
        type=int,
        default=config.min_new_tokens,
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.75,
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=40,
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.92,
    )
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.08,
    )

    a = ap.parse_args()

    # Select device and load checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(
        a.checkpoint,
        map_location=device,
        weights_only=False,
    )

    # Load tokenizer and initialize model
    tok = BPETokenizer.load(a.data_dir / VOCAB.name)

    model = ModernGPT(**ck["config"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    # Encode prompt and prepare model input
    ids = tok.encode(a.prompt)

    idx = torch.tensor(
        [ids[-ck["config"]["block_size"]:]],
        dtype=torch.long,
        device=device,
    )

    # Generate and decode text
    out = model.generate(
        idx,
        a.max_new_tokens,
        a.temperature,
        a.top_k,
        a.top_p,
        a.repetition_penalty,
        tok.EOS,
        a.min_new_tokens,
    )

    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
