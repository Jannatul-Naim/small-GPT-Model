from pathlib import Path
import argparse
import json

import torch

from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer
from training import config


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT / "data/processed/dataset.pt"
)

TOKENIZER = (
    ROOT / "data/processed/vocab.json"
)

CHECKPOINT = (
    ROOT / config.checkpoint_path
)


def get_batch(
    data,
    block_size,
    batch_size,
    device,
):
    max_start = (
        len(data)
        - block_size
        - 1
    )

    if max_start <= 0:
        raise ValueError(
            "Dataset is smaller than "
            "the context window."
        )

    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,),
    )

    x = torch.stack(
        [
            data[
                i:i + block_size
            ]
            for i in starts
        ]
    )

    y = torch.stack(
        [
            data[
                i + 1:i + block_size + 1
            ]
            for i in starts
        ]
    )

    return (
        x.to(device=device, dtype=torch.long, non_blocking=(device == "cuda")),
        y.to(device=device, dtype=torch.long, non_blocking=(device == "cuda")),
    )


@torch.no_grad()
def evaluate(
    model,
    train_data,
    val_data,
    cfg,
    device,
):
    model.eval()

    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data),
    ]:
        losses = []

        for _ in range(
            config.eval_iters
        ):
            x, y = get_batch(
                data,
                cfg["block_size"],
                config.batch_size,
                device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    device == "cuda"
                ),
            ):
                _, loss = model(x, y)

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


def build_model(
    cfg,
    device,
):
    return ModernGPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        dim=cfg["dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        ffn_dim=cfg["ffn_dim"],
        dropout=cfg["dropout"],
    ).to(device)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "new",
            "resume",
            "update",
        ],
        default="new",
    )

    args = parser.parse_args()

    torch.manual_seed(
        config.seed
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    if not DATASET.exists():
        raise FileNotFoundError(
            "Prepare the dataset first:\n"
            "python -m dataset.collect\n"
            "python -m dataset.prepare"
        )

    package = torch.load(
        DATASET,
        map_location="cpu",
    )

    train_data = package["train"]
    val_data = package["val"]

    tokenizer = BPETokenizer.load(
        TOKENIZER
    )

    cfg = dict(
        config.CONFIGS[
            config.MODEL_SIZE
        ]
    )

    if args.mode == "new":
        cfg["vocab_size"] = (
            tokenizer.vocab_size
        )

        model = build_model(
            cfg,
            device,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        start_step = 0
        best_val = float("inf")

    else:
        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                "No checkpoint found. "
                "Run --mode new first."
            )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location=device,
        )

        cfg = checkpoint["config"]

        if (
            cfg["vocab_size"]
            != tokenizer.vocab_size
        ):
            raise ValueError(
                "Tokenizer vocabulary changed. "
                "Do not rebuild the tokenizer "
                "when resuming/updating."
            )

        model = build_model(
            cfg,
            device,
        )

        model.load_state_dict(
            checkpoint[
                "model_state"
            ]
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state"
            ]
        )

        start_step = (
            checkpoint["step"] + 1
        )

        best_val = checkpoint.get(
            "best_val",
            float("inf"),
        )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameters:,} "
        f"({parameters / 1e9:.3f}B)"
    )

    if config.MODEL_SIZE == "1.8b":
        print()
        print(
            "WARNING: 1.8B training is "
            "not practical on a 12GB RTX 3060."
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device == "cuda"
        ),
    )

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    for step in range(
        start_step,
        start_step + config.max_steps,
    ):
        if (
            step
            % config.eval_interval
            == 0
        ):
            losses = evaluate(
                model,
                train_data,
                val_data,
                cfg,
                device,
            )

            print(
                f"step {step:6d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

            if (
                losses["val"]
                < best_val
            ):
                best_val = (
                    losses["val"]
                )

                torch.save(
                    {
                        "model_state":
                            model.state_dict(),

                        "optimizer_state":
                            optimizer.state_dict(),

                        "step": step,

                        "best_val":
                            best_val,

                        "config": cfg,

                        "model_size":
                            config.MODEL_SIZE,

                        "tokenizer_vocab_size":
                            tokenizer.vocab_size,
                    },
                    CHECKPOINT,
                )

                print(
                    "checkpoint saved"
                )

        for _ in range(
            config.gradient_accumulation_steps
        ):
            x, y = get_batch(
                train_data,
                cfg["block_size"],
                config.batch_size,
                device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    device == "cuda"
                ),
            ):
                _, loss = model(x, y)

                loss = (
                    loss
                    / config.gradient_accumulation_steps
                )

            scaler.scale(
                loss
            ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        optimizer.zero_grad(
            set_to_none=True
        )

    print(
        "Training complete."
    )


if __name__ == "__main__":
    main()
