from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import torch

from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer
from training import config


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/dataset.pt"
TOKENIZER = ROOT / "data/processed/vocab.json"
TOKENIZER_META = ROOT / "data/processed/meta.json"
CHECKPOINT = ROOT / config.checkpoint_path
BEST_CHECKPOINT = ROOT / config.best_checkpoint_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_batch(data, block_size: int, batch_size: int, device: str):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset is smaller than the context window.")

    starts = torch.randint(0, max_start + 1, (batch_size,))

    x = torch.stack(
        [data[i:i + block_size] for i in starts]
    )
    y = torch.stack(
        [data[i + 1:i + block_size + 1] for i in starts]
    )

    # The processed dataset uses uint16 to reduce host RAM and disk usage.
    # nn.Embedding requires integer token indices, so conversion happens here.
    x = x.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
    y = y.to(device=device, dtype=torch.long, non_blocking=(device == "cuda"))
    return x, y


def autocast_context(device: str):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(device == "cuda"),
    )


@torch.no_grad()
def evaluate(model, train_data, val_data, cfg, device):
    model.eval()
    results = {}

    for name, data in (("train", train_data), ("val", val_data)):
        losses = []

        for _ in range(config.eval_iters):
            x, y = get_batch(
                data,
                cfg["block_size"],
                config.batch_size,
                device,
            )

            with autocast_context(device):
                _, loss = model(x, y)

            losses.append(float(loss.item()))

        results[name] = sum(losses) / len(losses)

    model.train()
    return results


def build_model(cfg, device):
    model = ModernGPT(**cfg).to(device)
    model.gradient_checkpointing = bool(config.gradient_checkpointing)
    return model


def atomic_torch_save(payload, path: Path):
    """Write a checkpoint safely on the same filesystem, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)

    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(
                payload,
                handle,
                _use_new_zipfile_serialization=True,
            )
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temp_path, path)

        # Best-effort directory sync for crash resilience on Linux filesystems.
        try:
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def checkpoint_payload(model, optimizer, step, best_val, cfg, tokenizer_hash):
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": int(step),
        "best_val": float(best_val),
        "config": dict(cfg),
        "model_size": config.MODEL_SIZE,
        "tokenizer_vocab_size": int(cfg["vocab_size"]),
        "tokenizer_sha256": tokenizer_hash,
    }


def validate_checkpoint_tokenizer(checkpoint, tokenizer):
    saved_vocab = checkpoint.get("tokenizer_vocab_size")
    if saved_vocab != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary is incompatible with the checkpoint. "
            f"checkpoint={saved_vocab}, current={tokenizer.vocab_size}. "
            "For resume/update, keep the tokenizer fixed."
        )

    saved_hash = checkpoint.get("tokenizer_sha256")
    current_hash = file_sha256(TOKENIZER)

    # Old checkpoints may not contain the fingerprint. They remain loadable,
    # but new checkpoints always record it.
    if saved_hash and saved_hash != current_hash:
        raise ValueError(
            "Tokenizer file changed since the checkpoint was created. "
            "Resume/update requires the same tokenizer."
        )

    return current_hash


def main():
    parser = argparse.ArgumentParser(
        description="Train Small-GPT on a single RTX 3060 or CPU."
    )
    parser.add_argument(
        "--mode",
        choices=["new", "resume", "update"],
        default="new",
        help=(
            "new = fresh model; resume = continue optimizer/step; "
            "update = continue model weights with a fresh optimizer and step counter"
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cuda":
        torch.cuda.empty_cache()
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f"GPU VRAM: {props.total_memory / 1024**3:.2f} GiB")

    if not DATASET.exists() or not TOKENIZER.exists():
        raise FileNotFoundError(
            "Prepare the dataset first:\n"
            "python -m dataset.collect\n"
            "python -m dataset.prepare"
        )

    package = torch.load(
        DATASET,
        map_location="cpu",
        weights_only=False,
    )
    train_data = package["train"]
    val_data = package["val"]

    tokenizer = BPETokenizer.load(TOKENIZER)
    tokenizer_hash = file_sha256(TOKENIZER)

    cfg = dict(config.CONFIGS[config.MODEL_SIZE])

    if args.mode == "new":
        cfg["vocab_size"] = tokenizer.vocab_size
        model = build_model(cfg, device)
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
                "No checkpoint found. Run --mode new first."
            )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location=device,
            weights_only=False,
        )

        validate_checkpoint_tokenizer(checkpoint, tokenizer)
        cfg = dict(checkpoint["config"])

        if args.mode == "resume":
            model = build_model(cfg, device)
            model.load_state_dict(checkpoint["model_state"])

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_step = int(checkpoint["step"]) + 1
            best_val = float(
                checkpoint.get("best_val", float("inf"))
            )

        else:  # update
            # Preserve learned weights, but start a clean optimizer because the
            # dataset has changed. This is an explicit continual-training mode.
            model = build_model(cfg, device)
            model.load_state_dict(checkpoint["model_state"])

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate * 0.5,
                weight_decay=config.weight_decay,
            )
            start_step = 0
            best_val = float(
                checkpoint.get("best_val", float("inf"))
            )
            print(
                "Update mode: model weights restored; "
                "optimizer and step counter reset."
            )

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameters:,} ({parameters / 1e6:.2f}M)")
    print(
        f"Batch: {config.batch_size} | "
        f"Accumulation: {config.gradient_accumulation_steps} | "
        f"Context: {cfg['block_size']}"
    )
    print(
        f"Effective tokens/optimizer step: "
        f"{config.batch_size * config.gradient_accumulation_steps * cfg['block_size']:,}"
    )
    print(
        "Gradient checkpointing:",
        "ON" if config.gradient_checkpointing else "OFF",
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device == "cuda"),
    )

    optimizer.zero_grad(set_to_none=True)

    end_step = start_step + config.max_steps

    for step in range(start_step, end_step):
        if step % config.eval_interval == 0:
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

            payload = checkpoint_payload(
                model,
                optimizer,
                step,
                best_val,
                cfg,
                tokenizer_hash,
            )

            # Always keep a valid latest checkpoint for resume.
            atomic_torch_save(payload, CHECKPOINT)
            print("latest checkpoint saved")

            if losses["val"] < best_val:
                best_val = losses["val"]
                payload = checkpoint_payload(
                    model,
                    optimizer,
                    step,
                    best_val,
                    cfg,
                    tokenizer_hash,
                )
                atomic_torch_save(payload, BEST_CHECKPOINT)
                print("best checkpoint saved")

        for _ in range(config.gradient_accumulation_steps):
            x, y = get_batch(
                train_data,
                cfg["block_size"],
                config.batch_size,
                device,
            )

            with autocast_context(device):
                _, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip,
        )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # Save the final state even when the last step is not an evaluation step.
    final_step = end_step - 1
    atomic_torch_save(
        checkpoint_payload(
            model,
            optimizer,
            final_step,
            best_val,
            cfg,
            tokenizer_hash,
        ),
        CHECKPOINT,
    )

    print("Training complete.")
    print(f"Latest checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
