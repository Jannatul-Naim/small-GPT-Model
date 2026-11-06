"""Research training engine: pretraining, resume, and continual replay updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer
from training import config


# Paths
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/processed"
CKPT = ROOT / config.checkpoint_path
BEST = ROOT / config.best_checkpoint_path


def sha(p):
    """Calculate a file's SHA-256 checksum."""
    h = hashlib.sha256()

    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)

    return h.hexdigest()


def check_build(data_dir):
    """Stop if an interrupted data build left a mix of old and new files."""
    meta_path = data_dir / "meta.json"

    if not meta_path.exists():
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    for key, name in (
        ("tokenizer_sha256", "vocab.json"),
        ("train_sha256", "train.bin"),
        ("val_sha256", "val.bin"),
        ("new_sha256", "new.bin"),
        ("replay_sha256", "replay.bin"),
        ("split_sha256", "split.json"),
    ):
        if key in meta and sha(data_dir / name) != meta[key]:
            raise SystemExit(
                f"{name} does not match meta.json, so a data build was "
                "probably interrupted. Rerun the build to finish it."
            )


def load_bin(data_dir, name):
    """Load a binary token dataset as a memory-mapped array."""
    p = data_dir / name

    if not p.exists():
        raise FileNotFoundError(p)

    return np.memmap(p, dtype=np.uint16, mode="r")


def batch(data, block, bs, device, generator=None):
    """Create input and next-token target batches."""
    if len(data) <= block + 1:
        raise ValueError(
            f"Dataset {len(data)} tokens is too small for block {block}."
        )

    starts = torch.randint(
        0,
        len(data) - block - 1,
        (bs,),
        generator=generator,
    ).tolist()

    x = torch.stack([
        torch.from_numpy(
            np.asarray(data[i:i + block], dtype=np.int64)
        )
        for i in starts
    ])

    y = torch.stack([
        torch.from_numpy(
            np.asarray(data[i + 1:i + block + 1], dtype=np.int64)
        )
        for i in starts
    ])

    return x.to(device), y.to(device)


def amp(device):
    """Enable CUDA autocasting when training on a GPU."""
    return torch.autocast(
        "cuda",
        dtype=torch.float16,
        enabled=device == "cuda",
    )


@torch.no_grad()
def evaluate(model, data, cfg, device):
    """Evaluate loss using repeatable validation windows."""
    # Reuse the same validation windows at every checkpoint.
    # This makes loss comparisons meaningful instead of adding
    # fresh sampling noise each time.
    model.eval()
    vals = []

    generator = torch.Generator().manual_seed(config.seed + 10_000)

    for _ in range(config.eval_iters):
        x, y = batch(
            data,
            cfg["block_size"],
            config.batch_size,
            device,
            generator,
        )

        with amp(device):
            _, loss = model(x, y)

        vals.append(float(loss))

    model.train()

    return sum(vals) / len(vals)


def atomic(payload, path):
    """Save a checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        prefix="._ckpt_",
        dir=path.parent,
    )

    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def lr(step, total, peak):
    """Calculate the warmup and cosine-decay learning rate."""
    if step < config.warmup_steps:
        return peak * max(1, step + 1) / config.warmup_steps

    p = min(
        1.0,
        (step - config.warmup_steps)
        / max(1, total - config.warmup_steps),
    )

    c = 0.5 * (1 + math.cos(math.pi * p))

    return peak * (
        config.min_lr_ratio
        + (1 - config.min_lr_ratio) * c
    )


def make_optimizer(model, peak, param_groups=2):
    """Create AdamW without decaying normalization scales or the tied embedding."""
    if param_groups == 1:
        # Version 3 checkpoints saved one group and must resume with one.
        groups = model.parameters()
    else:
        embedding = model.embedding.weight
        decay = []
        no_decay = []

        for p in model.parameters():
            (no_decay if p.ndim < 2 or p is embedding else decay).append(p)

        groups = [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    return torch.optim.AdamW(
        groups,
        lr=peak,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


def payload(model, opt, step, best, cfg, tokhash, run):
    """Build a checkpoint containing model, optimizer, schedule, and RNG state."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "step": step,
        "best_val": best,
        "config": cfg,
        "tokenizer_sha256": tokhash,
        "rng_state": state,
        "run": run,
        "version": 4,
    }


def restore_rng(state):
    """Restore random-number generator states from a checkpoint."""
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # The checkpoint is loaded with map_location=device, which moves even
    # the CPU RNG ByteTensor to CUDA during GPU training. CPU RNG state must
    # be restored from a CPU ByteTensor.
    torch.set_rng_state(state["torch"].cpu())

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])


def main():
    # Parse training mode and step count
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        choices=["new", "resume", "update"],
        default="new",
    )
    ap.add_argument(
        "--steps",
        type=int,
    )
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CKPT.parent,
        help="Directory for this run's checkpoints",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --mode new to replace existing checkpoints",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="Directory holding train.bin, val.bin, and vocab.json",
    )
    ap.add_argument(
        "--eval-interval",
        type=int,
        default=config.eval_interval,
        help="Steps between validation runs and checkpoint saves",
    )
    ap.add_argument(
        "--model-size",
        choices=list(config.CONFIGS),
        default=config.MODEL_SIZE,
        help="Model profile for --mode new; other modes use the checkpoint's",
    )

    args = ap.parse_args()

    data_dir = args.data_dir
    ckpt_path = args.checkpoint_dir / CKPT.name
    best_path = args.checkpoint_dir / BEST.name

    if args.mode == "new" and not args.overwrite:
        existing = [str(p) for p in (ckpt_path, best_path) if p.exists()]

        if existing:
            raise SystemExit(
                f"Refusing to replace {', '.join(existing)}. Use "
                "--checkpoint-dir for a separate run, or --overwrite."
            )

    # Initialize random seeds and select the device
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB"
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Verify the prepared dataset
    if (
        not (data_dir / "train.bin").exists()
        or not (data_dir / "val.bin").exists()
        or not (data_dir / "vocab.json").exists()
    ):
        raise FileNotFoundError("Prepare data first.")

    check_build(data_dir)

    tok = BPETokenizer.load(data_dir / "vocab.json")
    tokhash = sha(data_dir / "vocab.json")
    val = load_bin(data_dir, "val.bin")

    cfg = dict(config.CONFIGS[args.model_size])
    cfg["vocab_size"] = tok.vocab_size

    # Initialize a new training run
    if args.mode == "new":
        model = ModernGPT(**cfg).to(device)
        opt = make_optimizer(model, config.learning_rate)

        start = 0
        best = float("inf")
        run = {
            "mode": "new",
            "peak_lr": config.learning_rate,
            "total_steps": args.steps or config.max_steps,
        }
        steps = run["total_steps"]

    # Resume a previous training run
    elif args.mode == "resume":
        ck = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        if ck.get("tokenizer_sha256") != tokhash:
            raise ValueError(
                "Checkpoint tokenizer hash does not match "
                "data/processed/vocab.json. Restore the matching "
                "tokenizer or start a new run."
            )

        cfg = dict(ck["config"])
        model = ModernGPT(**cfg).to(device)
        model.load_state_dict(ck["model_state"])

        start = int(ck["step"]) + 1
        best = float(ck.get("best_val", float("inf")))

        # Older checkpoints did not record their schedule. Treating it as
        # finished holds the minimum learning rate instead of re-warming.
        run = ck.get("run") or {
            "mode": "new",
            "peak_lr": config.learning_rate,
            "total_steps": start,
        }

        opt = make_optimizer(
            model,
            run["peak_lr"],
            len(ck["optimizer_state"]["param_groups"]),
        )
        opt.load_state_dict(ck["optimizer_state"])

        restore_rng(ck.get("rng_state"))
        del ck

        steps = args.steps or run["total_steps"] - start

        if steps <= 0:
            raise SystemExit(
                f"The checkpoint's run ended at step {run['total_steps']}. "
                "Pass --steps to train further at the minimum learning rate."
            )

        print("Resuming:", start)

    # Continue training the best checkpoint with new data and replay data
    else:
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))

        if not meta.get("new_sources"):
            raise SystemExit(
                "The last data build found no new sources, so new.bin holds "
                "only fallback data. Add material and rebuild before updating."
            )

        ck = torch.load(
            best_path,
            map_location="cpu",
            weights_only=False,
        )

        if ck.get("tokenizer_sha256") != tokhash:
            raise ValueError(
                "Checkpoint tokenizer hash does not match "
                "data/processed/vocab.json. Continual updates "
                "must reuse the original tokenizer."
            )

        cfg = dict(ck["config"])
        model = ModernGPT(**cfg).to(device)
        model.load_state_dict(ck["model_state"])
        del ck

        opt = make_optimizer(model, config.update_learning_rate)

        start = 0
        run = {
            "mode": "update",
            "peak_lr": config.update_learning_rate,
            "total_steps": args.steps or config.max_steps,
        }
        steps = run["total_steps"]

        # Score the base model first so that gpt-best.pt is only replaced by
        # an update that really improves on it.
        best = evaluate(model, val, cfg, device)
        print(f"Base validation loss: {best:.4f}")

    if run["mode"] == "update":
        new = load_bin(data_dir, "new.bin")
        replay = load_bin(data_dir, "replay.bin")

        print(
            f"Continual update: new={len(new):,} "
            f"replay={len(replay):,} tokens; "
            f"replay ratio={config.replay_ratio}"
        )
    else:
        train = load_bin(data_dir, "train.bin")

    # Configure the model and training loop
    model.gradient_checkpointing = config.gradient_checkpointing

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        "Context:",
        cfg["block_size"],
        "| effective tokens/step:",
        config.batch_size
        * config.gradient_accumulation_steps
        * cfg["block_size"],
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device == "cuda",
    )

    opt.zero_grad(set_to_none=True)

    train_loss = 0.0
    train_steps = 0

    # Training loop
    for step in range(start, start + steps):
        current_lr = lr(step, run["total_steps"], run["peak_lr"])

        for g in opt.param_groups:
            g["lr"] = current_lr

        for micro in range(config.gradient_accumulation_steps):
            if run["mode"] == "update":
                # Independent batches approximate the token-level replay ratio.
                use_replay = (
                    torch.rand(()) < config.replay_ratio
                ).item()

                source = replay if use_replay else new

            else:
                source = train

            x, y = batch(
                source,
                cfg["block_size"],
                config.batch_size,
                device,
            )

            with amp(device):
                _, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            train_loss += loss.item()

        train_steps += 1

        scaler.unscale_(opt)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip,
        )

        scaler.step(opt)
        scaler.update()

        opt.zero_grad(set_to_none=True)

        # Periodically evaluate and save checkpoints
        if (
            (step + 1) % args.eval_interval == 0
            or step == start + steps - 1
        ):
            v = evaluate(model, val, cfg, device)

            # Flush so that a log written through tee survives a power cut.
            print(
                f"step {step:6d} | "
                f"train {train_loss / train_steps:.4f} | "
                f"val {v:.4f} | "
                f"lr {current_lr:.2e}",
                flush=True,
            )

            train_loss = 0.0
            train_steps = 0

            if v < best:
                best = v

                atomic(
                    payload(model, opt, step, best, cfg, tokhash, run),
                    best_path,
                )

                print("best checkpoint saved", flush=True)

            atomic(
                payload(model, opt, step, best, cfg, tokhash, run),
                ckpt_path,
            )

    # Save the final checkpoint
    final = start + steps - 1

    atomic(
        payload(model, opt, final, best, cfg, tokhash, run),
        ckpt_path,
    )

    print("Training complete:", ckpt_path)


if __name__ == "__main__":
    main()
