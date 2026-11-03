import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer
from training import config
from training import train


TINY = {
    "vocab_size": 300,
    "block_size": 16,
    "dim": 32,
    "n_layers": 1,
    "n_heads": 2,
    "n_kv_heads": 1,
    "ffn_dim": 64,
    "dropout": 0.0,
}


class TrainingRunTests(unittest.TestCase):
    """Run the real entry point on a tiny model and temporary paths."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)

        self.data = root / "processed"
        self.data.mkdir()
        self.ckpt = root / "checkpoints" / "gpt.pt"
        self.best = root / "checkpoints" / "gpt-best.pt"

        tok = BPETokenizer()
        tok.train("the quick brown fox " * 20, vocab_size=280, verbose=False)
        tok.save(self.data / "vocab.json")

        rng = np.random.default_rng(0)
        for name in ("train.bin", "val.bin", "new.bin", "replay.bin"):
            rng.integers(4, tok.vocab_size, 2000, dtype=np.uint16).tofile(self.data / name)

        self.write_meta(new_sources=1)

        for patcher in (
            mock.patch.object(train, "DATA", self.data),
            mock.patch.object(train, "CKPT", self.ckpt),
            mock.patch.object(train, "BEST", self.best),
            mock.patch.multiple(
                config,
                MODEL_SIZE="tiny",
                CONFIGS={"tiny": TINY},
                batch_size=2,
                gradient_accumulation_steps=1,
                warmup_steps=2,
                eval_interval=2,
                eval_iters=1,
                gradient_checkpointing=False,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_meta(self, new_sources):
        (self.data / "meta.json").write_text(json.dumps({"new_sources": new_sources}))

    def run_train(self, *args):
        with mock.patch("sys.argv", ["train", *args]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                train.main()
        return output.getvalue()

    def load(self, path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def test_new_refuses_to_replace_checkpoints(self):
        self.run_train("--mode", "new", "--steps", "2")
        written = self.best.stat().st_mtime_ns

        with self.assertRaises(SystemExit):
            self.run_train("--mode", "new", "--steps", "2")
        self.assertEqual(self.best.stat().st_mtime_ns, written)

        self.run_train("--mode", "new", "--steps", "2", "--overwrite")
        self.assertNotEqual(self.best.stat().st_mtime_ns, written)

    def test_checkpoint_dir_keeps_runs_apart(self):
        other = self.ckpt.parent.parent / "smoke"
        self.run_train("--mode", "new", "--steps", "2", "--checkpoint-dir", str(other))

        self.assertTrue((other / "gpt.pt").exists())
        self.assertTrue((other / "gpt-best.pt").exists())
        self.assertFalse(self.ckpt.exists())

    def test_data_dir_and_eval_interval_are_used(self):
        other = self.data.parent / "other"
        shutil.copytree(self.data, other)
        (other / "train.bin").write_bytes((self.data / "train.bin").read_bytes()[:1000])
        (self.data / "train.bin").unlink()

        output = self.run_train(
            "--mode", "new", "--steps", "3",
            "--data-dir", str(other),
            "--eval-interval", "1",
        )

        self.assertEqual(output.count("| val "), 3)

    def test_model_size_selects_the_profile_and_resume_keeps_it(self):
        with mock.patch.dict(config.CONFIGS, {"other": {**TINY, "n_layers": 2}}):
            self.run_train("--mode", "new", "--steps", "4", "--model-size", "other")
            self.assertEqual(self.load(self.ckpt)["config"]["n_layers"], 2)

        self.run_train("--mode", "resume", "--steps", "1")
        self.assertEqual(self.load(self.ckpt)["config"]["n_layers"], 2)

    def test_output_reports_training_loss(self):
        self.assertRegex(
            self.run_train("--mode", "new", "--steps", "2"),
            r"train \d+\.\d+ \| val \d+\.\d+",
        )

    def test_resume_continues_the_saved_schedule(self):
        self.run_train("--mode", "new", "--steps", "4")

        # Pretend the run was planned for 8 steps and interrupted after 4.
        ck = self.load(self.ckpt)
        ck["run"]["total_steps"] = 8
        torch.save(ck, self.ckpt)

        with mock.patch.object(train, "lr", wraps=train.lr) as schedule:
            self.run_train("--mode", "resume")

        self.assertEqual(
            [call.args[:2] for call in schedule.call_args_list],
            [(4, 8), (5, 8), (6, 8), (7, 8)],
        )
        self.assertEqual(self.load(self.ckpt)["step"], 7)

    def test_resume_of_finished_run_needs_steps(self):
        self.run_train("--mode", "new", "--steps", "2")

        with self.assertRaises(SystemExit):
            self.run_train("--mode", "resume")

        self.run_train("--mode", "resume", "--steps", "1")
        self.assertEqual(self.load(self.ckpt)["step"], 2)

    def test_resume_of_version_3_checkpoint_stays_at_minimum_lr(self):
        self.run_train("--mode", "new", "--steps", "1")
        tokhash = self.load(self.ckpt)["tokenizer_sha256"]

        cfg = dict(TINY, vocab_size=BPETokenizer.load(self.data / "vocab.json").vocab_size)
        model = ModernGPT(**cfg)
        opt = torch.optim.AdamW(model.parameters())
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "step": 9,
                "best_val": 5.0,
                "config": cfg,
                "tokenizer_sha256": tokhash,
                "version": 3,
            },
            self.ckpt,
        )

        output = self.run_train("--mode", "resume", "--steps", "1")
        minimum = config.learning_rate * config.min_lr_ratio
        self.assertIn(f"lr {minimum:.2e}", output)

    def test_resume_of_update_run_keeps_update_settings(self):
        self.run_train("--mode", "new", "--steps", "2")
        self.run_train("--mode", "update", "--steps", "2")

        output = self.run_train("--mode", "resume", "--steps", "1")
        self.assertIn("Continual update", output)
        self.assertEqual(self.load(self.ckpt)["run"]["peak_lr"], config.update_learning_rate)

    def test_update_only_replaces_best_when_it_improves(self):
        self.run_train("--mode", "new", "--steps", "2")
        base_step = self.load(self.best)["step"]

        with mock.patch.object(train, "evaluate", side_effect=[1.0, 2.0]):
            self.run_train("--mode", "update", "--steps", "2")
        self.assertEqual(self.load(self.best)["step"], base_step)
        self.assertEqual(self.load(self.ckpt)["run"]["mode"], "update")

        with mock.patch.object(train, "evaluate", side_effect=[1.0, 0.5]):
            self.run_train("--mode", "update", "--steps", "2")
        self.assertEqual(self.load(self.best)["run"]["mode"], "update")
        self.assertEqual(self.load(self.best)["best_val"], 0.5)

    def test_stops_when_data_files_do_not_match_meta(self):
        (self.data / "meta.json").write_text(
            json.dumps({"new_sources": 1, "val_sha256": "0" * 64})
        )

        with self.assertRaisesRegex(SystemExit, "val.bin does not match"):
            self.run_train("--mode", "new", "--steps", "2")

        self.assertFalse(self.ckpt.exists())

    def test_update_refuses_when_no_new_sources_were_built(self):
        self.run_train("--mode", "new", "--steps", "2")
        self.write_meta(new_sources=0)

        with self.assertRaises(SystemExit):
            self.run_train("--mode", "update", "--steps", "2")


class OptimizerTests(unittest.TestCase):
    def test_decay_skips_norm_scales_and_embedding(self):
        model = ModernGPT(**TINY)
        decay, no_decay = train.make_optimizer(model, 1e-3).param_groups

        self.assertEqual(no_decay["weight_decay"], 0.0)
        self.assertEqual(decay["weight_decay"], config.weight_decay)
        self.assertTrue(any(p is model.embedding.weight for p in no_decay["params"]))
        self.assertFalse(any(p is model.embedding.weight for p in decay["params"]))
        self.assertTrue(all(p.ndim == 2 for p in decay["params"]))
        self.assertEqual(
            len(decay["params"]) + len(no_decay["params"]),
            len(list(model.parameters())),
        )


if __name__ == "__main__":
    unittest.main()
