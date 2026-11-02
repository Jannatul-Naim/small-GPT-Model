import contextlib
import io
import itertools
import json
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dataset import engine
from dataset.engine import (
    atomic_open,
    canonical_file_id,
    chunk_text,
    known_content_hashes,
    normalize_text,
    quality,
    read_local,
    sha256,
    split_bucket,
)
from tokenizer.bpe import BPETokenizer


class DataEngineTests(unittest.TestCase):
    def test_normalization(self):
        raw = "A\r\nB-\nC\n\n\nD\xa0E"
        self.assertEqual(normalize_text(raw), "A\nBC\n\nD E")

    def test_quality(self):
        self.assertEqual(quality("abc", 10), (False, "too-short<10"))
        self.assertEqual(quality("1234567890" * 10, 10), (False, "low-alpha-ratio"))
        self.assertEqual(quality("Research data " * 100, 100), (True, "ok"))

    def test_chunk_overlap_and_boundaries(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three " * 50
        chunks = list(chunk_text(text, 120, 20))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(x) <= 120 for x in chunks))

    def test_local_txt_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt = root / "note.txt"
            html = root / "page.html"
            txt.write_text("Machine learning " * 100, encoding="utf-8")
            html.write_text("<html><body><p>Machine learning</p><script>alert(1)</script></body></html>", encoding="utf-8")
            txt_text, txt_kind = read_local(txt)
            html_text, html_kind = read_local(html)
            self.assertEqual(txt_kind, "text")
            self.assertEqual(html_kind, "html")
            self.assertIn("Machine learning", txt_text)
            self.assertIn("Machine learning", html_text)
            self.assertNotIn("alert", html_text)

    def test_canonical_file_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "papers" / "paper.txt"
            nested.parent.mkdir()
            nested.touch()
            self.assertEqual(canonical_file_id(nested, root), "file:papers/paper.txt")

    def test_versions_of_a_source_share_a_split_bucket(self):
        self.assertEqual(
            split_bucket("file:a.txt#sha256:abc123"),
            split_bucket("file:a.txt"),
        )

    def test_rows_saved_by_older_normalization_are_recognized(self):
        # U+2011 survived the older cleaner but NFKC now rewrites it, so the
        # stored hash alone would not match a re-read of the same file.
        stored = "self‑esteem and resilience"
        rows = {
            "topic:old:/moved/path/a.txt": {
                "id": "topic:old:/moved/path/a.txt",
                "sha256": sha256(stored),
                "text": stored,
            }
        }
        self.assertNotEqual(sha256(normalize_text(stored)), sha256(stored))
        self.assertEqual(
            known_content_hashes(rows)[sha256(normalize_text(stored))],
            "topic:old:/moved/path/a.txt",
        )

    def test_binaries_separate_sources_with_eos(self):
        rng = random.Random(0)
        words = ["trust", "family", "memory", "habit", "culture", "routine", "identity"]
        rows = [
            {
                "id": f"file:doc{i}.txt",
                "text": " ".join(rng.choice(words) for _ in range(300)),
            }
            for i in range(6)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp)

            with mock.patch.multiple(
                engine,
                PROCESSED=processed,
                VOCAB=processed / "vocab.json",
                META=processed / "meta.json",
            ):
                meta = engine.write_training_binaries(rows, [], False, 265, 50)

            train = np.fromfile(processed / "train.bin", dtype=np.uint16)
            val = np.fromfile(processed / "val.bin", dtype=np.uint16)

        eos_total = int((train == BPETokenizer.EOS).sum() + (val == BPETokenizer.EOS).sum())
        self.assertEqual(eos_total, len(rows))
        self.assertEqual(train[-1], BPETokenizer.EOS)
        self.assertEqual(val[-1], BPETokenizer.EOS)
        self.assertEqual(meta["train_tokens"], len(train))
        self.assertEqual(meta["val_tokens"], len(val))

    def test_failed_write_keeps_the_old_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.txt"
            path.write_text("old", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                with atomic_open(path) as f:
                    f.write("new")
                    raise RuntimeError

            self.assertEqual(path.read_text(encoding="utf-8"), "old")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["data.txt"])


class PowerCut(Exception):
    pass


class RebuildSafetyTests(unittest.TestCase):
    """Run the real build on temporary paths, interrupting it at every write."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

        self.base = self.tmp / "base"
        for i in range(4):
            self.add_source(self.base, f"doc{i}", i)
        self.build(self.base)

    def add_source(self, root, name, seed):
        rng = random.Random(seed)
        words = ["trust", "family", "memory", "habit", "culture", "routine", "identity"]
        path = root / "input" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(" ".join(rng.choice(words) for _ in range(300)), encoding="utf-8")

    def add_new_training_source(self, root):
        # Choose a name the split sends to training, which an update requires.
        name = next(
            f"new{i}" for i in itertools.count()
            if split_bucket(f"file:new{i}.txt") % 100 >= 30
        )
        self.add_source(root, name, 99)
        return f"file:{name}.txt"

    def build(self, root, *extra):
        raw = root / "raw"
        processed = root / "processed"
        argv = [
            "engine",
            "--input", str(root / "input"),
            "--no-web",
            "--vocab-size", "265",
            "--val-percent", "30",
            *extra,
        ]

        with (
            mock.patch.multiple(
                engine,
                RAW=raw,
                PROCESSED=processed,
                MANIFEST=raw / "sources.jsonl",
                CHUNKS=raw / "chunks.jsonl",
                CORPUS=raw / "corpus.txt",
                NEW_IDS=raw / "new_sources.json",
                STATE=raw / "state.json",
                STATS=raw / "stats.json",
                VOCAB=processed / "vocab.json",
                META=processed / "meta.json",
            ),
            mock.patch("sys.argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            engine.main()

    def outputs(self, root):
        """Return every build output except the cache, which only affects speed."""
        return {
            p.relative_to(root).as_posix(): p.read_bytes()
            for folder in ("raw", "processed")
            for p in (root / folder).rglob("*")
            if p.is_file() and "token_cache" not in p.parts
        }

    def run_with_power_cut(self, root, cut_after, *extra):
        """Return False if the build finished before the cut was due."""
        real_replace = os.replace
        finished = 0

        def replace_until_cut(src, dst):
            nonlocal finished
            if finished == cut_after:
                raise PowerCut
            finished += 1
            real_replace(src, dst)

        with mock.patch("os.replace", replace_until_cut):
            try:
                self.build(root, *extra)
            except PowerCut:
                return True
        return False

    def test_rerun_after_a_cut_at_any_write_matches_a_clean_build(self):
        clean = self.tmp / "clean"
        shutil.copytree(self.base, clean)
        new_id = self.add_new_training_source(clean)
        self.build(clean, "--reuse-tokenizer")
        expected = self.outputs(clean)

        # The new source must survive the cut, or updates would refuse to run.
        self.assertEqual(json.loads(expected["raw/new_sources.json"]), [new_id])
        self.assertEqual(json.loads(expected["processed/meta.json"])["new_sources"], 1)

        for cut in itertools.count():
            root = self.tmp / f"cut{cut}"
            shutil.copytree(self.base, root)
            self.add_new_training_source(root)

            if not self.run_with_power_cut(root, cut, "--reuse-tokenizer"):
                break

            self.build(root, "--reuse-tokenizer")
            self.assertEqual(self.outputs(root), expected, f"cut after write {cut}")
            shutil.rmtree(root)

        self.assertGreater(cut, 8)

    def test_old_build_stays_whole_until_encoding_has_finished(self):
        before = self.outputs(self.base)
        self.add_new_training_source(self.base)

        # The first write only caches the new source's tokens.
        self.assertTrue(self.run_with_power_cut(self.base, 1, "--reuse-tokenizer"))
        self.assertEqual(self.outputs(self.base), before)

    def test_rerun_only_encodes_what_is_left(self):
        self.add_new_training_source(self.base)
        real_encode = BPETokenizer.encode

        with mock.patch.object(
            BPETokenizer, "encode", autospec=True, side_effect=real_encode
        ) as encode:
            self.build(self.base, "--reuse-tokenizer")

        self.assertEqual(encode.call_count, 1)

    def test_rerun_does_not_retrain_the_tokenizer(self):
        root = self.tmp / "fresh"
        for i in range(4):
            self.add_source(root, f"doc{i}", i)

        # The first write saves the trained tokenizer to the cache.
        self.assertTrue(self.run_with_power_cut(root, 1))
        self.assertFalse((root / "processed" / "vocab.json").exists())

        with mock.patch.object(BPETokenizer, "train", side_effect=AssertionError):
            self.build(root)

        self.assertEqual(self.outputs(root), self.outputs(self.base))

    def test_leftover_temp_files_are_removed(self):
        leftovers = [
            self.base / "raw" / "._tmp_abc",
            self.base / "processed" / "._tmp_vocab.json",
        ]
        cache = self.base / "processed" / "token_cache"
        leftovers.append(next(p for p in cache.iterdir() if p.is_dir()) / "._tmp_def")

        for path in leftovers:
            path.write_text("partial", encoding="utf-8")

        self.build(self.base, "--reuse-tokenizer")

        self.assertFalse(any(path.exists() for path in leftovers))

    def test_damaged_manifest_stops_the_build(self):
        manifest = self.base / "raw" / "sources.jsonl"
        damaged = manifest.read_bytes()[:-20]
        manifest.write_bytes(damaged)

        with self.assertRaisesRegex(ValueError, "is damaged"):
            self.build(self.base, "--reuse-tokenizer")

        self.assertEqual(manifest.read_bytes(), damaged)


if __name__ == "__main__":
    unittest.main()
