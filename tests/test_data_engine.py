import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dataset import engine
from dataset.engine import (
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
                "text": " ".join(rng.choice(words) for _ in range(1500)),
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
                NEW_IDS=processed / "new_sources.json",
            ):
                meta = engine.write_training_binaries(rows, False, 300, 50)

            train = np.fromfile(processed / "train.bin", dtype=np.uint16)
            val = np.fromfile(processed / "val.bin", dtype=np.uint16)

        eos_total = int((train == BPETokenizer.EOS).sum() + (val == BPETokenizer.EOS).sum())
        self.assertEqual(eos_total, len(rows))
        self.assertEqual(train[-1], BPETokenizer.EOS)
        self.assertEqual(val[-1], BPETokenizer.EOS)
        self.assertEqual(meta["train_tokens"], len(train))
        self.assertEqual(meta["val_tokens"], len(val))


if __name__ == "__main__":
    unittest.main()
