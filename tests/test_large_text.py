import contextlib
import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dataset.large_text import BLOCK_CHARS, SEPARATOR, build, read_documents
from tokenizer.bpe import BPETokenizer


WORDS = ["once", "there", "was", "a", "little", "girl", "named", "Lily", "who", "loved", "her", "dog."]


def story(rng, words=200):
    return " ".join(rng.choice(WORDS) for _ in range(words))


class ReadDocumentsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "stories.txt"

    def test_splits_on_separator_and_skips_blanks(self):
        self.path.write_text(f"one\n{SEPARATOR}\n\n{SEPARATOR}\ntwo\n{SEPARATOR}\nthree", encoding="utf-8")
        self.assertEqual(list(read_documents(self.path)), ["one", "two", "three"])

    def test_finds_a_separator_cut_by_a_read_boundary(self):
        first = "a" * (BLOCK_CHARS - len(SEPARATOR) // 2)
        self.path.write_text(f"{first}{SEPARATOR}second", encoding="utf-8")
        self.assertEqual(list(read_documents(self.path)), [first, "second"])


class BuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        rng = random.Random(0)

        self.train_stories = [story(rng) for _ in range(20)]
        self.val_stories = [story(rng) for _ in range(5)]

        self.train = root / "train.txt"
        self.val = root / "val.txt"
        self.out = root / "out"
        self.train.write_text(f"\n{SEPARATOR}\n".join(self.train_stories), encoding="utf-8")
        self.val.write_text(f"\n{SEPARATOR}\n".join(self.val_stories), encoding="utf-8")

    def run_build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return build(self.train, self.val, self.out, 300)

    def tokens(self, name):
        return np.fromfile(self.out / name, dtype=np.uint16)

    def test_binaries_hold_every_document_in_order(self):
        meta = self.run_build()
        tok = BPETokenizer.load(self.out / "vocab.json")

        for name, stories in (("train.bin", self.train_stories), ("val.bin", self.val_stories)):
            tokens = self.tokens(name)
            ends = np.flatnonzero(tokens == tok.EOS)
            self.assertEqual(len(ends), len(stories))
            self.assertEqual(ends[-1], len(tokens) - 1)
            self.assertEqual(
                [tok.decode(part.tolist()) for part in np.split(tokens[:-1], ends[:-1] + 1)],
                stories,
            )

        self.assertTrue(tok.word_level)
        self.assertEqual(meta["train_documents"], 20)
        self.assertEqual(meta["val_documents"], 5)
        self.assertEqual(meta["train_tokens"], len(self.tokens("train.bin")))
        self.assertEqual(meta["val_tokens"], len(self.tokens("val.bin")))

    def test_meta_records_each_outputs_hash(self):
        meta = self.run_build()
        saved = json.loads((self.out / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, meta)

        for key, name in (
            ("tokenizer_sha256", "vocab.json"),
            ("train_sha256", "train.bin"),
            ("val_sha256", "val.bin"),
            ("split_sha256", "split.json"),
        ):
            self.assertEqual(hashlib.sha256((self.out / name).read_bytes()).hexdigest(), meta[key])

    def test_rerun_gives_identical_files_and_clears_leftovers(self):
        self.run_build()
        names = ["vocab.json", "train.bin", "val.bin", "split.json", "meta.json"]
        first = {name: (self.out / name).read_bytes() for name in names}

        leftover = self.out / "._tmp_partial"
        leftover.write_bytes(b"partial")
        self.run_build()

        self.assertFalse(leftover.exists())
        self.assertEqual({name: (self.out / name).read_bytes() for name in names}, first)

    def test_cut_during_encoding_leaves_no_partial_binary(self):
        self.run_build()
        before = (self.out / "train.bin").read_bytes()
        tok_encode = BPETokenizer.encode
        calls = []

        def fail_on_third_document(tok, text):
            calls.append(text)
            if len(calls) == 3:
                raise KeyboardInterrupt
            return tok_encode(tok, text)

        with mock.patch.object(BPETokenizer, "encode", fail_on_third_document):
            with self.assertRaises(KeyboardInterrupt):
                self.run_build()

        self.assertEqual((self.out / "train.bin").read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.out.glob("._tmp_*")), [])


if __name__ == "__main__":
    unittest.main()
