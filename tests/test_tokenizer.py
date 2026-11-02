import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path

from tokenizer.bpe import BPETokenizer


class TokenizerTests(unittest.TestCase):
    def test_round_trip_including_unseen_unicode(self):
        tok = BPETokenizer()
        tok.train("the quick brown fox jumps over the lazy dog " * 20, vocab_size=300, verbose=False)

        for text in ["the lazy fox", "café Ω 日本語 🙂", "line one\n\nline two"]:
            self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_save_and_load_preserve_encoding(self):
        tok = BPETokenizer()
        tok.train("the quick brown fox " * 20, vocab_size=280, verbose=False)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            tok.save(path)
            loaded = BPETokenizer.load(path)

        self.assertEqual(loaded.vocab_size, tok.vocab_size)
        self.assertEqual(loaded.encode("the quick fox"), tok.encode("the quick fox"))

    def test_training_skips_merged_away_nodes(self):
        # The ten a's allow at most five non-overlapping (a, a) merges. The
        # second-to-last token is also "a", so reading tokens[-2] through a
        # dead node's nxt == -2 would count phantom merges.
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            BPETokenizer().train("zaaaaaaaaz aa", vocab_size=261, min_frequency=2)

        applied = int(re.search(r"applied=(\d+)", output.getvalue()).group(1))
        self.assertLessEqual(applied, 5)


if __name__ == "__main__":
    unittest.main()
