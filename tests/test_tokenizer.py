import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path

from tokenizer.bpe import WORD_PIECE, BPETokenizer


WORD_LEVEL_TEXT = "the quick brown fox jumps over the lazy dog. The fox said: don't stop! " * 30


class WordLevelTokenizerTests(unittest.TestCase):
    def train(self, vocab_size=320):
        tok = BPETokenizer()
        tok.train_words([WORD_LEVEL_TEXT], vocab_size=vocab_size, verbose=False)
        return tok

    def test_pieces_keep_every_character(self):
        for text in [
            "Hello, world!  Two  spaces\n\nnew line\ttab ",
            "snake_case __x__ 3.14 $100 café Ω 日本語 \U0001f642 x² ½",
            "   leading and trailing   ",
            "\n\n\n",
        ]:
            self.assertEqual("".join(WORD_PIECE.findall(text)), text)

    def test_round_trip_including_unseen_text(self):
        tok = self.train()

        for text in [
            "the lazy fox",
            "Unseen   words, with\n\nspaces and café Ω 日本語 \U0001f642 snake_case_9",
        ]:
            self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_tokens_never_span_two_words(self):
        tok = self.train(vocab_size=400)
        self.assertGreater(len(tok.merges), 30)

        for token in range(260, tok.vocab_size):
            self.assertLessEqual(len(tok.decode([token]).split()), 1)

    def test_common_words_become_single_tokens(self):
        tok = self.train(vocab_size=400)
        self.assertEqual(len(tok.encode(" fox")), 1)

    def test_training_is_deterministic(self):
        self.assertEqual(self.train().merges, self.train().merges)

    def test_save_and_load_keep_word_level_mode(self):
        tok = self.train()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            tok.save(path)
            loaded = BPETokenizer.load(path)

        self.assertTrue(loaded.word_level)
        self.assertEqual(loaded.encode(WORD_LEVEL_TEXT), tok.encode(WORD_LEVEL_TEXT))

    def test_stops_when_no_pair_is_frequent_enough(self):
        tok = BPETokenizer()
        tok.train_words(["ab cd"], vocab_size=1000, verbose=False)
        self.assertEqual(tok.merges, [])

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            BPETokenizer().train_words([], verbose=False)


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
        self.assertFalse(loaded.word_level)

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
