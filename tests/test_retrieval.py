import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from rag import Retriever, toks


def write_chunks(path, texts):
    path.write_text(
        "\n".join(
            json.dumps({"id": str(i), "source": f"doc{i}", "text": text})
            for i, text in enumerate(texts)
        ),
        encoding="utf-8",
    )


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "chunks.jsonl"

    def test_terms_drop_trailing_punctuation_only(self):
        self.assertEqual(
            toks("Learning, learning. node.js self-esteem C++"),
            ["learning", "learning", "node.js", "self-esteem", "c++"],
        )

    def test_search_matches_a_word_next_to_punctuation(self):
        write_chunks(self.path, ["Children enjoy learning.", "Rivers flood in spring."])
        hits = Retriever(self.path).search("learning")
        self.assertEqual([h["source"] for h in hits], ["doc0"])

    def test_missing_index_returns_nothing(self):
        self.assertEqual(Retriever(self.path).search("anything"), [])

    def test_refresh_reloads_only_when_the_file_changes(self):
        write_chunks(self.path, ["alpha beta"])
        retriever = Retriever(self.path)
        chunks = retriever.chunks

        retriever.refresh()
        self.assertIs(retriever.chunks, chunks)

        write_chunks(self.path, ["gamma delta"])
        os.utime(self.path, ns=(0, self.path.stat().st_mtime_ns + 10 ** 9))
        retriever.refresh()
        self.assertEqual(retriever.search("gamma")[0]["source"], "doc0")
        self.assertEqual(retriever.search("alpha"), [])

    def test_search_cost_does_not_grow_with_corpus_per_match(self):
        # Averaging the corpus inside the scoring loop took minutes for
        # questions containing common words.
        words = " ".join(f"word{i}" for i in range(100))
        write_chunks(self.path, [f"the {words}"] * 3000)
        retriever = Retriever(self.path)

        started = time.perf_counter()
        retriever.search("the word1 word2 word3")
        self.assertLess(time.perf_counter() - started, 1.0)


if __name__ == "__main__":
    unittest.main()
