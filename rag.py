"""Local lexical retrieval for current/source-grounded answers."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT / "data" / "raw" / "chunks.jsonl"

# Keeps "node.js", "self-esteem", and "c++" whole, without making "learning."
# a different term from "learning".
WORD = re.compile(r"[A-Za-z0-9_+#]+(?:[.-][A-Za-z0-9_+#]+)*")


def toks(s):
    return [x.lower() for x in WORD.findall(s)]


class Retriever:
    def __init__(self, path=CHUNKS):
        self.path = Path(path)
        self.stamp = -1
        self.refresh()

    def refresh(self):
        """Rebuild the index only if the chunk file has changed."""
        stamp = (
            self.path.stat().st_mtime_ns
            if self.path.exists()
            else None
        )

        if stamp == self.stamp:
            return

        self.stamp = stamp
        self.chunks = []
        self.tf = []
        self.lengths = []
        self.post = defaultdict(list)
        self.df = Counter()
        self.n = 0
        self.avg_length = 1

        if stamp is None:
            return

        for line in self.path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():
            if not line.strip():
                continue

            self.chunks.append(json.loads(line))

        for i, c in enumerate(self.chunks):
            tf = Counter(toks(c["text"]))
            self.tf.append(tf)
            self.lengths.append(sum(tf.values()))

            for t in tf:
                self.df[t] += 1
                self.post[t].append(i)

        self.n = len(self.chunks)
        self.avg_length = max(1, sum(self.lengths) / max(1, self.n))

    def search(self, q, top_n=5):
        if not self.chunks:
            return []

        qt = toks(q)
        scores = Counter()

        for term in qt:
            idf = math.log(
                1
                + (
                    self.n - self.df.get(term, 0) + 0.5
                )
                / (
                    self.df.get(term, 0) + 0.5
                )
            )

            for i in self.post.get(term, []):
                tf = self.tf[i][term]
                dl = self.lengths[i]

                scores[i] += idf * (tf * 2.2) / (
                    tf + 1.2 * (0.25 + 0.75 * dl / self.avg_length)
                )

        return [
            self.chunks[i]
            for i, _ in scores.most_common(top_n)
        ]
