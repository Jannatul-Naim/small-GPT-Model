from __future__ import annotations

import gc
import heapq
import json
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


class BPETokenizer:
    """
    Memory-efficient byte-level BPE tokenizer.

    Token IDs:
        0   PAD
        1   UNK
        2   BOS
        3   EOS
        4-259 raw byte tokens
        260+ learned merge tokens
    """

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    OFFSET = 4
    BYTE_VOCAB_SIZE = 256

    def __init__(self, vocab=None, merges=None):
        self.vocab = vocab or {}
        self.merges = [
            tuple(map(int, x))
            for x in (merges or [])
        ]

        self.merge_rank = {
            pair: i
            for i, pair in enumerate(self.merges)
        }

        self._byte_cache: dict[int, bytes] = {}

    @property
    def vocab_size(self) -> int:
        return (
            self.OFFSET
            + self.BYTE_VOCAB_SIZE
            + len(self.merges)
        )

    def _symbol_bytes(self, token: int) -> bytes:
        token = int(token)

        cached = self._byte_cache.get(token)

        if cached is not None:
            return cached

        if (
            self.OFFSET
            <= token
            < self.OFFSET + self.BYTE_VOCAB_SIZE
        ):
            value = bytes(
                (token - self.OFFSET,)
            )

            self._byte_cache[token] = value

            return value

        if token < self.OFFSET:
            raise ValueError(
                f"Special token {token} has no byte representation"
            )

        merge_index = (
            token
            - self.OFFSET
            - self.BYTE_VOCAB_SIZE
        )

        if not 0 <= merge_index < len(self.merges):
            raise ValueError(
                f"Invalid token ID: {token}"
            )

        left, right = self.merges[merge_index]

        value = (
            self._symbol_bytes(left)
            + self._symbol_bytes(right)
        )

        self._byte_cache[token] = value

        return value

    def _rebuild_vocab(self) -> None:
        vocab = {
            str(self.PAD): "<PAD>",
            str(self.UNK): "<UNK>",
            str(self.BOS): "<BOS>",
            str(self.EOS): "<EOS>",
        }

        for token in range(
            self.OFFSET,
            self.OFFSET
            + self.BYTE_VOCAB_SIZE
            + len(self.merges),
        ):
            vocab[str(token)] = (
                self._symbol_bytes(token)
                .decode(
                    "latin1",
                    errors="ignore",
                )
            )

        self.vocab = vocab

    @staticmethod
    def _initial_pair_counts(
        tokens: list[int],
    ):
        counts = Counter(
            zip(tokens, tokens[1:])
        )

        positions = defaultdict(set)

        for i, pair in enumerate(
            zip(tokens, tokens[1:])
        ):
            positions[pair].add(i)

        return counts, positions

    @staticmethod
    def _heap_push(
        heap,
        pair,
        count,
    ):
        if count > 0:
            heapq.heappush(
                heap,
                (-count, pair),
            )

    def train(
        self,
        text: str,
        vocab_size: int = 8192,
        min_frequency: int = 3,
        verbose: bool = True,
    ) -> None:

        if vocab_size < (
            self.OFFSET
            + self.BYTE_VOCAB_SIZE
        ):
            raise ValueError(
                "vocab_size must be at least "
                f"{self.OFFSET + self.BYTE_VOCAB_SIZE}"
            )

        if min_frequency < 1:
            raise ValueError(
                "min_frequency must be >= 1"
            )

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        data = text.encode("utf-8")

        if not data:
            raise ValueError(
                "Cannot train tokenizer on empty text"
            )

        self.merges = []
        self.merge_rank = {}
        self._byte_cache.clear()

        tokens = [
            self.OFFSET + b
            for b in data
        ]

        n = len(tokens)

        prev = array("i", [-1]) * n
        nxt = array("i", [-1]) * n

        for i in range(n - 1):
            nxt[i] = i + 1
            prev[i + 1] = i

        counts, positions = (
            self._initial_pair_counts(tokens)
        )

        heap = []

        for pair, count in counts.items():
            self._heap_push(
                heap,
                pair,
                count,
            )

        target_merges = (
            vocab_size
            - self.OFFSET
            - self.BYTE_VOCAB_SIZE
        )

        def add_pair(
            pair,
            left_index,
        ):
            if left_index < 0:
                return

            counts[pair] += 1

            positions[pair].add(
                left_index
            )

            self._heap_push(
                heap,
                pair,
                counts[pair],
            )

        def remove_pair(
            pair,
            left_index,
        ):
            if left_index < 0:
                return

            old = counts.get(
                pair,
                0,
            )

            if old <= 1:
                counts.pop(
                    pair,
                    None,
                )

                pos = positions.get(
                    pair
                )

                if pos is not None:
                    pos.discard(
                        left_index
                    )

                    if not pos:
                        positions.pop(
                            pair,
                            None,
                        )

            else:
                counts[pair] = old - 1

                pos = positions.get(
                    pair
                )

                if pos is not None:
                    pos.discard(
                        left_index
                    )

                self._heap_push(
                    heap,
                    pair,
                    old - 1,
                )

        for merge_number in range(
            target_merges
        ):

            best_pair = None
            best_count = 0

            while heap:
                neg_count, pair = (
                    heapq.heappop(heap)
                )

                current = counts.get(
                    pair,
                    0,
                )

                if (
                    current == -neg_count
                    and current >= min_frequency
                ):
                    best_pair = pair
                    best_count = current
                    break

            if best_pair is None:
                break

            left_token, right_token = (
                best_pair
            )

            new_id = (
                self.OFFSET
                + self.BYTE_VOCAB_SIZE
                + len(self.merges)
            )

            self.merges.append(
                best_pair
            )

            self.merge_rank[
                best_pair
            ] = len(self.merges) - 1

            occurrences = list(
                positions.get(
                    best_pair,
                    (),
                )
            )

            positions.pop(
                best_pair,
                None,
            )

            counts.pop(
                best_pair,
                None,
            )

            merged = 0

            for left in occurrences:

                right = nxt[left]

                if (
                    right == -1
                    or tokens[left]
                    != left_token
                    or tokens[right]
                    != right_token
                ):
                    continue

                left_left = prev[left]
                right_right = nxt[right]

                if left_left != -1:
                    remove_pair(
                        (
                            tokens[left_left],
                            tokens[left],
                        ),
                        left_left,
                    )

                remove_pair(
                    (
                        tokens[left],
                        tokens[right],
                    ),
                    left,
                )

                if right_right != -1:
                    remove_pair(
                        (
                            tokens[right],
                            tokens[right_right],
                        ),
                        right,
                    )

                tokens[left] = new_id

                nxt[left] = right_right

                if right_right != -1:
                    prev[right_right] = left

                prev[right] = -2
                nxt[right] = -2

                if left_left != -1:
                    add_pair(
                        (
                            tokens[left_left],
                            tokens[left],
                        ),
                        left_left,
                    )

                if right_right != -1:
                    add_pair(
                        (
                            tokens[left],
                            tokens[right_right],
                        ),
                        left,
                    )

                merged += 1

            if verbose and (
                merge_number < 10
                or (merge_number + 1) % 1000 == 0
            ):
                print(
                    f"  merge "
                    f"{merge_number + 1:>5}/"
                    f"{target_merges}: "
                    f"{left_token},{right_token} "
                    f"-> {new_id} "
                    f"(freq={best_count:,}, "
                    f"applied={merged:,})"
                )

            if merged == 0:
                break

        if verbose:
            print(
                f"Tokenizer training complete. "
                f"Merges: {len(self.merges)}"
            )

        # Release large training structures
        del counts
        del positions
        del heap
        del prev
        del nxt
        del tokens
        del data

        gc.collect()

        self._rebuild_vocab()

        gc.collect()

    def _encode_chunk(
        self,
        chunk: str,
    ) -> list[int]:

        data = chunk.encode("utf-8")

        if not data:
            return []

        tokens = [
            self.OFFSET + b
            for b in data
        ]

        n = len(tokens)

        if n == 1:
            return tokens

        prev = array("i", [-1]) * n
        nxt = array("i", [-1]) * n

        for i in range(n - 1):
            nxt[i] = i + 1
            prev[i + 1] = i

        heap = []

        for i in range(n - 1):

            pair = (
                tokens[i],
                tokens[i + 1],
            )

            rank = self.merge_rank.get(
                pair
            )

            if rank is not None:
                heapq.heappush(
                    heap,
                    (
                        rank,
                        i,
                        pair,
                    ),
                )

        while heap:

            rank, left, pair = (
                heapq.heappop(heap)
            )

            if left < 0 or left >= n:
                continue

            right = nxt[left]

            if right == -1:
                continue

            if (
                tokens[left]
                != pair[0]
                or tokens[right]
                != pair[1]
            ):
                continue

            current_rank = (
                self.merge_rank.get(pair)
            )

            if current_rank != rank:
                continue

            left_left = prev[left]
            right_right = nxt[right]

            merged_id = (
                self.OFFSET
                + self.BYTE_VOCAB_SIZE
                + rank
            )

            tokens[left] = merged_id

            nxt[left] = right_right

            if right_right != -1:
                prev[right_right] = left

            prev[right] = -2
            nxt[right] = -2

            if left_left != -1:

                new_pair = (
                    tokens[left_left],
                    tokens[left],
                )

                new_rank = (
                    self.merge_rank.get(
                        new_pair
                    )
                )

                if new_rank is not None:
                    heapq.heappush(
                        heap,
                        (
                            new_rank,
                            left_left,
                            new_pair,
                        ),
                    )

            if right_right != -1:

                new_pair = (
                    tokens[left],
                    tokens[right_right],
                )

                new_rank = (
                    self.merge_rank.get(
                        new_pair
                    )
                )

                if new_rank is not None:
                    heapq.heappush(
                        heap,
                        (
                            new_rank,
                            left,
                            new_pair,
                        ),
                    )

        result = []

        i = 0

        while prev[i] != -1:
            i = prev[i]

        while i != -1:
            result.append(
                tokens[i]
            )

            i = nxt[i]

        del tokens
        del prev
        del nxt
        del heap
        del data

        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        chunk_size: int = 8192,
    ) -> list[int]:

        if not text:
            if add_special_tokens:
                return [
                    self.BOS,
                    self.EOS,
                ]

            return []

        result = []

        total = len(text)

        for start in range(
            0,
            total,
            chunk_size,
        ):

            end = min(
                start + chunk_size,
                total,
            )

            chunk = text[
                start:end
            ]

            encoded = self._encode_chunk(
                chunk
            )

            result.extend(encoded)

            del encoded
            del chunk

            if (
                start // chunk_size
            ) % 8 == 0:

                gc.collect()

        if add_special_tokens:
            return [
                self.BOS,
                *result,
                self.EOS,
            ]

        return result

    def decode(
        self,
        ids: Iterable[int],
    ) -> str:

        output = bytearray()

        for token in ids:

            token = int(token)

            if token in (
                self.PAD,
                self.UNK,
                self.BOS,
                self.EOS,
            ):
                continue

            output.extend(
                self._symbol_bytes(
                    token
                )
            )

        return bytes(output).decode(
            "utf-8",
            errors="replace",
        )

    def save(
        self,
        path,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "vocab": self.vocab,
            "merges": [
                list(x)
                for x in self.merges
            ],
        }

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path,
    ):

        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        return cls(
            vocab=data["vocab"],
            merges=data["merges"],
        )