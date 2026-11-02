"""Build a word-level tokenizer and training binaries from large text files.

dataset.engine keeps every source's full text in memory and writes several copies
of it, which suits small corpora. This build streams from disk instead, so a
multi-gigabyte file needs little memory and little extra disk.

Example:
    python -m dataset.large_text \
        --train data/tinystories/raw/TinyStoriesV2-GPT4-train.txt \
        --val data/tinystories/raw/TinyStoriesV2-GPT4-valid.txt \
        --out data/tinystories

Rerun the same command after a power cut: every output is swapped in whole, and
the trainer refuses a mix of old and new files until the build has finished.
"""
from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path
from typing import Iterator

from dataset.engine import TEMP_PREFIX, atomic_open, atomic_write_text, file_sha
from tokenizer.bpe import BPETokenizer


SEPARATOR = '<|endoftext|>'
DEFAULT_VOCAB_SIZE = 4096
BLOCK_CHARS = 1 << 20
FLUSH_TOKENS = 1 << 20
REPORT_EVERY = 100_000


def read_documents(path: Path, separator: str = SEPARATOR) -> Iterator[str]:
    """Yield documents from a file without loading all of it into memory."""
    tail = ''
    with path.open(encoding='utf-8', errors='replace') as f:
        for block in iter(lambda: f.read(BLOCK_CHARS), ''):
            parts = (tail + block).split(separator)
            tail = parts.pop()
            for part in parts:
                text = part.strip()
                if text:
                    yield text
    text = tail.strip()
    if text:
        yield text


def write_tokens(tok: BPETokenizer, source: Path, target: Path, separator: str) -> tuple[int, int]:
    """Encode a file to uint16 tokens, with EOS after each document."""
    documents = 0
    tokens = 0
    buffer = array('H')
    with atomic_open(target, 'wb') as f:
        for text in read_documents(source, separator):
            buffer.extend(tok.encode(text))
            buffer.append(tok.EOS)
            documents += 1
            if len(buffer) >= FLUSH_TOKENS:
                tokens += len(buffer)
                buffer.tofile(f)
                del buffer[:]
            if documents % REPORT_EVERY == 0:
                print(f'{target.name}: {documents:,} documents', flush=True)
        tokens += len(buffer)
        buffer.tofile(f)
    return documents, tokens


def build(train: Path, val: Path, out: Path, vocab_size: int, separator: str = SEPARATOR) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for leftover in out.glob(f'{TEMP_PREFIX}*'):
        leftover.unlink()

    print('Training tokenizer...', flush=True)
    tok = BPETokenizer()
    tok.train_words(read_documents(train, separator), vocab_size=vocab_size)
    tok.save(out / 'vocab.json')
    print(f'Tokenizer: {tok.vocab_size:,} tokens', flush=True)

    train_documents, train_tokens = write_tokens(tok, train, out / 'train.bin', separator)
    val_documents, val_tokens = write_tokens(tok, val, out / 'val.bin', separator)

    split_path = out / 'split.json'
    atomic_write_text(
        split_path,
        json.dumps(
            {
                'strategy': 'separate files',
                'train_file': train.name,
                'train_file_sha256': file_sha(train),
                'validation_file': val.name,
                'validation_file_sha256': file_sha(val),
            },
            indent=2,
        ),
    )
    meta = {
        'vocab_size': tok.vocab_size,
        'sources': train_documents + val_documents,
        'new_sources': 0,
        'train_documents': train_documents,
        'val_documents': val_documents,
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'tokenizer_sha256': file_sha(out / 'vocab.json'),
        'train_sha256': file_sha(out / 'train.bin'),
        'val_sha256': file_sha(out / 'val.bin'),
        'split_sha256': file_sha(split_path),
        'split_strategy': 'separate files',
        'source_separator': 'EOS',
        'schema_version': 3,
    }
    atomic_write_text(out / 'meta.json', json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--train', type=Path, required=True, help='Training text file')
    ap.add_argument('--val', type=Path, required=True, help='Validation text file')
    ap.add_argument('--out', type=Path, required=True, help='Output data directory')
    ap.add_argument('--vocab-size', type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument('--separator', default=SEPARATOR, help='Text between documents')
    args = ap.parse_args()
    meta = build(args.train, args.val, args.out, args.vocab_size, args.separator)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
