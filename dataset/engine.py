"""End-to-end data engine for Small-GPT.
Ingests local PDF/TXT/HTML files and optional URLs, applies deterministic
normalization + quality filtering + exact/version deduplication, writes a
source manifest and retrieval chunks, then builds tokenizer + uint16 training
binaries compatible with the existing training code.
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import os
import re
import stat
import tempfile
import time
import unicodedata
from array import array
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterable, Iterator

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from tokenizer.bpe import BPETokenizer


# Paths and configuration
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'data' / 'ingest'
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
MANIFEST = RAW / 'sources.jsonl'
CHUNKS = RAW / 'chunks.jsonl'
CORPUS = RAW / 'corpus.txt'
NEW_IDS = RAW / 'new_sources.json'
STATE = RAW / 'state.json'
STATS = RAW / 'stats.json'
VOCAB = PROCESSED / 'vocab.json'
META = PROCESSED / 'meta.json'
HEADERS = {'User-Agent': 'Small-GPT-Research/3.0 (educational research)'}
SUPPORTED = {'.pdf': 'pdf', '.txt': 'text', '.html': 'html', '.htm': 'html'}
DEFAULT_VOCAB_SIZE = 16384
DEFAULT_CHARS = 1800
DEFAULT_OVERLAP = 250
TEMP_PREFIX = '._tmp_'

# Writes that a power cut cannot leave half finished
@contextmanager
def atomic_open(path: Path, mode: str = 'w') -> Iterator[IO]:
    """Write to a temp file, then rename it over path once it is safely on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=TEMP_PREFIX, dir=path.parent)
    try:
        os.fchmod(fd, stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644)
        with os.fdopen(fd, mode, encoding=None if 'b' in mode else 'utf-8') as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    with atomic_open(path) as f:
        f.write(text)


def remove_leftover_temp_files() -> None:
    """Delete temp files left behind if the power went during a write."""
    for folder in (RAW, PROCESSED):
        for leftover in folder.rglob(f'{TEMP_PREFIX}*'):
            leftover.unlink()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# Text normalization and validation
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()


def normalize_text(text: str) -> str:
    """Make extracted text stable across PDF/HTML/TXT sources."""
    text = html_lib.unescape(text)
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\x00', '')
    text = unicodedata.normalize('NFKC', text)
    text = re.sub('(?<=\\w)-\\n(?=\\w)', '', text)
    text = re.sub('[ \\t]+', ' ', text)
    text = re.sub('\\n[ \\t]+', '\n', text)
    text = re.sub('\\n{3,}', '\n\n', text)
    text = ''.join((c for c in text if c in '\n\t' or c.isprintable()))
    return text.strip()


def quality(text: str, min_chars: int) -> tuple[bool, str]:
    if len(text) < min_chars:
        return (False, f'too-short<{min_chars}')
    letters = sum((c.isalpha() for c in text))
    if letters / max(1, len(text)) < 0.35:
        return (False, 'low-alpha-ratio')
    return (True, 'ok')

# Local and web source extraction
def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ''
        text = normalize_text(text)
        if text:
            pages.append(f'[Page {idx}]\n{text}')
    return '\n\n'.join(pages)


def extract_html_text(text: str) -> str:
    soup = BeautifulSoup(text, 'html.parser')
    for tag in soup(
        ['script', 'style', 'noscript', 'svg', 'nav', 'footer', 'header', 'form', 'aside']
    ):
        tag.decompose()
    return soup.get_text('\n')


def read_local(path: Path) -> tuple[str, str]:
    kind = SUPPORTED[path.suffix.lower()]
    if kind == 'pdf':
        return (extract_pdf(path), kind)
    raw = path.read_text(encoding='utf-8', errors='ignore')
    if kind == 'html':
        raw = extract_html_text(raw)
    return (raw, kind)


def scrape(url: str, timeout: int=30) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get('content-type', '').lower()
    if 'pdf' in content_type or url.lower().split('?')[0].endswith('.pdf'):
        import io
        reader = PdfReader(io.BytesIO(response.content))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            page_text = normalize_text(page.extract_text() or '')
            if page_text:
                pages.append(f'[Page {idx}]\n{page_text}')
        return '\n\n'.join(pages)
    return extract_html_text(response.text)

# Source manifest and retrieval chunks
def canonical_file_id(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    return f'file:{rel}'


def split_bucket(source_id: str) -> int:
    """Hash a source's base ID so all its versions land on one side of the split."""
    base_id = source_id.split('#sha256:')[0]
    return int(hashlib.sha256(base_id.encode()).hexdigest()[:8], 16)


def known_content_hashes(rows: dict[str, dict]) -> dict[str, str]:
    """Map content hashes to source IDs, including rows normalized by older code."""
    known: dict[str, str] = {}
    for sid, row in rows.items():
        known[sha256(normalize_text(row['text']))] = sid
        known[row['sha256']] = sid
    return known


def load_previous() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if MANIFEST.exists():
        with MANIFEST.open(encoding='utf-8', errors='ignore') as f:
            for number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    # Skipping the row would re-add its file under a new ID,
                    # which can move it across the train/validation split.
                    raise ValueError(
                        f'{MANIFEST} line {number} is damaged. Restore the file before rebuilding.'
                    ) from exc
                if row.get('id'):
                    rows[row['id']] = row
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with atomic_open(path) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def chunk_text(text: str, chars: int, overlap: int) -> Iterable[str]:
    if overlap >= chars:
        raise ValueError('chunk overlap must be smaller than chunk size')
    start = 0
    while start < len(text):
        end = min(len(text), start + chars)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        boundary = text.rfind('\n\n', start, end)
        if boundary <= start + chars // 3:
            boundary = text.rfind(' ', start, end)
        next_start = boundary + 1 if boundary > start else end
        start = max(start + 1, next_start - overlap)


def build_chunks(rows: list[dict], chars: int, overlap: int) -> int:
    count = 0
    with atomic_open(CHUNKS) as out:
        for row in rows:
            for chunk_idx, text in enumerate(chunk_text(row['text'], chars, overlap)):
                item = {
                    'id': f"{row['id']}#chunk:{chunk_idx}",
                    'source_id': row['id'],
                    'source': row['label'],
                    'kind': row['kind'],
                    'sha256': row['sha256'],
                    'chunk_index': chunk_idx,
                    'text': text,
                }
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
    return count

# Tokenizer and training binary generation
def load_or_train_tokenizer(corpus: str, vocab_size: int) -> BPETokenizer:
    """Keep the trained tokenizer in the cache so a rerun can skip training."""
    key = sha256(f'{vocab_size}\n{corpus}')
    path = PROCESSED / 'token_cache' / f'tokenizer-{key}.json'
    if path.exists():
        return BPETokenizer.load(path)
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab_size, min_frequency=2, verbose=True)
    tok.save(path)
    return tok


def encode_cached(tok: BPETokenizer, text: str, cache_dir: Path) -> list[int]:
    """Keep each source's tokens on disk so a rerun only encodes what is left."""
    path = cache_dir / f'{sha256(text)}.bin'
    if path.exists():
        cached = array('H')
        cached.frombytes(path.read_bytes())
        return cached.tolist()
    ids = tok.encode(text)
    with atomic_open(path, 'wb') as f:
        array('H', ids).tofile(f)
    return ids


def write_training_binaries(
    rows: list[dict],
    new_ids: Iterable[str],
    reuse_tokenizer: bool,
    vocab_size: int,
    val_percent: int,
) -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    new_ids = set(new_ids)
    if reuse_tokenizer:
        if not VOCAB.exists():
            raise FileNotFoundError('No tokenizer found; run without --reuse-tokenizer first.')
        tok = BPETokenizer.load(VOCAB)
    else:
        corpus = '\n\n'.join((r['text'] for r in rows))
        if len(corpus) < 100000:
            print('WARNING: corpus is below 100k characters; add more material for meaningful pretraining.')
        tok = load_or_train_tokenizer(corpus, vocab_size)
    if tok.vocab_size > vocab_size:
        raise ValueError(f'Tokenizer overflow: {tok.vocab_size} > {vocab_size}')
    cache_dir = PROCESSED / 'token_cache' / sha256(json.dumps(tok.merges))[:16]
    train: list[list[int]] = []
    val: list[list[int]] = []
    train_sources: list[str] = []
    val_sources: list[str] = []
    token_ids_by_source: dict[str, list[int]] = {}
    for index, row in enumerate(rows, start=1):
        ids = encode_cached(tok, row['text'], cache_dir)
        if index % 50 == 0 or index == len(rows):
            print(f'Encoded {index}/{len(rows)} sources', flush=True)
        if len(ids) < 769:
            continue
        token_ids_by_source[row['id']] = ids
        h = split_bucket(row['id'])
        if h % 100 < val_percent:
            val.append(ids)
            val_sources.append(row['id'])
        else:
            train.append(ids)
            train_sources.append(row['id'])
    if not val and len(rows) >= 2:
        candidates = sorted(
            (split_bucket(sid), sid, ids) for sid, ids in token_ids_by_source.items()
        )
        if len(candidates) >= 2:
            chosen = candidates[0]
            train = []
            val = []
            train_sources = []
            val_sources = []
            for h, sid, ids in candidates:
                if sid == chosen[1]:
                    val.append(ids)
                    val_sources.append(sid)
                else:
                    train.append(ids)
                    train_sources.append(sid)
    if not train or not val:
        raise ValueError(
            'Need at least two distinct, sufficiently long sources for train/validation splits.'
        )
    new_tokens = [token_ids_by_source[sid] for sid in train_sources if sid in new_ids]
    old_tokens = [token_ids_by_source[sid] for sid in train_sources if sid not in new_ids]
    if new_ids and (not new_tokens):
        raise ValueError(
            'New sources are present, but none landed in the training split. '
            'Add more new sources so at least one is assigned to training.'
        )
    if not new_tokens:
        new_tokens = old_tokens[:1]
    if not old_tokens:
        old_tokens = train[:1]

    def write_bin(path: Path, token_lists: list[list[int]]) -> int:
        # EOS marks each source boundary and is what teaches the model to stop.
        total = 0
        with atomic_open(path, 'wb') as f:
            for ids in token_lists:
                array('H', [*ids, tok.EOS]).tofile(f)
                total += len(ids) + 1
        return total

    # The existing build stays untouched until here. meta.json goes last, so the
    # trainer can tell if a power cut left a mix of old and new files.
    if not reuse_tokenizer:
        tok.save(VOCAB)
    train_n = write_bin(PROCESSED / 'train.bin', train)
    val_n = write_bin(PROCESSED / 'val.bin', val)
    new_n = write_bin(PROCESSED / 'new.bin', new_tokens if new_tokens else old_tokens[:1])
    replay_n = write_bin(PROCESSED / 'replay.bin', old_tokens if old_tokens else train[:1])

    split_path = PROCESSED / 'split.json'
    atomic_write_text(
        split_path,
        json.dumps(
            {
                'strategy': 'sha256(base_source_id) mod 100',
                'validation_percent': val_percent,
                'train_source_ids': sorted(train_sources),
                'validation_source_ids': sorted(val_sources),
            },
            indent=2,
        ),
    )
    meta = {
        'vocab_size': tok.vocab_size,
        'sources': len(rows),
        'new_sources': len(new_ids),
        'train_tokens': train_n,
        'val_tokens': val_n,
        'new_tokens': new_n,
        'replay_tokens': replay_n,
        'tokenizer_sha256': file_sha(VOCAB),
        'train_sha256': file_sha(PROCESSED / 'train.bin'),
        'val_sha256': file_sha(PROCESSED / 'val.bin'),
        'new_sha256': file_sha(PROCESSED / 'new.bin'),
        'replay_sha256': file_sha(PROCESSED / 'replay.bin'),
        'split_sha256': file_sha(split_path),
        'split_strategy': 'sha256(base_source_id) mod 100',
        'validation_percent': val_percent,
        'source_separator': 'EOS',
        'schema_version': 3,
    }
    atomic_write_text(META, json.dumps(meta, indent=2))
    return meta

# Pipeline entry point
def main() -> None:
    ap = argparse.ArgumentParser(
        description='Build Small-GPT training data from PDF/TXT/HTML and optional URLs.'
    )
    ap.add_argument(
        '--input',
        type=Path,
        default=DEFAULT_INPUT,
        help='Directory recursively scanned for PDF/TXT/HTML files',
    )
    ap.add_argument(
        '--links', type=Path, default=ROOT / 'data' / 'links.txt', help='Optional URL list'
    )
    ap.add_argument('--no-web', action='store_true', help='Skip links.txt')
    ap.add_argument(
        '--reuse-tokenizer', action='store_true', help='Reuse data/processed/vocab.json'
    )
    ap.add_argument('--min-chars', type=int, default=800)
    ap.add_argument('--chunk-chars', type=int, default=DEFAULT_CHARS)
    ap.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP)
    ap.add_argument('--vocab-size', type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument(
        '--val-percent',
        type=int,
        default=5,
        help='Approximate percentage of sources assigned to validation',
    )
    args = ap.parse_args()
    if not 1 <= args.val_percent <= 50:
        raise ValueError('--val-percent must be between 1 and 50')
    args.input.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    remove_leftover_temp_files()
    rows = load_previous()
    previous_ids = set(rows)
    new_ids: list[str] = []
    stats = {
        'files_seen': 0,
        'files_accepted': 0,
        'files_skipped': 0,
        'web_seen': 0,
        'web_accepted': 0,
        'web_failed': 0,
        'duplicates': 0,
        'changed_sources': 0,
    }
    content_hash_to_id = known_content_hashes(rows)

    def add(source_id: str, label: str, text: str, kind: str) -> bool:
        nonlocal rows
        normalized = normalize_text(text)
        ok, reason = quality(normalized, args.min_chars)
        if not ok:
            stats['files_skipped'] += 1 if kind != 'web' else 0
            print(f'SKIP [{kind}] {label}: {reason}')
            return False
        digest = sha256(normalized)
        existing = rows.get(source_id)
        if existing and existing.get('sha256') == digest:
            return
        duplicate_id = content_hash_to_id.get(digest)
        if duplicate_id and duplicate_id != source_id:
            stats['duplicates'] += 1
            print(f'DEDUP [{kind}] {label} -> {duplicate_id}')
            return False
        final_id = source_id
        if existing and existing.get('sha256') != digest:
            final_id = f'{source_id}#sha256:{digest[:12]}'
            stats['changed_sources'] += 1
        item = {
            'id': final_id,
            'label': label,
            'kind': kind,
            'sha256': digest,
            'characters': len(normalized),
            'text': normalized,
        }
        rows[final_id] = item
        content_hash_to_id[digest] = final_id
        if final_id not in previous_ids:
            new_ids.append(final_id)
        return True
    files = [p for p in args.input.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED]
    for path in sorted(files):
        stats['files_seen'] += 1
        try:
            text, kind = read_local(path)
            accepted = add(
                canonical_file_id(path, args.input),
                path.relative_to(args.input).as_posix(),
                text,
                kind,
            )
            if accepted:
                stats['files_accepted'] += 1
        except Exception as exc:
            stats['files_skipped'] += 1
            print(f'FAILED [{path.name}]: {exc}')
    if not args.no_web and args.links.exists():
        for raw in args.links.read_text(encoding='utf-8', errors='ignore').splitlines():
            url = raw.strip()
            if not url or url.startswith('#'):
                continue
            stats['web_seen'] += 1
            print(f'[WEB] {url}')
            try:
                if add(f'url:{url}', url, scrape(url), 'web'):
                    stats['web_accepted'] += 1
            except Exception as exc:
                stats['web_failed'] += 1
                print(f'FAILED [web] {url}: {exc}')
            time.sleep(0.2)
    ordered = [rows[k] for k in sorted(rows)]
    # The manifest is written last. Until then a power cut leaves the old one in
    # place, so a rerun finds the same new sources and builds the same files.
    meta = write_training_binaries(
        ordered, new_ids, args.reuse_tokenizer, args.vocab_size, args.val_percent
    )
    chunk_count = build_chunks(ordered, args.chunk_chars, args.overlap)
    atomic_write_text(
        CORPUS,
        '\n\n'.join(
            f"===== SOURCE: {r['label']} =====\n{r['text']}" for r in ordered
        ),
    )
    stats.update(
        {
            'sources_total': len(ordered),
            'new_sources': len(new_ids),
            'chunks': chunk_count,
            'corpus_characters': sum(r['characters'] for r in ordered),
            'schema_version': 2,
        }
    )
    atomic_write_text(STATS, json.dumps(stats, indent=2))
    atomic_write_text(NEW_IDS, json.dumps(sorted(new_ids), indent=2))
    atomic_write_text(
        STATE,
        json.dumps(
            {
                'known_ids': sorted(rows),
                'last_new_count': len(new_ids),
                'schema_version': 2,
            },
            indent=2,
        ),
    )
    write_jsonl(MANIFEST, ordered)
    print('\n=== DATA ENGINE COMPLETE ===')
    print(json.dumps(stats, indent=2))
    print(json.dumps(meta, indent=2))
    print(f'Manifest : {MANIFEST}')
    print(f'Chunks   : {CHUNKS}')
    print(f"Training : {PROCESSED / 'train.bin'}")
    print(f"Valid    : {PROCESSED / 'val.bin'}")


if __name__ == '__main__':
    main()
