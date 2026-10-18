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
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterable

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


def load_previous() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding='utf-8', errors='ignore').splitlines():
            try:
                row = json.loads(line)
                if row.get('id'):
                    rows[row['id']] = row
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
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
    with CHUNKS.open('w', encoding='utf-8') as out:
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
def write_training_binaries(
    rows: list[dict], reuse_tokenizer: bool, vocab_size: int, val_percent: int
) -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if reuse_tokenizer:
        if not VOCAB.exists():
            raise FileNotFoundError('No tokenizer found; run without --reuse-tokenizer first.')
        tok = BPETokenizer.load(VOCAB)
    else:
        corpus = '\n\n'.join((r['text'] for r in rows))
        if len(corpus) < 100000:
            print('WARNING: corpus is below 100k characters; add more material for meaningful pretraining.')
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=vocab_size, min_frequency=2, verbose=True)
        tok.save(VOCAB)
    if tok.vocab_size > vocab_size:
        raise ValueError(f'Tokenizer overflow: {tok.vocab_size} > {vocab_size}')
    new_ids = set(json.loads(NEW_IDS.read_text(encoding='utf-8'))) if NEW_IDS.exists() else set()
    train: list[list[int]] = []
    val: list[list[int]] = []
    train_sources: list[str] = []
    val_sources: list[str] = []
    token_ids_by_source: dict[str, list[int]] = {}
    for row in rows:
        ids = tok.encode(row['text'])
        if len(ids) < 769:
            continue
        token_ids_by_source[row['id']] = ids
        h = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16)
        if h % 100 < val_percent:
            val.append(ids)
            val_sources.append(row['id'])
        else:
            train.append(ids)
            train_sources.append(row['id'])
    if not val and len(rows) >= 2:
        candidates = []
        for row in rows:
            ids = tok.encode(row['text'])
            if len(ids) >= 769:
                h = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16)
                candidates.append((h, row['id'], ids))
        if len(candidates) >= 2:
            candidates.sort()
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
    new_id_set = set(new_ids)
    new_tokens = [token_ids_by_source[sid] for sid in train_sources if sid in new_id_set]
    old_tokens = [token_ids_by_source[sid] for sid in train_sources if sid not in new_id_set]
    if new_id_set and (not new_tokens):
        raise ValueError(
            'New sources are present, but none landed in the training split. '
            'Add more new sources so at least one is assigned to training.'
        )
    if not new_tokens:
        new_tokens = old_tokens[:1]
    if not old_tokens:
        old_tokens = train[:1]

    from array import array

    def write_bin(path: Path, token_lists: list[list[int]]) -> int:
        total = 0
        with path.open('wb') as f:
            for ids in token_lists:
                array('H', ids).tofile(f)
                total += len(ids)
        return total

    train_n = write_bin(PROCESSED / 'train.bin', train)
    val_n = write_bin(PROCESSED / 'val.bin', val)
    new_n = write_bin(PROCESSED / 'new.bin', new_tokens if new_tokens else old_tokens[:1])
    replay_n = write_bin(PROCESSED / 'replay.bin', old_tokens if old_tokens else train[:1])

    def file_sha(path: Path) -> str:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        return h.hexdigest()

    split_path = PROCESSED / 'split.json'
    split_path.write_text(
        json.dumps(
            {
                'strategy': 'sha256(source_id) mod 100',
                'validation_percent': val_percent,
                'train_source_ids': sorted(train_sources),
                'validation_source_ids': sorted(val_sources),
            },
            indent=2,
        ),
        encoding='utf-8',
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
        'split_strategy': 'sha256(source_id) mod 100',
        'validation_percent': val_percent,
        'schema_version': 2,
    }
    META.write_text(json.dumps(meta, indent=2), encoding='utf-8')
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
    args.input.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
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
    content_hash_to_id = {r.get('sha256'): sid for sid, r in rows.items() if r.get('sha256')}

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
    write_jsonl(MANIFEST, ordered)
    NEW_IDS.write_text(json.dumps(sorted(new_ids), indent=2), encoding='utf-8')
    STATE.write_text(
        json.dumps(
            {
                'known_ids': sorted(rows),
                'last_new_count': len(new_ids),
                'schema_version': 2,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    CORPUS.write_text(
        '\n\n'.join(
            f"===== SOURCE: {r['label']} =====\n{r['text']}" for r in ordered
        ),
        encoding='utf-8',
    )
    chunk_count = build_chunks(ordered, args.chunk_chars, args.overlap)
    if not 1 <= args.val_percent <= 50:
        raise ValueError('--val-percent must be between 1 and 50')
    meta = write_training_binaries(ordered, args.reuse_tokenizer, args.vocab_size, args.val_percent)
    stats.update(
        {
            'sources_total': len(ordered),
            'new_sources': len(new_ids),
            'chunks': chunk_count,
            'corpus_characters': sum(r['characters'] for r in ordered),
            'schema_version': 2,
        }
    )
    STATS.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print('\n=== DATA ENGINE COMPLETE ===')
    print(json.dumps(stats, indent=2))
    print(json.dumps(meta, indent=2))
    print(f'Manifest : {MANIFEST}')
    print(f'Chunks   : {CHUNKS}')
    print(f"Training : {PROCESSED / 'train.bin'}")
    print(f"Valid    : {PROCESSED / 'val.bin'}")


if __name__ == '__main__':
    main()
