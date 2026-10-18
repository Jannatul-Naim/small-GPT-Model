"""Incremental, deduplicating corpus collector."""
from __future__ import annotations
from pathlib import Path
import hashlib, json, re, time
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
LINKS = ROOT / "data/links.txt"
TEXT_DIR = ROOT / "data/text"
TOPICS_DIR = ROOT / "data/topics"
RAW_DIR = ROOT / "data/raw"
MANIFEST = RAW_DIR / "sources.jsonl"
CORPUS = RAW_DIR / "corpus.txt"
NEW_IDS = RAW_DIR / "new_sources.json"
STATE = RAW_DIR / "state.json"
HEADERS = {"User-Agent": "Small-GPT-Research/3.0 (educational research)"}


def sha(text):
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def clean(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def quality(text):
    # Cheap deterministic quality gate; not a truth detector.
    if len(text) < 800:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters / max(1, len(text)) < 0.35:
        return False
    return True


def scrape(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    return clean(soup.get_text("\n"))


def read_rows():
    rows = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                x=json.loads(line); rows[x["id"]]=x
            except Exception: pass
    return rows


def main():
    for p in (TEXT_DIR, TOPICS_DIR, RAW_DIR): p.mkdir(parents=True, exist_ok=True)
    rows=read_rows()
    previous_known=set(json.loads(STATE.read_text()).get("known_ids", [])) if STATE.exists() else set()
    new_ids=[]

    def add(source_id, label, text, kind):
        text=clean(text)
        if not quality(text):
            print(f"      SKIP quality/too-short: {label}")
            return
        digest=sha(text)
        old=rows.get(source_id)
        # Same source + same content: no new version.
        if old and old.get("sha256")==digest:
            return
        # Changed source gets a content-version ID so historical material is retained.
        if old and old.get("sha256")!=digest:
            source_id=f"{source_id}#sha256:{digest[:12]}"
        is_new = source_id not in previous_known
        rows[source_id]={"id":source_id,"label":label,"kind":kind,"sha256":digest,"characters":len(text),"text":text}
        if is_new: new_ids.append(source_id)

    for path in sorted(TEXT_DIR.glob("*.txt")):
        print(f"[TXT] {path.name}")
        add(f"file:{path.resolve()}", path.name, path.read_text(encoding="utf-8", errors="ignore"), "text")
    for topic in sorted(p for p in TOPICS_DIR.iterdir() if p.is_dir()):
        for path in sorted(topic.rglob("*.txt")):
            print(f"[TOPIC:{topic.name}] {path.name}")
            add(f"topic:{topic.name}:{path.resolve()}", f"{topic.name}/{path.name}", path.read_text(encoding="utf-8", errors="ignore"), "topic")
    if LINKS.exists():
        for raw in LINKS.read_text(encoding="utf-8", errors="ignore").splitlines():
            url=raw.strip()
            if not url or url.startswith("#"): continue
            print(f"[WEB] {url}")
            try: add(f"url:{url}", url, scrape(url), "web")
            except Exception as e: print(f"      FAILED: {e}")
            time.sleep(.2)

    MANIFEST.write_text("\n".join(json.dumps(rows[k], ensure_ascii=False) for k in sorted(rows))+"\n", encoding="utf-8")
    NEW_IDS.write_text(json.dumps(new_ids, indent=2), encoding="utf-8")
    STATE.write_text(json.dumps({"known_ids": sorted(rows), "last_new_count": len(new_ids)}, indent=2), encoding="utf-8")
    CORPUS.write_text("\n\n".join(f"===== SOURCE: {rows[k]['label']} =====\n{rows[k]['text']}" for k in sorted(rows)), encoding="utf-8")
    print(f"Sources: {len(rows):,} | New this run: {len(new_ids):,} | chars: {CORPUS.stat().st_size:,}")

if __name__ == "__main__": main()
