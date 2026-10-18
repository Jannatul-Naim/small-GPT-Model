"""Create retrieval chunks with source metadata."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data/raw/sources.jsonl"
OUT = ROOT / "data/raw/chunks.jsonl"

CHARS = 1800
OVERLAP = 250


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError("Run python -m dataset.collect")

    count = 0

    with OUT.open("w", encoding="utf-8") as f:
        for line in MANIFEST.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            if not line.strip():
                continue

            row = json.loads(line)
            text = row["text"]
            start = 0

            while start < len(text):
                end = min(len(text), start + CHARS)
                chunk = text[start:end].strip()

                if chunk:
                    f.write(
                        json.dumps(
                            {
                                "id": f"{row['id']}:{count}",
                                "source_id": row["id"],
                                "source": row["label"],
                                "kind": row["kind"],
                                "text": chunk,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    count += 1

                if end >= len(text):
                    break

                start = max(start + 1, end - OVERLAP)

    print(f"Chunks: {count:,}")


if __name__ == "__main__":
    main()