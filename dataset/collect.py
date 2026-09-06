from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]

LINKS = ROOT / "data/links.txt"
TEXT_DIR = ROOT / "data/text"
RAW_DIR = ROOT / "data/raw"
OUTPUT = RAW_DIR / "corpus.txt"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; MyGPT-Educational/1.0)"
    )
}


def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text,
    )
    return text.strip()


def scrape(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
        "form",
        "aside",
    ]):
        tag.decompose()

    return clean_text(
        soup.get_text("\n")
    )


def main():
    TEXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep previous corpus so repeated collection
    # accumulates data instead of deleting it.
    pieces = []

    if OUTPUT.exists():
        old = OUTPUT.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        if old.strip():
            pieces.append(old)

    if LINKS.exists():
        for line in LINKS.read_text(
            encoding="utf-8"
        ).splitlines():

            url = line.strip()

            if (
                not url
                or url.startswith("#")
            ):
                continue

            print(f"[WEB] {url}")

            try:
                text = scrape(url)

                if text:
                    pieces.append(
                        f"\n\n===== SOURCE: {url} =====\n\n"
                        f"{text}"
                    )

                    print(
                        f"      {len(text):,} characters"
                    )

            except Exception as exc:
                print(
                    f"      FAILED: {exc}"
                )

    for path in sorted(
        TEXT_DIR.glob("*.txt")
    ):
        print(f"[TXT] {path.name}")

        text = clean_text(
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

        if text:
            pieces.append(
                f"\n\n===== FILE: {path.name} =====\n\n"
                f"{text}"
            )

    corpus = "\n".join(pieces).strip()

    OUTPUT.write_text(
        corpus,
        encoding="utf-8",
    )

    print()
    print(f"Corpus: {OUTPUT}")
    print(f"Characters: {len(corpus):,}")


if __name__ == "__main__":
    main()
