"""Generate a synthetic educational text with an Ollama model.

Generated files are written to data/text/ and deliberately kept out of the
primary data/ingest/ corpus until a person has reviewed them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import ollama

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-oss:120b-cloud"
OUTPUT_DIR = ROOT / "data" / "text"/ "ollama"
NUM_PREDICT = 20_000


def safe_filename(topic: str) -> str:
    """Convert a topic into a portable filename stem."""
    filename = re.sub(r"[^a-z0-9]+", "_", topic.lower().strip()).strip("_")
    return filename or "generated_text"


def generate_text(topic: str, model: str, num_predict: int) -> str:
    prompt = f"""
Write a very large, detailed, high-quality educational text about:

"{topic}"

Requirements:

- Explain the topic from basic concepts to advanced concepts.
- Assume the reader is a beginner.
- Use clear and natural language.
- Include definitions and explanations.
- Include examples where useful.
- Explain important terminology.
- Explain how and why things work.
- Include practical applications.
- Include advantages and disadvantages when relevant.
- Include common mistakes and misconceptions.
- Include important subtopics related to the main topic.
- Organize the content with clear headings and subheadings.
- Make the document coherent rather than repeating the same information.
- Do not mention that you are an AI.
- Do not discuss this prompt.
- Produce ONLY the actual educational content.

Generate as much useful content as possible.
"""

    print(f"Generating synthetic text using {model}...")
    print(f"Topic: {topic}\n")
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"num_predict": num_predict},
    )
    text = response["response"].strip()
    if not text:
        raise RuntimeError("The model returned an empty response.")
    return text


def available_path(topic: str) -> Path:
    """Choose a fresh output path so a previous generation is preserved."""
    stem = safe_filename(topic)
    candidate = OUTPUT_DIR / f"{stem}.txt"
    version = 2
    while candidate.exists() or candidate.with_suffix(".json").exists():
        candidate = OUTPUT_DIR / f"{stem}_{version}.txt"
        version += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", nargs="+", help="Subject for the educational text")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model (default: {MODEL})")
    parser.add_argument("--num-predict", type=int, default=NUM_PREDICT,
                        help=f"Maximum output tokens (default: {NUM_PREDICT})")
    args = parser.parse_args()
    if args.num_predict < 1:
        parser.error("--num-predict must be positive")
    topic = " ".join(args.topic).strip()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        text = generate_text(topic, args.model, args.num_predict)
        output_path = available_path(topic)
        output_path.write_text(text + "\n", encoding="utf-8")
        metadata = {
            "synthetic": True,
            "topic": topic,
            "model": args.model,
            "num_predict_limit": args.num_predict,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "text_file": output_path.name,
            "review_status": "unreviewed",
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        print("Generation complete.")
        print(f"Characters: {len(text):,}")
        print(f"Words:      {len(text.split()):,}")
        print(f"Saved text: {output_path.relative_to(ROOT)}")
        print(f"Metadata:   {output_path.with_suffix('.json').relative_to(ROOT)}")
        print("This is synthetic text. Review it and its source provenance before training on it.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
