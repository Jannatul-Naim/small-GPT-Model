"""
Simple repeatable domain evaluation.

Usage:
  python evaluation.py --topic "linear algebra"

It asks the same questions before/after a training update so you can track
whether the model actually improved instead of trusting training loss alone.
"""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "data/eval_questions.txt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="domain")
    ap.add_argument("--questions", default=str(DEFAULT))
    args = ap.parse_args()
    qfile = Path(args.questions)
    if not qfile.exists():
        qfile.parent.mkdir(parents=True, exist_ok=True)
        qfile.write_text(
            "Define the main concepts of this topic.\n"
            "Explain one important mechanism step by step.\n"
            "Give a practical example.\n"
            "What are common mistakes?\n",
            encoding="utf-8"
        )
    print(f"Evaluation set for: {args.topic}")
    for q in qfile.read_text(encoding="utf-8").splitlines():
        if q.strip():
            print("\nQUESTION:", q)
            subprocess.run(
                [sys.executable, "-m", "generation.generate",
                 "--prompt", q, "--max-new-tokens", "120"],
                cwd=ROOT
            )

if __name__ == "__main__":
    main()
