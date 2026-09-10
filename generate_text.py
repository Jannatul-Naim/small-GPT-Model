import os
import re
import sys
import ollama


# ============================================================
# CONFIG
# ============================================================

MODEL = "gpt-oss:120b-cloud"
OUTPUT_DIR = "/data/text"

# Number of tokens requested from Ollama.
# Increase this if you want larger documents.
NUM_PREDICT = 20000


# ============================================================
# FUNCTIONS
# ============================================================

def safe_filename(topic):
    """Convert topic into a safe filename."""
    filename = topic.lower().strip()
    filename = re.sub(r"[^a-z0-9]+", "_", filename)
    filename = filename.strip("_")

    if not filename:
        filename = "generated_text"

    return filename + ".txt"


def generate_text(topic):
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

    print(f"Generating text using {MODEL}...")
    print(f"Topic: {topic}")
    print()

    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={
            "num_predict": NUM_PREDICT,
        }
    )

    return response["response"]


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python generate_text.py "your topic"')
        print()
        print("Example:")
        print('  python generate_text.py "Neural Networks"')
        sys.exit(1)

    topic = " ".join(sys.argv[1:]).strip()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = safe_filename(topic)
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        text = generate_text(topic)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print("Generation complete!")
        print()
        print(f"Characters: {len(text):,}")
        print(f"Words:      {len(text.split()):,}")
        print(f"Saved to:   {output_path}")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()