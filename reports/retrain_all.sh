#!/usr/bin/env bash
# Rebuild the story data, retrain TinyStories v2, v1 and stories-v3 with their original settings, and score them.
# Each step is skipped once done and resumed after a power cut, so rerun this same command.
# Run it from the "ai" conda env: bash reports/retrain_all.sh
# With --prepare-only it stops once the stories-v3 data is built, before training v3.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1

mkdir -p checkpoints
exec 9>checkpoints/.retrain.lock
flock -n 9 || { echo "Another retrain_all.sh is already running."; exit 0; }

download() {
  if [ ! -f data/tinystories/raw/TinyStoriesV2-GPT4-train.txt ] || [ ! -f data/tinystories/raw/TinyStoriesV2-GPT4-valid.txt ]; then
    hf download roneneldan/TinyStories TinyStoriesV2-GPT4-train.txt TinyStoriesV2-GPT4-valid.txt \
      --repo-type dataset --local-dir data/tinystories/raw
  fi
  if [ ! -f data/simplestories/raw/test-00000-of-00001.parquet ]; then
    hf download SimpleStories/SimpleStories --repo-type dataset --include "data/*.parquet" \
      --local-dir data/simplestories/dl
    mkdir -p data/simplestories/raw
    mv data/simplestories/dl/data/train-*.parquet data/simplestories/raw/
    mv data/simplestories/dl/data/test-*.parquet data/simplestories/raw/
  fi
}

build() {  # name train-text val-text vocab-size
  [ -f "data/$1/meta.json" ] && return
  python -m dataset.large_text --train "$2" --val "$3" --out "data/$1" --vocab-size "$4" \
    2>&1 | tee -a "reports/$1-retrain-build.log"
}

train() {  # run data-dir model-size steps
  local dir="checkpoints/$1"
  [ -f "$dir/DONE" ] && return
  if [ ! -f "$dir/TRAINED" ]; then
    if [ -f "$dir/gpt.pt" ]; then
      mode=(--mode resume)
    else
      mode=(--mode new --model-size "$3" --steps "$4")
    fi
    python -m training.train "${mode[@]}" --eval-interval 500 --data-dir "data/$2" --checkpoint-dir "$dir" \
      2>&1 | tee -a "reports/$1-retrain-train.log"
    touch "$dir/TRAINED"
  fi
  python evaluation.py --checkpoint "$dir/gpt-best.pt" --data-dir "data/$2" --output "reports/$1-retrain.json"
  touch "$dir/DONE"
}

report() {  # output-file command...
  local out=$1
  shift
  [ -f "$out" ] && return
  if "$@" > "$out.tmp" 2>&1; then mv "$out.tmp" "$out"; else echo "Report failed, see $out.tmp"; fi
}

download
build tinystories data/tinystories/raw/TinyStoriesV2-GPT4-train.txt data/tinystories/raw/TinyStoriesV2-GPT4-valid.txt 4096
train tinystories-v2 tinystories medium 45000
report reports/tinystories-v2-retrain-comparison.log python reports/eval_stories.py checkpoints/tinystories-v2 data/tinystories
train tinystories-v1 tinystories small 20000
report reports/tinystories-v1-retrain-comparison.log python reports/eval_stories.py checkpoints/tinystories-v1 data/tinystories
report reports/tinystories-v1-v2-retrain-sample.log python reports/compare_stories.py \
  checkpoints/tinystories-v1=data/tinystories checkpoints/tinystories-v2=data/tinystories
[ -f data/stories-v3/raw/val.txt ] || python -I reports/make_stories_text.py | tee -a reports/stories-v3-retrain-text.log
build stories-v3 data/stories-v3/raw/train.txt data/stories-v3/raw/val.txt 8192
[ "${1:-}" = --prepare-only ] && { echo "stories-v3 data is ready; stopping before v3 training."; exit 0; }
train stories-v3 stories-v3 large 93400
report reports/stories-v3-retrain-comparison.log python reports/eval_stories.py checkpoints/stories-v3 data/stories-v3
report reports/stories-v2-v3-retrain-sample.log python reports/compare_stories.py \
  checkpoints/tinystories-v2=data/tinystories checkpoints/stories-v3=data/stories-v3
report reports/stories-retrain-bits.log python reports/score_bits.py
echo "All retraining done."
