"""Rebuild topic data with the fixed tokenizer and run a replay update."""
from __future__ import annotations
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]; TOPICS=ROOT/'data/topics'; CK=ROOT/'checkpoints/gpt.pt'; VOCAB=ROOT/'data/processed/vocab.json'
def run(*args):
 print('>>>',' '.join(args)); subprocess.run(args,cwd=ROOT,check=True)
def slug(s):
 import re; return re.sub(r'[^a-z0-9]+','_',s.lower()).strip('_') or 'general'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--topic',required=True); ap.add_argument('--steps',type=int,default=1500); args=ap.parse_args()
 if not CK.exists() or not VOCAB.exists():
  raise SystemExit('Create a base model first: build data with dataset.engine, then train with --mode new.')
 d=TOPICS/slug(args.topic); d.mkdir(parents=True,exist_ok=True)
 print(f'Put new permitted .txt material into: {d}')
 run(sys.executable,'-m','dataset.engine','--input',str(TOPICS),'--no-web','--reuse-tokenizer')
 new_ids=json.loads((ROOT/'data/raw/new_sources.json').read_text(encoding='utf-8'))
 if not new_ids:
  raise SystemExit('No new source versions were accepted; add new material before running an update.')
 run(sys.executable,'-m','training.train','--mode','update','--steps',str(args.steps))
 report=ROOT/'reports'/f'{slug(args.topic)}-update-validation.json'
 run(sys.executable,'evaluation.py','--checkpoint','checkpoints/gpt-best.pt','--output',str(report.relative_to(ROOT)))
 print('\nDAILY LEARNING COMPLETE')
if __name__=='__main__':main()
