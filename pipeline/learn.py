"""Daily learning flow for a focused topic."""
from __future__ import annotations
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]; TOPICS=ROOT/'data/topics'; CK=ROOT/'checkpoints/gpt.pt'
def run(*args):
 print('>>>',' '.join(args)); subprocess.run(args,cwd=ROOT,check=True)
def slug(s):
 import re; return re.sub(r'[^a-z0-9]+','_',s.lower()).strip('_') or 'general'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--topic',required=True); ap.add_argument('--steps',type=int,default=1500); args=ap.parse_args()
 d=TOPICS/slug(args.topic); d.mkdir(parents=True,exist_ok=True)
 print(f'Put new permitted .txt material into: {d}')
 run(sys.executable,'-m','dataset.collect'); run(sys.executable,'-m','dataset.chunk')
 run(sys.executable,'-m','dataset.prepare','--reuse-tokenizer')
 mode='new' if not CK.exists() else 'update'
 run(sys.executable,'-m','training.train','--mode',mode,'--steps',str(args.steps))
 run(sys.executable,'experiments/benchmark.py')
 print('\nDAILY LEARNING COMPLETE')
if __name__=='__main__':main()
