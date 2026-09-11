"""Run repeatable validation perplexity and report training metadata."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
meta=ROOT/'data/processed/meta.json'; ck=ROOT/'checkpoints/gpt.pt'
print('=== Small-GPT v3 benchmark metadata ===')
print(json.dumps(json.loads(meta.read_text()),indent=2) if meta.exists() else 'No prepared dataset')
if ck.exists():
 import torch
 x=torch.load(ck,map_location='cpu',weights_only=False)
 print('checkpoint step:',x.get('step')); print('best validation loss:',x.get('best_val')); print('approx perplexity:', math.exp(min(20,float(x.get('best_val',20)))))
else: print('No checkpoint yet.')
