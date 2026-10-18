"""Research training engine: pretraining, resume, and continual replay updates."""
from __future__ import annotations
import argparse, hashlib, math, os, random, tempfile
from pathlib import Path
import numpy as np
import torch
from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer
from training import config
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/processed'; CKPT=ROOT/config.checkpoint_path; BEST=ROOT/config.best_checkpoint_path

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load_bin(name):
 p=DATA/name
 if not p.exists(): raise FileNotFoundError(p)
 return np.memmap(p,dtype=np.uint16,mode='r')

def batch(data, block, bs, device, generator=None):
 if len(data)<=block+1: raise ValueError(f'Dataset {len(data)} tokens is too small for block {block}.')
 starts=torch.randint(0,len(data)-block-1,(bs,),generator=generator).tolist()
 x=torch.stack([torch.from_numpy(np.asarray(data[i:i+block],dtype=np.int64)) for i in starts])
 y=torch.stack([torch.from_numpy(np.asarray(data[i+1:i+block+1],dtype=np.int64)) for i in starts])
 return x.to(device),y.to(device)

def amp(device): return torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda')
@torch.no_grad()
def evaluate(model,data,cfg,device):
 # Reuse the same validation windows at every checkpoint. This makes loss
 # comparisons meaningful instead of adding fresh sampling noise each time.
 model.eval(); vals=[]; generator=torch.Generator().manual_seed(config.seed+10_000)
 for _ in range(config.eval_iters):
  x,y=batch(data,cfg['block_size'],config.batch_size,device,generator)
  with amp(device): _,loss=model(x,y)
  vals.append(float(loss))
 model.train(); return sum(vals)/len(vals)

def atomic(payload,path):
 path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix='._ckpt_',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f: torch.save(payload,f); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path)
 except Exception:
  Path(tmp).unlink(missing_ok=True); raise

def lr(step,total,peak):
 if step<config.warmup_steps: return peak*max(1,step+1)/config.warmup_steps
 p=min(1.0,(step-config.warmup_steps)/max(1,total-config.warmup_steps)); c=.5*(1+math.cos(math.pi*p)); return peak*(config.min_lr_ratio+(1-config.min_lr_ratio)*c)

def payload(model,opt,step,best,cfg,tokhash):
 state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
 if torch.cuda.is_available(): state['cuda']=torch.cuda.get_rng_state_all()
 return {'model_state':model.state_dict(),'optimizer_state':opt.state_dict(),'step':step,'best_val':best,'config':cfg,'tokenizer_sha256':tokhash,'rng_state':state,'version':3}

def restore_rng(state):
 if not state: return
 random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
 if torch.cuda.is_available() and 'cuda' in state: torch.cuda.set_rng_state_all(state['cuda'])

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['new','resume','update'],default='new'); ap.add_argument('--steps',type=int); args=ap.parse_args()
 torch.manual_seed(config.seed); np.random.seed(config.seed)
 device='cuda' if torch.cuda.is_available() else 'cpu'; print('Device:',device)
 if device=='cuda':
  print('GPU:',torch.cuda.get_device_name(0)); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB'); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
 if not (DATA/'train.bin').exists() or not (DATA/'val.bin').exists() or not (DATA/'vocab.json').exists(): raise FileNotFoundError('Prepare data first.')
 tok=BPETokenizer.load(DATA/'vocab.json'); tokhash=sha(DATA/'vocab.json'); val=load_bin('val.bin')
 cfg=dict(config.CONFIGS[config.MODEL_SIZE]); cfg['vocab_size']=tok.vocab_size
 if args.mode=='new':
  train=load_bin('train.bin'); model=ModernGPT(**cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,betas=config.betas,weight_decay=config.weight_decay); start=0; best=float('inf'); peak=config.learning_rate
 elif args.mode=='resume':
  train=load_bin('train.bin'); ck=torch.load(CKPT,map_location=device,weights_only=False)
  if ck.get('tokenizer_sha256') != tokhash: raise ValueError('Checkpoint tokenizer hash does not match data/processed/vocab.json. Restore the matching tokenizer or start a new run.')
  cfg=dict(ck['config']); model=ModernGPT(**cfg).to(device); model.load_state_dict(ck['model_state']); opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,betas=config.betas,weight_decay=config.weight_decay); opt.load_state_dict(ck['optimizer_state']); start=int(ck['step'])+1; best=float(ck.get('best_val',float('inf'))); peak=config.learning_rate; restore_rng(ck.get('rng_state')); print('Resuming:',start)
 else:
  ck=torch.load(CKPT,map_location=device,weights_only=False)
  if ck.get('tokenizer_sha256') != tokhash: raise ValueError('Checkpoint tokenizer hash does not match data/processed/vocab.json. Continual updates must reuse the original tokenizer.')
  cfg=dict(ck['config']); model=ModernGPT(**cfg).to(device); model.load_state_dict(ck['model_state']); new=load_bin('new.bin'); replay=load_bin('replay.bin'); opt=torch.optim.AdamW(model.parameters(),lr=config.update_learning_rate,betas=config.betas,weight_decay=config.weight_decay); start=0; best=float('inf'); peak=config.update_learning_rate; train=None; print(f'Continual update: new={len(new):,} replay={len(replay):,} tokens; replay ratio={config.replay_ratio}')
 model.gradient_checkpointing=config.gradient_checkpointing; print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}"); print('Context:',cfg['block_size'],'| effective tokens/step:',config.batch_size*config.gradient_accumulation_steps*cfg['block_size'])
 steps=args.steps or config.max_steps; scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda'); opt.zero_grad(set_to_none=True)
 for step in range(start,start+steps):
  current_lr=lr(step-start,steps,peak)
  for g in opt.param_groups:g['lr']=current_lr
  for micro in range(config.gradient_accumulation_steps):
   if args.mode=='update':
    # independent batches approximate token-level replay ratio.
    use_replay=(torch.rand(())<config.replay_ratio).item(); source=replay if use_replay else new
   else: source=train
   x,y=batch(source,cfg['block_size'],config.batch_size,device)
   with amp(device): _,loss=model(x,y); loss=loss/config.gradient_accumulation_steps
   scaler.scale(loss).backward()
  scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
  if (step+1)%config.eval_interval==0 or step==start+steps-1:
   v=evaluate(model,val,cfg,device); print(f'step {step:6d} | val {v:.4f} | lr {current_lr:.2e}')
   if v<best: best=v; atomic(payload(model,opt,step,best,cfg,tokhash),BEST); print('best checkpoint saved')
   atomic(payload(model,opt,step,best,cfg,tokhash),CKPT)
 final=start+steps-1; atomic(payload(model,opt,final,best,cfg,tokhash),CKPT); print('Training complete:',CKPT)

if __name__=='__main__': main()
