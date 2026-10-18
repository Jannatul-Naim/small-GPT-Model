"""Prepare uint16 token binaries without loading the whole dataset into RAM."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from array import array
from tokenizer.bpe import BPETokenizer

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/sources.jsonl'; NEW=ROOT/'data/raw/new_sources.json'
OUT=ROOT/'data/processed'; VOCAB=OUT/'vocab.json'; META=OUT/'meta.json'
VOCAB_SIZE=16384; VAL_MOD=20

def write_bin(path, token_lists):
    with path.open('wb') as f:
        for ids in token_lists:
            a=array('H', ids); a.tofile(f)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reuse-tokenizer',action='store_true'); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not RAW.exists(): raise FileNotFoundError('Run python -m dataset.collect')
    rows=[json.loads(x) for x in RAW.read_text(encoding='utf-8',errors='ignore').splitlines() if x.strip()]
    if not rows: raise ValueError('No sources collected.')
    if args.reuse_tokenizer:
        if not VOCAB.exists(): raise FileNotFoundError('No tokenizer. Run prepare without --reuse-tokenizer first.')
        tok=BPETokenizer.load(VOCAB); print('Reusing tokenizer:', tok.vocab_size)
    else:
        corpus='\n\n'.join(r['text'] for r in rows)
        if len(corpus)<100_000: print('WARNING: corpus is below 100k characters; add more material for meaningful pretraining.')
        tok=BPETokenizer(); tok.train(corpus,vocab_size=VOCAB_SIZE,min_frequency=2,verbose=True); tok.save(VOCAB)
    if tok.vocab_size>VOCAB_SIZE: raise ValueError(f'Tokenizer overflow: {tok.vocab_size} > {VOCAB_SIZE}')

    new=set(json.loads(NEW.read_text())) if NEW.exists() else set()
    train=[]; val=[]; new_tokens=[]; old_tokens=[]
    total=0
    for r in rows:
        ids=tok.encode(r['text'])
        if len(ids)<769: continue
        # deterministic source-level validation split: prevents text-fragment leakage.
        h=int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16)
        target=val if h%VAL_MOD==0 else train
        target.append(ids)
        total+=len(ids)
        if r['id'] in new: new_tokens.append(ids)
        else: old_tokens.append(ids)
    if not train or not val: raise ValueError('Need more distinct sources to create train/validation splits.')
    write_bin(OUT/'train.bin',train); write_bin(OUT/'val.bin',val)
    write_bin(OUT/'new.bin',new_tokens if new_tokens else old_tokens[:1])
    write_bin(OUT/'replay.bin',old_tokens if old_tokens else train[:1])
    def sha(p):
        h=hashlib.sha256();
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''): h.update(b)
        return h.hexdigest()
    meta={'vocab_size':tok.vocab_size,'sources':len(rows),'new_sources':len(new),'train_tokens':sum(len(x) for x in train),'val_tokens':sum(len(x) for x in val),'new_tokens':sum(len(x) for x in new_tokens),'replay_tokens':sum(len(x) for x in old_tokens),'tokenizer_sha256':sha(VOCAB)}
    META.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
