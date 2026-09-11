from pathlib import Path
import json,torch
from fastapi import FastAPI
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer
from rag import Retriever
ROOT=Path(__file__).resolve().parents[1]; CKPT=ROOT/'checkpoints/gpt.pt'; VOCAB=ROOT/'data/processed/vocab.json'; WEB=ROOT/'web'; device='cuda' if torch.cuda.is_available() else 'cpu'
if not CKPT.exists(): raise FileNotFoundError(CKPT)
ck=torch.load(CKPT,map_location=device,weights_only=False); tok=BPETokenizer.load(VOCAB); model=ModernGPT(**ck['config']).to(device); model.load_state_dict(ck['model_state']); model.eval(); retriever=Retriever()
app=FastAPI(title='Small-GPT Research Assistant',version='3.0'); app.mount('/static',StaticFiles(directory=WEB),name='static')
class Req(BaseModel): prompt:str; use_retrieval:bool=True; max_new_tokens:int=180; temperature:float=.75; top_k:int=40; top_p:float=.92
@app.get('/')
def home():return FileResponse(WEB/'index.html')
@app.get('/health')
def health(): return {'status':'ok','device':device,'parameters':sum(p.numel() for p in model.parameters()),'retrieval_chunks':len(retriever.chunks),'checkpoint_step':ck.get('step')}
@app.post('/search')
def search(r:Req): retriever.refresh(); return {'results':retriever.search(r.prompt,5)}
@app.post('/stream')
def stream(r:Req):
 retriever.refresh(); hits=retriever.search(r.prompt,3) if r.use_retrieval else []; sources=[h['source'] for h in hits]
 context='\n\n'.join(f"[SOURCE {i+1}: {h['source']}]\n{h['text']}" for i,h in enumerate(hits))
 prompt=("Use the retrieved source material when relevant. If it does not contain the answer, say you are unsure.\n\n"+context+"\n\nQUESTION:\n"+r.prompt+"\nANSWER:\n") if context else r.prompt
 ids=tok.encode(prompt); idx=torch.tensor([ids[-ck['config']['block_size']:]],dtype=torch.long,device=device)
 def events():
  cur=idx
  for _ in range(min(max(1,r.max_new_tokens),512)):
   with torch.no_grad():
    with torch.autocast('cuda',dtype=torch.float16,enabled=device=='cuda'): logits,_=model(cur[:,-ck['config']['block_size']:])
   logits=logits[:,-1,:]/max(r.temperature,1e-5)
   if r.top_k:
    v,_=torch.topk(logits,min(r.top_k,logits.size(-1))); logits[logits<v[:,-1:]]=float('-inf')
   if 0<r.top_p<1:
    sl,si=torch.sort(logits,descending=True); sp=torch.softmax(sl,dim=-1); cum=torch.cumsum(sp,dim=-1); rm=cum>r.top_p; rm[:,1:]=rm[:,:-1].clone(); rm[:,0]=False; sl[rm]=float('-inf'); logits=torch.full_like(logits,float('-inf')); logits.scatter_(1,si,sl)
   nxt=torch.multinomial(torch.softmax(logits,dim=-1),1); cur=torch.cat([cur,nxt],1); token=tok.decode([int(nxt.item())]); yield 'data: '+json.dumps({'token':token,'sources':sources})+'\n\n'
   if int(nxt.item())==tok.EOS: break
  yield 'data: [DONE]\n\n'
 return StreamingResponse(events(),media_type='text/event-stream')
