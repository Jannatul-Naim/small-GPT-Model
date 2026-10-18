"""Local lexical retrieval for current/source-grounded answers."""
from pathlib import Path
import json, math, re
from collections import Counter,defaultdict



ROOT=Path(__file__).resolve().parent; CHUNKS=ROOT/'data/raw/chunks.jsonl'; WORD=re.compile(r"[A-Za-z0-9_+#.-]+")


def toks(s): return [x.lower() for x in WORD.findall(s)]

class Retriever:
 
 def __init__(self): 
  self.refresh()
 def refresh(self):
  self.chunks=[]; self.post=defaultdict(list); self.df=Counter()
  if not CHUNKS.exists(): return
  for line in CHUNKS.read_text(encoding='utf-8',errors='ignore').splitlines():
   if not line.strip(): continue
   c=json.loads(line); self.chunks.append(c)
  for i,c in enumerate(self.chunks):
   tf=Counter(toks(c['text'])); c['_tf']=tf
   for t in tf: self.df[t]+=1; self.post[t].append(i)
  self.n=len(self.chunks)
 def search(self,q,top_n=5):
  if not self.chunks:return []
  qt=toks(q); scores=Counter()
  for term in qt:
   idf=math.log(1+(self.n-self.df.get(term,0)+.5)/(self.df.get(term,0)+.5))
   for i in self.post.get(term,[]):
    tf=self.chunks[i]['_tf'][term]; dl=sum(self.chunks[i]['_tf'].values()); avg=max(1,sum(sum(c['_tf'].values()) for c in self.chunks)/self.n)
    scores[i]+=idf*(tf*2.2)/(tf+1.2*(.25+.75*dl/avg))
  out=[]
  for i,_ in scores.most_common(top_n):
   c=dict(self.chunks[i]); c.pop('_tf',None); out.append(c)
  return out
