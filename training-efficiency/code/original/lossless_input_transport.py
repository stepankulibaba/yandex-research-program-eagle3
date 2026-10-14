"""Lossless row deduplication and gzip transport; no numeric/data transformations."""
import gzip,hashlib,json,shutil,tempfile
from pathlib import Path
import torch

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(4*2**20):h.update(b)
 return h.hexdigest()
def load_lossless_inputs(inp):
 inp=Path(inp);meta=json.loads((inp/'input_manifest.json').read_text());p=inp/'compact_teacher_inputs.pt.gz'
 if p.exists():
  assert sha(p)==meta['transport_gzip_sha256']
  with tempfile.TemporaryDirectory(prefix='eagle-lossless-decode-') as d:
   tmp=Path(d)/'compact.pt'
   with gzip.open(p,'rb') as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,4*2**20)
   assert sha(tmp)==meta['transport_compact_sha256'];compact=torch.load(tmp,weights_only=True,map_location='cpu')
 else:
  # Some dataset services expand archives at mount time; exact byte SHA still required.
  p=inp/'compact_teacher_inputs.pt';assert sha(p)==meta['transport_compact_sha256'];compact=torch.load(p,weights_only=True,map_location='cpu')
 assert compact['format']=='exact-row-dedup-v1';result={}
 for n,record in compact['cases'].items():
  p=compact['rows'].index_select(0,record['teacher_indices'].reshape(-1).long()).reshape(*record['teacher_shape']);cache=dict(record['other']);cache['target_probs']=p
  assert set(cache)==set(meta['logical_tensor_manifest'][n])
  for key,t in cache.items():
   expected=meta['logical_tensor_manifest'][n][key];assert list(t.shape)==expected['shape'] and str(t.dtype)==expected['dtype'] and tensor_hash(t)==expected['sha256'],(n,key)
  result[n]=cache
 return result
