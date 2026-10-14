"""Windows loading workaround: read-only safetensors mappings, unchanged values.

Used only during setup. No vendor or installed library files are edited.
Each tensor is cloned before it leaves the loader, so downstream model setup
receives writable storage. Only the file mapping itself remains read-only.
"""
import contextlib
import importlib
import json
import math
import mmap
import sys
import warnings
import torch

DTYPES={'F64':torch.float64,'F32':torch.float32,'F16':torch.float16,'BF16':torch.bfloat16,
        'I64':torch.int64,'I32':torch.int32,'I16':torch.int16,'I8':torch.int8,
        'U8':torch.uint8,'BOOL':torch.bool}


class ReadOnlySafeOpen:
    def __init__(self, filename, framework='pt', device='cpu'):
        if framework not in ('pt','pytorch'):
            raise ValueError('Only PyTorch supported')
        self.device=torch.device('cuda',device) if isinstance(device,int) else torch.device(device)
        with open(filename,'rb') as f:
            size=int.from_bytes(f.read(8),'little')
            self.header=json.loads(f.read(size))
            self.buffer=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        self.offset=8+size
    def __enter__(self): return self
    def __exit__(self,*a):
        # torch.frombuffer owns a reference to the buffer until its tensors die.
        return False
    def keys(self): return sorted(k for k in self.header if k!='__metadata__')
    def metadata(self): return self.header.get('__metadata__')
    def get_tensor(self,key):
        h=self.header[key]; dtype=DTYPES[h['dtype']]; count=math.prod(h['shape'])
        if count==0: return torch.empty(h['shape'],dtype=dtype,device=self.device)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='The given buffer is not writable')
            x=torch.frombuffer(self.buffer,dtype=dtype,count=count,offset=self.offset+h['data_offsets'][0])
        return x.reshape(h['shape']).clone().to(self.device)
    def get_slice(self,key): return ReadOnlySlice(self,key)


class ReadOnlySlice:
    def __init__(self,owner,key): self.owner,self.key=owner,key
    def get_shape(self): return self.owner.header[self.key]['shape']
    def get_dtype(self): return self.owner.header[self.key]['dtype']
    def __getitem__(self,index): return self.owner.get_tensor(self.key)[index]


@contextlib.contextmanager
def read_only_safetensors():
    # cnets imports safe_open locally in its constructor, so patch the module
    # export as well as Transformers' already-bound import during setup.
    modules=[importlib.import_module('transformers.modeling_utils'),
             importlib.import_module('safetensors')]
    cnets=importlib.import_module('eagle.model.cnets')
    if hasattr(cnets,'safe_open'): modules.append(cnets)
    originals=[(m,m.safe_open) for m in modules]
    try:
        for m,_ in originals: m.safe_open=ReadOnlySafeOpen
        yield
    finally:
        for m,fn in originals: m.safe_open=fn
