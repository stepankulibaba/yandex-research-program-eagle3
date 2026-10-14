"""Isolate the pinned author metadata tail for CPU replay only."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch


def build_reference_metadata():
    source=Path(__file__).resolve().parents[2]/'EAGLE/eagle/model/cnets.py'
    tree=ast.parse(source.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Model')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='topK_genrate')
    idx=next(i for i,x in enumerate(fn.body) if isinstance(x,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='mask_index_list' for t in x.targets))
    tail=[x for x in fn.body[idx:] if not isinstance(x,ast.Delete)]
    replay=ast.parse('def replay(draft_tokens, mask_index):\n    pass').body[0]
    replay.body=ast.parse('total_tokens=mask_index.numel()\nhidden_states=_hidden\nlogits_processor=None').body+tail
    module=ast.Module(body=[replay],type_ignores=[]); ast.fix_missing_locations(module)
    scope={'torch':torch,'_hidden':SimpleNamespace(device=torch.device('cpu'))}
    exec(compile(module,'<author_cpu_replay>','exec'),scope)
    return scope['replay']
