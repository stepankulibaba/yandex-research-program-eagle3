"""Surgical AST replacement of metadata tail; candidate selection stays upstream."""
import ast
import copy
import inspect
import textwrap
import types
import torch
from metadata import finish_tensor_metadata


def make_topk(ea_layer, upstream, variant="upstream", observer=None, ranges=False):
    if variant not in ("upstream", "leaf_cpu", "tensor_gpu_leaf", "tensor_cpu"):
        raise ValueError(variant)
    source = textwrap.dedent(inspect.getsource(upstream))
    tree = ast.parse(source)
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef))
    boundary = [i for i, x in enumerate(fn.body)
                if isinstance(x, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'mask_index_list'
                    for t in x.targets)]
    if len(boundary) != 1:
        raise RuntimeError('Unsupported upstream metadata boundary')
    tail = copy.deepcopy(fn.body[boundary[0]:])
    if variant == 'leaf_cpu':
        assignments=[x for x in tail if isinstance(x,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id=='noleaf_index' for t in x.targets)]
        if len(assignments)!=1:
            raise RuntimeError('Unsupported upstream leaf boundary')
        assignments[0].value=ast.parse('sorted(set(mask_index_list))',mode='eval').body
    if variant in ('tensor_cpu', 'tensor_gpu_leaf'):
        tail = ast.parse('''
result = finish_tensor_metadata(draft_tokens, mask_index, hidden_states.device, logits_processor, gpu_unique=_gpu_unique)
return result
''').body
    if observer is not None:
        tail.insert(0, ast.parse('_parents_for_capture = mask_index').body[0])
        ret = tail[-1]
        if not isinstance(ret, ast.Return):
            raise RuntimeError('Upstream tail must end in return')
        tail[-1:] = [ast.Assign(targets=[ast.Name(id='_out', ctx=ast.Store())], value=ret.value)] + ast.parse('''
_observer(_parents_for_capture, _out)
return _out
''').body
    if ranges:
        # Observer capture is diagnostic only, never active in clean timing.
        ret = tail.pop()
        context = ast.With(items=[ast.withitem(context_expr=ast.parse(
            'torch.profiler.record_function("tree_metadata")', mode='eval').body)], body=tail)
        tail = [context, ret]
    fn.body = fn.body[:boundary[0]] + tail
    ast.fix_missing_locations(tree)
    scope = dict(inspect.unwrap(upstream.__func__).__globals__)
    scope.update(finish_tensor_metadata=finish_tensor_metadata, _observer=observer,
                 _gpu_unique=(variant=='tensor_gpu_leaf'))
    exec(compile(tree, '<eagle_metadata_patch>', 'exec'), scope)
    return types.MethodType(scope[fn.name], ea_layer)
