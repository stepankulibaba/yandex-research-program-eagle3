"""Build a speed-only adapter in a new work dir; never erase completed runs."""
import ast
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent.parent / 'inference_speed'
sys.path.insert(0, str(PACKAGE))
from runtime import atomic_json, atomic_text

ADAPTER_VERSION = 3


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Pinned EAGLE source changed: {old[:80]}')
    return source.replace(old, new, 1)


def adapt_main(source, warmup):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_dataset_rank')
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [
        'def build_dataset_rank(tokenizer, datapath):\n',
        '    from training_data import original_dataset\n',
        '    return original_dataset(datapath)\n']
    s = ''.join(lines)
    s = replace_once(s, '"num_epochs": 40,', '"num_epochs": 1,')
    s = replace_once(s, '"gradient_checkpoint": True', '"gradient_checkpoint": False')
    s = replace_once(s, "max_length = max(item['input_ids'].shape[1] for item in features)", 'max_length = 2048')
    s = s.replace('num_workers=4', 'num_workers=0')
    s = replace_once(s, 'shuffle=True)', 'shuffle=False)')
    s = replace_once(s, 'model.scandata(args.trainpath, args.basepath)', """from training_data import mapping
from runtime import BenchmarkWindow, atomic_json, digest, read_json
import math
_selected, _mask = mapping(os.environ['SHARED_MAPPING'], model.vocab_size)
model.register_buffer('d2t', _selected - torch.arange(len(_selected)))
model.register_buffer('t2d', _mask)
model.l1smooth = nn.SmoothL1Loss(reduction='none')""")
    old_loop = '    for batch_idx, data in enumerate(tqdm(train_loader)):\n\n        model.zero_grad()'
    new_loop = f"""    _speed = BenchmarkWindow({warmup}, len(train_loader) // 2, 'original', {{
        'dtype': 'bfloat16', 'target_attention': 'eager', 'draft_attention': 'eager',
        'seq_length': 2048, 'micro_batch': 1, 'accumulation': 2, 'packing': 0,
        'ttt_steps': model.length, 'mapping_sha256': digest(read_json(os.environ['SHARED_MAPPING'])),
        'lr': 5e-5, 'compile_applied': False, 'fp8_modules': []}})
    for batch_idx, data in enumerate(tqdm(train_loader)):
        _speed.before_batch(batch_idx // 2, data, torch)
        # DeepSpeed clears gradients at optimizer boundaries; do not zero each micro-batch."""
    s = replace_once(s, old_loop, new_loop)
    s = replace_once(s, '        model_engine.backward(loss)', """        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError('Nonfinite training loss')
        model_engine.backward(loss)""")
    s = replace_once(s, '        model_engine.step()', """        model_engine.step()
        if (batch_idx + 1) % 2 == 0:
            _speed.after_step((batch_idx + 1) // 2, torch)""")
    anchor = '    for i in range(len(epoch_acces)):\n'
    if s.count(anchor) < 1:
        raise ValueError('Missing epoch boundary')
    s = s.replace(anchor, """    atomic_json(os.environ['SPEED_OUTPUT'], _speed.result(torch))
    import sys as _sys
    _sys.exit(0)
""" + anchor, 1)
    ast.parse(s)
    return s


def main(repo, warmup='20', destination='work/original'):
    import shutil
    src = Path(repo) / 'eagle/traineagle3'
    dst = Path(destination)
    # Different attempts can reuse code, but never recursively delete artifacts.
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix in ('.py', '.json'):
            shutil.copy2(p, dst / p.name)
    atomic_text(dst / 'wandb.py', 'def login(*a, **k): pass\ndef init(*a, **k): pass\ndef log(*a, **k): pass\n')
    atomic_text(dst / 'main.py', adapt_main((src / 'main.py').read_text(encoding='utf-8'), int(warmup)))
    cnets = (src / 'cnets.py').read_text(encoding='utf-8')
    cnets = replace_once(cnets, 'torch_dtype=torch.float16', 'torch_dtype=torch.bfloat16')
    atomic_text(dst / 'cnets.py', cnets)
    config = json.loads((src / 'ds_config.json').read_text(encoding='utf-8'))
    config.pop('scheduler', None)
    config['fp16'] = {'enabled': False}
    config['bf16'] = {'enabled': True}
    config['optimizer']['params']['lr'] = 5e-5
    config['optimizer']['params']['torch_adam'] = True
    atomic_json(dst / 'ds_config.json', config)
    draft = json.loads((src / 'config.json').read_text(encoding='utf-8'))
    draft['torch_dtype'] = 'bfloat16'
    atomic_json(dst / 'config.json', draft)
    print('Built adapted BF16/fixed2048/constant-LR/no-checkpoint trainer:', dst)


if __name__ == '__main__':
    main(*sys.argv[1:])
