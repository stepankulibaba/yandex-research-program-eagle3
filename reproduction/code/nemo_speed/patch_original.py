"""Task 4: a copy of the authors' trainer (traineagle3) prepared for timing, in one of two modes.

    python patch_original.py <EAGLE repo> <warm-up optimizer steps> <work dir> [matched|asis]

Copies eagle/traineagle3 into <work dir> and edits it. Every edit is an exact text replacement that must match
once in the pinned code, so a different EAGLE version fails loudly.

Both modes:
    bug fix      the pinned cnets.py reads `self.train_config.gradient_checkpointing` (and `.max_len`), but main.py
                 passes a dict: the published trainer crashes with AttributeError. Read the dict keys instead.
    data         build_dataset_rank -> our shared corpus (training_common.original_dataset)
    vocabulary   model.scandata(...) -> our shared 32K mapping (training_common.mapping)
    epochs       40 -> 1; after the epoch write SPEED_OUTPUT (SpeedWindow) and exit
    optimizer    torch's AdamW inside DeepSpeed (its fused Adam needs a CUDA toolkit to build)
    wandb        replaced by a stub

mode=asis    the authors' settings otherwise untouched: fp16 with loss scaling, gradient checkpointing,
             padding to the longest example of the batch (micro-batch 1: no padding), their warm-up/decay lr
             schedule, shuffled loader with 4 workers, model.zero_grad() every micro-batch.
             -> "how fast is the authors' code as published"
mode=matched the same conditions as the NeMo runs: bf16, no checkpointing, every row padded to 2048, constant
             lr 5e-5, fixed order, no workers, no per-micro-batch zero_grad (DeepSpeed clears gradients at
             optimizer steps).  -> "how fast is each implementation on identical work"
"""
import ast
import json
from pathlib import Path
import shutil
import sys

PACKAGE = Path(__file__).resolve().parent.parent / 'inference_speed'
sys.path.insert(0, str(PACKAGE))
from common import atomic_json, atomic_text   # noqa: E402

MODES = ('matched', 'asis')


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Pinned EAGLE source changed: {old[:80]}')
    return source.replace(old, new, 1)


SHARED_VOCABULARY = """from training_common import SpeedWindow, mapping
from common import atomic_json, digest, read_json
_selected, _mask = mapping(os.environ['SHARED_MAPPING'], model.vocab_size)
model.register_buffer('d2t', _selected - torch.arange(len(_selected)))
model.register_buffer('t2d', _mask)
model.l1smooth = nn.SmoothL1Loss(reduction='none')"""

SPEED_WINDOW = """    _speed = SpeedWindow({warmup}, len(train_loader) // 2, 'original', {{
        'dtype': {dtype!r}, 'target_attention': 'eager', 'draft_attention': 'eager',
        'seq_length': {seq_length!r}, 'micro_batch': 1, 'accumulation': 2, 'packing': 0,
        'gradient_checkpointing': {checkpointing!r}, 'lr_schedule': {schedule!r},
        'ttt_steps': model.length, 'mapping_sha256': digest(read_json(os.environ['SHARED_MAPPING'])),
        'lr': 5e-5, 'compile_applied': False, 'fp8_modules': []}}, protocol={protocol!r})
"""
LOOP = '    for batch_idx, data in enumerate(tqdm(train_loader)):\n'
COUNT_BATCH = '        _speed.before_batch(batch_idx // 2, data, torch)\n'

CHECKED_BACKWARD = """        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError('Nonfinite training loss')
        model_engine.backward(loss)"""

TIMED_STEP = """        model_engine.step()
        if (batch_idx + 1) % 2 == 0:
            _speed.after_step((batch_idx + 1) // 2, torch)"""

WRITE_AND_EXIT = """    atomic_json(os.environ['SPEED_OUTPUT'], _speed.result(torch))
    import sys as _sys
    _sys.exit(0)
"""


def adapt_main(source, warmup, mode='matched'):
    """The edited text of traineagle3/main.py."""
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    matched = mode == 'matched'
    # Data: the whole build_dataset_rank function is replaced by a call to our loader.
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_dataset_rank')
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = ['def build_dataset_rank(tokenizer, datapath):\n',
                                              '    from training_common import original_dataset\n',
                                              '    return original_dataset(datapath)\n']
    s = ''.join(lines)
    s = replace_once(s, '"num_epochs": 40,', '"num_epochs": 1,')
    s = replace_once(s, 'model.scandata(args.trainpath, args.basepath)', SHARED_VOCABULARY)
    if matched:
        s = replace_once(s, '"gradient_checkpoint": True', '"gradient_checkpoint": False')
        s = replace_once(s, "max_length = max(item['input_ids'].shape[1] for item in features)",
                         'max_length = 2048')
        s = s.replace('num_workers=4', 'num_workers=0')
        s = replace_once(s, 'shuffle=True)', 'shuffle=False)')

    window = SPEED_WINDOW.format(
        warmup=warmup, dtype='bfloat16' if matched else 'float16',
        seq_length=2048 if matched else 'longest in batch', checkpointing=not matched,
        schedule='constant' if matched else 'WarmupDecayLR (authors)',
        protocol='matched-bf16-fixed2048-v1' if matched else 'author-as-is-fp16-v1')
    old_loop_start = LOOP + '\n        model.zero_grad()'
    if matched:   # DeepSpeed clears gradients at optimizer boundaries; no zero_grad per micro-batch
        s = replace_once(s, old_loop_start, window + LOOP + COUNT_BATCH)
    else:
        s = replace_once(s, old_loop_start, window + LOOP + COUNT_BATCH + '        model.zero_grad()')
    s = replace_once(s, '        model_engine.backward(loss)', CHECKED_BACKWARD)
    s = replace_once(s, '        model_engine.step()', TIMED_STEP)
    end_of_epoch = '    for i in range(len(epoch_acces)):\n'      # first line after the training loop
    if end_of_epoch not in s:
        raise ValueError('Missing epoch boundary')
    s = s.replace(end_of_epoch, WRITE_AND_EXIT + end_of_epoch, 1)
    ast.parse(s)
    return s


def adapt_cnets(source, mode='matched'):
    """The edited text of traineagle3/cnets.py: the dict-access bug fix, plus bf16 in matched mode."""
    s = replace_once(source, 'self.train_config.gradient_checkpointing', 'self.train_config["gradient_checkpoint"]')
    s = replace_once(s, 'self.train_config.max_len', 'self.train_config["max_len"]')
    if mode == 'matched':
        s = replace_once(s, 'torch_dtype=torch.float16', 'torch_dtype=torch.bfloat16')
    ast.parse(s)
    return s


def main(repo, warmup='20', destination='work/original', mode='matched'):
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    src = Path(repo) / 'eagle/traineagle3'
    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)      # reused between attempts; results live elsewhere
    for p in src.iterdir():
        if p.is_file() and p.suffix in ('.py', '.json'):
            shutil.copy2(p, dst / p.name)
    atomic_text(dst / 'wandb.py', 'def login(*a, **k): pass\ndef init(*a, **k): pass\ndef log(*a, **k): pass\n')
    atomic_text(dst / 'main.py', adapt_main((src / 'main.py').read_text(encoding='utf-8'), int(warmup), mode))
    atomic_text(dst / 'cnets.py', adapt_cnets((src / 'cnets.py').read_text(encoding='utf-8'), mode))

    config = json.loads((src / 'ds_config.json').read_text(encoding='utf-8'))
    config['optimizer']['params']['torch_adam'] = True
    if mode == 'matched':
        config.pop('scheduler', None)
        config['fp16'] = {'enabled': False}
        config['bf16'] = {'enabled': True}
        config['optimizer']['params']['lr'] = 5e-5
    atomic_json(dst / 'ds_config.json', config)
    if mode == 'matched':
        draft = json.loads((src / 'config.json').read_text(encoding='utf-8'))
        draft['torch_dtype'] = 'bfloat16'
        atomic_json(dst / 'config.json', draft)
    print(f'Built the authors\' trainer in {mode} mode:', dst)


if __name__ == '__main__':
    main(*sys.argv[1:])
