"""Stage 1: exact-objective chunked target supervision, not a training-speed claim.

Uses the pinned NeMo target-preparation functions as the dense reference.
No NeMo installation, target backbone, or checkpoint modification is needed.
"""
import argparse
import ast
import gc
import hashlib
import json
import math
import platform
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
NEMO_SHA = '3914f200a4c782d44b58ee7a01b4685e4158e19c'
QWEN_SHA = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
REF = ROOT / 'Automodel'


def load_function(relative, name):
    """Execute just a reviewed function, avoiding the full NeMo dependency stack."""
    hashes = json.loads((REF / 'source_hashes.json').read_text())
    source = (REF / relative).read_bytes()
    assert hashlib.sha256(source).hexdigest() == hashes[relative]
    tree = ast.parse(source)
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(REF / relative), 'exec'), namespace)
    return namespace[name]


shift = load_function('nemo_automodel/components/speculative/eagle/target.py', '_shift_left_with_zero')
distribution = load_function('nemo_automodel/components/speculative/eagle/core.py', '_compute_target_distribution')


@torch.no_grad()
def dense(hidden, weight, ids, membership, loss_mask, bias=None):
    logits = F.linear(hidden, weight, bias)
    shifted_logits = shift(logits)
    # generate_batch returns the shifted tensor; its original outputs/logits
    # die before trainer-side distribution preparation. Mirror that lifetime.
    del logits
    return distribution(shifted_logits, ids, membership, shift(loss_mask))


@torch.no_grad()
def chunked(hidden, weight, ids, membership, loss_mask, chunk, bias=None):
    batch, length, _ = hidden.shape
    probs = torch.empty(batch, length, ids.numel(), device=hidden.device, dtype=torch.float32)
    mask = torch.zeros(batch, length, 1, device=hidden.device, dtype=torch.bool)
    # NeMo shifts full logits and zero-fills the tail. A zero-logit row has
    # uniform probabilities even when the lm_head has a bias.
    probs[:, -1].fill_(1.0 / ids.numel())
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        # Compute ALL original positions, including unused position 0, just as
        # the dense baseline. Only the distribution for that position is omitted.
        logits = F.linear(hidden[:, start:end], weight, bias)
        first_kept = max(start, 1)
        if first_kept == end:
            del logits
            continue
        kept = logits[:, first_kept - start:]
        top_ids = kept.argmax(-1)
        selected = kept.index_select(-1, ids)
        probs[:, first_kept - 1:end - 1] = torch.softmax(selected.float(), -1)
        mask[:, first_kept - 1:end - 1, 0] = membership[top_ids] & loss_mask[:, first_kept:end].bool()
        # Releasing intermediates per chunk is essential; neither target nor
        # supervision carries an autograd graph, matching the frozen target.
        del logits, kept, top_ids, selected
    return probs, mask


def soft_ce(logits, probs, mask):
    loss = -(probs * F.log_softmax(logits.float(), -1)).sum(-1)
    valid = mask.squeeze(-1).float()
    return (loss * valid).sum() / valid.sum().clamp_min(1)


def audit(device, dtype):
    """Small forward and downstream-gradient checks, including zero/tie cases."""
    torch.manual_seed(20261006)
    hidden = torch.randn(2, 9, 8, device=device, dtype=dtype)
    weight = torch.randn(19, 8, device=device, dtype=dtype) * .1
    bias = torch.randn(19, device=device, dtype=dtype) * .01
    ids = torch.tensor([18, 2, 0, 13, 7], device=device)
    membership = torch.zeros(19, device=device, dtype=torch.bool); membership[ids] = True
    atol, rtol = ((2e-6, 2e-5) if dtype == torch.float32 else (2e-4, 2e-3))
    cases = []
    for label in ('sparse', 'empty', 'all', 'tie', 'length_one'):
        h, w, b = hidden, weight, bias
        lm = torch.ones(2, 9, device=device, dtype=torch.long)
        if label == 'sparse': lm[:, ::2] = 0
        if label == 'empty': lm.zero_()
        if label == 'tie': w = torch.zeros_like(weight); b = torch.zeros_like(bias)
        if label == 'length_one': h = hidden[:, :1]; lm = lm[:, :1]
        expected, expected_mask = dense(h, w, ids, membership, lm, b)
        for chunk in (1, 4, 32):
            actual, actual_mask = chunked(h, w, ids, membership, lm, chunk, b)
            assert torch.equal(actual_mask, expected_mask), (label, chunk, 'mask')
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            # FP32 downstream head isolates the effects of preparing supervision;
            # this is not the NeMo fused CE benchmark or a full draft backward.
            torch.manual_seed(7)
            draft_hidden = torch.randn(2, h.shape[1], 6, device=device, dtype=torch.float32)
            draft_weight = torch.randn(ids.numel(), 6, device=device, dtype=torch.float32) * .1
            values = []
            for p, m in ((expected, expected_mask), (actual, actual_mask)):
                dh = draft_hidden.clone().requires_grad_(); dw = draft_weight.clone().requires_grad_()
                value = soft_ce(F.linear(dh, dw), p, m)
                grads = torch.autograd.grad(value, (dh, dw))
                values.append((value.detach(), tuple(g.detach() for g in grads)))
            torch.testing.assert_close(values[0][0], values[1][0], atol=atol, rtol=rtol)
            for g0, g1 in zip(values[0][1], values[1][1]):
                torch.testing.assert_close(g0, g1, atol=atol, rtol=rtol)
            cases.append({'case': label, 'chunk': chunk, 'mask_exact': True,
                          'probs_max_abs': float((actual - expected).abs().max()),
                          'loss_abs': float((values[0][0] - values[1][0]).abs()),
                          'grad_max_abs': max(float((g0-g1).abs().max()) for g0,g1 in zip(values[0][1],values[1][1]))})
    return {'dtype': str(dtype), 'device': str(device), 'atol': atol, 'rtol': rtol, 'cases': cases}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    state = {'state': 'starting', 'stage': 'correctness', 'runs_done': 0, 'runs_total': 48,
             'scope': 'target-output preparation only; no full model training or draft-loss timing',
             'nemo_commit': NEMO_SHA, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    dump(out/'status.json', state)
    try:
        correctness = [audit('cpu', torch.float32)]
        dump(out/'correctness.json', correctness)
        if args.cpu_only:
            state.update(state='complete', stage='cpu_correctness', runs_total=0)
            dump(out/'status.json', state); print('CPU_CORRECTNESS_PASSED', flush=True); return
        assert torch.cuda.is_available(), 'CUDA required'
        torch.backends.cuda.matmul.allow_tf32 = False
        correctness.extend([audit('cuda', torch.float32), audit('cuda', torch.float16)])
        dump(out/'correctness.json', correctness)
        assert args.checkpoint and Path(args.checkpoint).name == QWEN_SHA
        cfg = json.loads((Path(args.checkpoint)/'config.json').read_text())
        from loading import ReadOnlySafeOpen
        index = Path(args.checkpoint)/'model.safetensors.index.json'
        if index.exists():
            weight_file = Path(args.checkpoint)/json.loads(index.read_text())['weight_map']['lm_head.weight']
        else:
            weight_file = Path(args.checkpoint)/'model.safetensors'
        with ReadOnlySafeOpen(weight_file) as reader:
            weight = reader.get_tensor('lm_head.weight').to(device='cuda', dtype=torch.float16)
        vocab, width = weight.shape
        assert vocab == cfg['vocab_size'] and width == cfg['hidden_size']
        # Inputs are synthetic, weight is the real unchanged frozen Qwen head.
        # Actual model activations/integration are deliberately a later gate.
        generator = torch.Generator(device='cpu').manual_seed(20261006)
        ids = torch.randperm(vocab, generator=generator)[:32000].cuda()
        membership = torch.zeros(vocab, device='cuda', dtype=torch.bool); membership[ids] = True
        env = {'torch': torch.__version__, 'python': platform.python_version(), 'os': platform.platform(),
               'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
               'checkpoint_revision': QWEN_SHA, 'head_shape': list(weight.shape),
               'target_dtype': str(weight.dtype), 'head_weight_original_dtype': reader.header['lm_head.weight']['dtype'],
               'draft_vocab': ids.numel(), 'input_kind': 'synthetic normalized hidden states; real frozen output head',
               'seed': 20261006, 'lengths': [256, 1024, 4096], 'chunks': [64, 256, 1024], 'repeats': 4,
               'probs_tolerance': {'atol': 2e-4, 'rtol': 2e-3}, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        dump(out/'environment.json', env)
        checks = []
        def inputs(length):
            gen = torch.Generator(device='cpu').manual_seed(20261006+length)
            h = torch.randn(1, length, width, generator=gen).to('cuda', torch.float16)
            mask = torch.ones(1, length, device='cuda', dtype=torch.long); mask[:, ::5] = 0
            return h, mask
        # All large-shape correctness checks precede the dependent timing queue.
        for length in env['lengths']:
            hidden, lm = inputs(length)
            expected, expected_mask = dense(hidden, weight, ids, membership, lm)
            for chunk in env['chunks']:
                actual, actual_mask = chunked(hidden, weight, ids, membership, lm, chunk)
                row = {'length': length, 'chunk': chunk, 'mask_exact': torch.equal(actual_mask, expected_mask),
                       'mask_mismatches': int((actual_mask != expected_mask).sum()),
                       'probs_max_abs': float((actual-expected).abs().max()),
                       'probs_close': torch.allclose(actual, expected, atol=2e-4, rtol=2e-3)}
                checks.append(row); dump(out/'large_checks.json', checks)
                assert row['mask_exact'] and row['probs_close'], row
                del actual, actual_mask
            del hidden, lm, expected, expected_mask
            gc.collect(); torch.cuda.empty_cache()
        state.update(stage='timing', correctness='passed', checks_small=sum(len(x['cases']) for x in correctness), checks_large=len(checks))
        dump(out/'status.json', state)
        modes = ['dense', 'chunk64', 'chunk256', 'chunk1024']; rows = []
        for length in env['lengths']:
            hidden, lm = inputs(length)
            for repeat in range(env['repeats']):
                order = modes[repeat:]+modes[:repeat]
                for position, mode in enumerate(order):
                    call = (lambda: dense(hidden, weight, ids, membership, lm)) if mode == 'dense' else (
                        lambda: chunked(hidden, weight, ids, membership, lm, int(mode[5:])))
                    gc.collect(); torch.cuda.empty_cache()
                    for _ in range(2):
                        result = call(); del result
                    torch.cuda.synchronize()
                    allocated_before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter(); result = call(); torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    row = {'length': length, 'repeat': repeat, 'order_position': position, 'mode': mode,
                           'wall_ms': elapsed*1000, 'allocated_before': allocated_before,
                           'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved(),
                           'incremental_peak_allocated': torch.cuda.max_memory_allocated()-allocated_before}
                    assert math.isfinite(elapsed) and elapsed > 0
                    rows.append(row); del result
                    dump(out/'rows.json', rows)
                    state['runs_done'] = len(rows); dump(out/'status.json', state)
                    print('MEASURE',len(rows),'/',state['runs_total'],length,mode,round(row['wall_ms'],3),flush=True)
            del hidden, lm
            gc.collect(); torch.cuda.empty_cache()
        assert len(rows) == state['runs_total']
        state.update(state='complete', stage='done', finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        dump(out/'status.json', state); print('SUPERVISION_PROBE_COMPLETE', flush=True)
    except Exception as exc:
        state.update(state='failed', error=str(exc)[:2000])
        dump(out/'status.json', state)
        (out/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
