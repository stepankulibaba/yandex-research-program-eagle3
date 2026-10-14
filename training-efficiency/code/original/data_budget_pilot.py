"""Finite warm-start data-cost pilot, fixed FP16/master protocol, home GPU."""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import gc
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'code' / 'original'
sys.path.insert(0, str(REPO))
import torch
import probe_nemo_ttt as base
from fp32_master_adamw import MasterAdamW, LOSS_SCALE, OPTIMIZER_KWARGS, cpu_gate
from loading import read_only_safetensors
from probe_nemo_supervision import QWEN_SHA, NEMO_SHA, distribution, shift


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(4 * 2**20):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def metrics(res):
    return {name: getattr(res, name).detach().cpu().tolist()
            for name in ('loss', 'accuracy', 'valid_tokens', 'step_prefix_hits', 'step_valid')}


def cache_from_doc(doc, head, ids, membership, dense_control=False):
    n = doc['tokens'].shape[1]
    batch = base.pack([doc], n, 'cuda')
    if dense_control:
        raw = base.prepare('dense', batch, head, ids, membership, 256)
        probs, mask = distribution(raw['target_logits'], ids, membership, shift(batch['loss_mask']))
        supervision = {'target_probs': probs, 'position_mask': mask}
    else:
        supervision = base.prepare('chunk256', batch, head, ids, membership, 256)
    probs, mask = supervision['target_probs'], supervision['position_mask']
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]), atol=2e-5, rtol=2e-5)
    assert mask.any(), 'No supervised tokens, refusing training'
    assert not mask[:, -1].any()
    cache = {'input_ids': shift(batch['tokens']), 'attention_mask': batch['attention_mask'],
             'loss_mask': shift(batch['loss_mask']), 'aux_hidden_states': batch['aux'],
             'position_ids': batch['position_ids'], 'seq_lens': batch['seq_lens'],
             'doc_remaining': batch['doc_remaining'], **supervision}
    return batch, {k: v.detach().cpu().clone() for k, v in cache.items()}


def dense_parity(trainer, batch, head, ids, membership):
    raw = base.prepare('dense', batch, head, ids, membership, 256)
    p, m = distribution(raw['target_logits'], ids, membership, shift(batch['loss_mask']))
    results = []
    for supervision in [raw, {'target_probs': p.cpu().cuda(), 'position_mask': m.cpu().cuda()}]:
        trainer.zero_grad(set_to_none=True)
        res = base.forward(trainer, batch, supervision)
        res.loss.backward()
        results.append((base.metrics(res), base.grads(trainer.draft_model)))
    m0, g0 = results[0]
    m1, g1 = results[1]
    check = {'metrics_exact': all(torch.equal(m0[k], m1[k]) for k in m0),
             'gradients_exact': all(torch.equal(g0[k], g1[k]) for k in g0),
             'gradient_names_exact': set(g0) == set(g1) == {n for n, p in trainer.draft_model.named_parameters() if p.requires_grad},
             'finite': all(bool(torch.isfinite(g).all()) for g in g0.values()),
             'parameters': len(g0), 'loss': float(m0['loss'])}
    check['passed'] = all(check[k] for k in ('metrics_exact', 'gradients_exact', 'gradient_names_exact', 'finite'))
    trainer.zero_grad(set_to_none=True)
    return check


def projection_diagnostic(batch, head, ids, membership):
    with torch.no_grad():
        raw = base.prepare('dense', batch, head, ids, membership, 256)['target_logits']
        repeat = base.prepare('dense', batch, head, ids, membership, 256)['target_logits']
        expected, mask = distribution(raw, ids, membership, shift(batch['loss_mask']))
        rows = []
        for size in (256, 512):
            actual = base.prepare('chunk256', batch, head, ids, membership, size)
            rows.append({'chunk': size, 'mask_exact': torch.equal(mask, actual['position_mask']),
                         'probabilities_exact': torch.equal(expected, actual['target_probs']),
                         'probabilities_close_old_tolerance': bool(torch.allclose(expected, actual['target_probs'], atol=2e-4, rtol=2e-3)),
                         'max_abs': float((expected - actual['target_probs']).abs().max())})
        # Separate numerical reference, never substituted into training.
        reference = shift(torch.nn.functional.linear(batch['hidden'].float(), head.float()))
        ref_probs, _ = distribution(reference, ids, membership, shift(batch['loss_mask']))
        return {'scope': 'diagnostic only, training remains dense FP16',
                'length': batch['tokens'].shape[1], 'dense_repeat_exact': torch.equal(raw, repeat),
                'chunks': rows, 'dense_vs_fp32_probs_max_abs': float((expected - ref_probs).abs().max()),
                'dense_vs_fp32_argmax_differences': int((raw.argmax(-1) != reference.argmax(-1)).sum())}


def evaluate(trainer, draft, caches, out, update):
    draft.eval()
    rows = []
    begun = time.perf_counter()
    with torch.no_grad():
        for item in caches:
            kw = {k: v.cuda() for k, v in item['cache'].items()}
            res = trainer(**kw)
            row = {'id': item['id'], 'update': update, **metrics(res)}
            assert torch.isfinite(res.loss) and res.loss > 0 and res.valid_tokens > 0
            rows.append(row)
            del kw, res
    torch.cuda.synchronize()
    total_valid = sum(r['valid_tokens'] for r in rows)
    hits = [sum(r['step_prefix_hits'][i] for r in rows) for i in range(7)]
    valid = [sum(r['step_valid'][i] for r in rows) for i in range(7)]
    summary = {'update': update, 'examples': len(rows), 'valid_tokens_all_ttt_steps': total_valid,
               'loss_weighted_by_valid_all_steps': sum(r['loss'] * r['valid_tokens'] for r in rows) / total_valid,
               'accuracy_weighted_by_valid_all_steps': sum(r['accuracy'] * r['valid_tokens'] for r in rows) / total_valid,
               'step_prefix_hits': hits, 'step_valid': valid,
               'prefix_survival': [a / max(b, 1) for a, b in zip(hits, valid)],
               'wall_seconds': time.perf_counter() - begun,
               'scope': 'offline teacher-forced TTT diagnostic, not real acceptance length or speedup'}
    # NeMo loss combines per-step means with its own weighting; this aggregation
    # is only a fixed validation diagnostic, not a pooled exact CE estimator.
    dump(out / f'validation_{update:03d}.json', {'rows': rows, 'summary': summary})
    draft.train()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate-sha', required=True)
    ap.add_argument('--runner-sha', required=True)
    ap.add_argument('--protocol-sha', required=True)
    ap.add_argument('--dense-control', action='store_true')
    ap.add_argument('--generated-sha')
    a = ap.parse_args()
    source = REPO / 'data_budget_candidates_20261008.json'
    assert sha(source) == a.candidate_sha and sha(__file__) == a.runner_sha
    protocol_name = 'DATA_BUDGET_DENSE_CONTROL.md' if a.dense_control else 'DATA_BUDGET_PILOT.md'
    assert sha(REPO / protocol_name) == a.protocol_sha
    out = ROOT / ('data_budget_dense_20261008' if a.dense_control else 'data_budget_pilot_20261008')
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    state = {'state': 'running', 'stage': 'cpu_gate', 'started_utc': now(),
             'counts': {'selected': [0, 32], 'generated': [0, 32], 'captured': [0, 32],
                        'teacher_cached': [0, 32], 'gates': [0, 4],
                        'updates': [0, 24], 'validation_examples': [0, 32]}}
    draft = None

    def status(stage=None):
        if stage:
            state['stage'] = stage
        state['updated_utc'] = now()
        dump(out / 'status.json', state)

    def gate(name, details):
        state['counts']['gates'][0] += 1
        dump(out / f'gate_{name}.json', details)
        status()

    status()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(20261008)
        assert torch.cuda.is_available()
        assert torch.cuda.mem_get_info()[0] >= 6 * 2**30
        checks = cpu_gate()
        gate('optimizer_cpu', checks)
        import run_pilot as pilot
        from transformers import AutoTokenizer, Qwen3ForCausalLM
        pilot.verify_vendor()
        path = ROOT / 'hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots' / QWEN_SHA
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        candidates = json.loads(source.read_text(encoding='utf-8'))
        chosen = []
        rejected = []
        for split, label, count in [('train_sft', 'train', 24), ('test_sft', 'validation', 8)]:
            selected = []
            for item in candidates['records']:
                if item['source_split'] != split:
                    continue
                ids = tok.apply_chat_template([{'role': 'user', 'content': item['prompt']}],
                                             tokenize=True, add_generation_prompt=True, enable_thinking=False)
                if not 16 <= len(ids) <= 256:
                    rejected.append({'id': item['id'], 'prompt_tokens': len(ids), 'reason': 'fixed_prompt_length_gate'})
                    continue
                selected.append({**item, 'pilot_split': label, 'prompt_ids': ids})
                if len(selected) == count:
                    break
            assert len(selected) == count, (label, len(selected))
            chosen.extend(selected)
        assert len({x['id'] for x in chosen}) == 32
        assert len({x['prompt_sha256'] for x in chosen}) == 32
        dump(out / 'selected.json', {'selected': chosen, 'rejected_before_quota': rejected,
                                    'selection_frozen_utc': now(), 'no_model_outputs_used': True})
        state['counts']['selected'][0] = 32
        inventory = ['probe_nemo_ttt.py', 'probe_nemo_supervision.py', 'nemo_ttt_runtime.py',
                     'fp32_master_adamw.py', 'loading.py', 'run_pilot.py',
                     'references/nemo_20261006/source_hashes.json']
        dump(out / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
             'gpu': torch.cuda.get_device_name(), 'python': platform.python_version(),
             'os': platform.platform(), 'target_revision': QWEN_SHA, 'draft_revision': base.DRAFT_SHA,
             'nemo_commit': NEMO_SHA, 'dataset_revision_observed': candidates['revision_observed'],
             'candidate_sha256': a.candidate_sha, 'runner_sha256': a.runner_sha,
             'protocol_sha256': a.protocol_sha, 'selected_sha256': sha(out / 'selected.json'),
             'seed': 20261008, 'ttt_steps': 7, 'microbatch': 1, 'updates': 24,
             'teacher_preparation': 'original dense FP16' if a.dense_control else 'chunk256 FP16',
             'precision': 'FP16/eager draft and target; FP32 master AdamW; PyTorch CE fallback',
             'optimizer': OPTIMIZER_KWARGS, 'loss_scale': LOSS_SCALE,
             'source_hashes': {p: sha(REPO / p) for p in inventory},
             'target_backend': 'transformers Qwen3ForCausalLM, eager, greedy no thinking',
             'training_scope': 'offline target; warm-start pretrained draft; embeddings frozen; unmodified pinned NeMo AST subset, not full recipe'})
        draft_path = ROOT / 'hf_cache/hub/models--AngelSlim--Qwen3-1.7B_eagle3/snapshots' / base.DRAFT_SHA
        source_weight_hashes = {str(p.relative_to(ROOT)): sha(p) for p in
                              [draft_path / 'pytorch_model.bin', *sorted(path.glob('*.safetensors'))]}
        assert len(source_weight_hashes) >= 2
        dump(out / 'source_weights_before.json', source_weight_hashes)
        status('target_generation')
        preparation = {}
        target_begin = time.perf_counter()
        with read_only_safetensors():
            target = Qwen3ForCausalLM.from_pretrained(str(path), torch_dtype=torch.float16,
                       device_map={'': 0}, attn_implementation='eager', local_files_only=True).eval().requires_grad_(False)
        preparation['target_load_seconds'] = time.perf_counter() - target_begin
        generated = []
        if a.dense_control:
            parent = ROOT / 'data_budget_pilot_20261008/generated.json'
            assert a.generated_sha and sha(parent) == a.generated_sha
            generated = json.loads(parent.read_text(encoding='utf-8'))
            assert len(generated) == len(chosen)
            for item, cached in zip(chosen, generated):
                assert item['id'] == cached['id'] and item['prompt_ids'] == cached['prompt_ids']
                assert item['pilot_split'] == cached['pilot_split']
            dump(out / 'generated.json', generated)
            assert sha(out / 'generated.json') == a.generated_sha
            dump(out / 'inherited_generation.json', {'parent': str(parent), 'sha256': a.generated_sha,
                 'generation_rerun': False, 'parent_failure_retained': True})
            state['counts']['generated'][0] = 32
            status()
        with torch.no_grad():
            for item in ([] if a.dense_control else chosen):
                ids = torch.tensor([item['prompt_ids']], dtype=torch.long, device='cuda')
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                begun = time.perf_counter()
                seq = target.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                        max_new_tokens=128, temperature=None, top_p=None, top_k=None,
                        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                n = seq.shape[1] - ids.shape[1]
                assert n > 0 and seq.shape[1] <= 384
                generated.append({**item, 'tokens': seq.cpu().tolist()[0],
                    'new_tokens': n, 'response': tok.decode(seq[0, ids.shape[1]:].cpu(), skip_special_tokens=False),
                    'generation_seconds': time.perf_counter() - begun,
                    'generation_peak_mib': torch.cuda.max_memory_allocated() / 2**20})
                dump(out / 'generated.json', generated)
                state['counts']['generated'][0] = len(generated)
                status()
                print('GENERATED', len(generated), item['pilot_split'], n, flush=True)
                del ids, seq
        preparation['generation_seconds_sum'] = sum(x['generation_seconds'] for x in generated)
        layers = [1, len(target.model.layers) // 2 - 1, len(target.model.layers) - 4]
        captured = {}

        def hook(i):
            def record(module, args, result):
                hidden = result[0] if isinstance(result, tuple) else result
                assert hidden.dim() == 3 and hidden.shape[0] == 1
                captured[i] = hidden.detach()
            return record

        handles = [target.model.layers[i].register_forward_hook(hook(i)) for i in layers]
        docs = []
        capture_rows = []
        status('target_capture')
        with torch.no_grad():
            for index, item in enumerate(generated):
                seq = torch.tensor([item['tokens']], dtype=torch.long, device='cuda')
                torch.cuda.synchronize()
                begun = time.perf_counter()
                hidden = target.model(input_ids=seq, use_cache=False).last_hidden_state
                aux = torch.cat([captured[i] for i in layers], -1)
                mask = torch.zeros_like(seq)
                mask[:, len(item['prompt_ids']):] = 1
                doc = {'tokens': seq.cpu(), 'hidden': hidden.cpu(), 'aux': aux.cpu(), 'loss_mask': mask.cpu()}
                torch.cuda.synchronize()
                capture_rows.append({'id': item['id'], 'seconds': time.perf_counter() - begun,
                                     'sequence_tokens': seq.shape[1], 'response_tokens': item['new_tokens']})
                docs.append(doc)
                if index == 0:
                    captured.clear()
                    repeat = target.model(input_ids=seq, use_cache=False).last_hidden_state
                    repeat_aux = torch.cat([captured[i] for i in layers], -1)
                    check = {'hidden_exact': torch.equal(hidden, repeat), 'aux_exact': torch.equal(aux, repeat_aux)}
                    assert all(check.values()), check
                    gate('capture_repeat', check)
                    del repeat, repeat_aux
                captured.clear()
                del seq, hidden, aux, mask
                state['counts']['captured'][0] = index + 1
                status()
        for handle in handles:
            handle.remove()
        dump(out / 'capture_rows.json', capture_rows)
        preparation['capture_seconds_sum'] = sum(x['seconds'] for x in capture_rows)
        head = target.lm_head.weight.detach()
        embedding = target.get_input_embeddings().weight.detach()
        del target
        gc.collect()
        torch.cuda.empty_cache()
        draft, ids, membership = base.load_draft(ROOT, embedding, out)
        initial = base.digest(draft)
        previous = json.loads((ROOT / 'nemo_ttt_20261006/environment.json').read_text())['initial_weights_sha256']
        assert initial == previous, 'Initial full draft differs from previous qualified checkpoint'
        frozen_before = hashlib.sha256(embedding.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        del embedding
        trainer = base.R.Eagle3TrainerModule(draft, selected_token_ids=ids, selected_token_mask=membership, ttt_steps=7)
        caches = []
        teacher_rows = []
        status('teacher_cache')
        for index, (item, doc) in enumerate(zip(generated, docs)):
            torch.cuda.synchronize()
            begun = time.perf_counter()
            batch, cache = cache_from_doc(doc, head, ids, membership, a.dense_control)
            torch.cuda.synchronize()
            teacher_rows.append({'id': item['id'], 'seconds': time.perf_counter() - begun,
                  'sequence_tokens': doc['tokens'].shape[1],
                  'supervised_positions': int(cache['position_mask'].sum()),
                  'cache_bytes': sum(v.numel() * v.element_size() for v in cache.values())})
            if index == 0:
                if a.dense_control:
                    dump(out / 'projection_diagnostic.json', projection_diagnostic(batch, head, ids, membership))
                dense = base.prepare('dense', batch, head, ids, membership, 256)
                p0, m0 = distribution(dense['target_logits'], ids, membership, shift(batch['loss_mask']))
                p1 = cache['target_probs'].cuda()
                check = {'mask_exact': torch.equal(m0.cpu(), cache['position_mask']),
                         'probabilities_close': bool(torch.allclose(p0, p1, atol=2e-4, rtol=2e-3)),
                         'max_abs': float((p0 - p1).abs().max())}
                if a.dense_control:
                    check['probabilities_exact'] = torch.equal(p0, p1)
                    assert check['mask_exact'] and check['probabilities_exact'], check
                else:
                    assert check['mask_exact'] and check['probabilities_close'], check
                gate('teacher_dense_cache' if a.dense_control else 'teacher_dense_chunk', check)
                del dense, p0, m0, p1
                check = dense_parity(trainer, batch, head, ids, membership) if a.dense_control else base.compare(trainer, batch, head, ids, membership, 256, 2e-4, 2e-3)
                assert check['passed'], 'Dense/chunk draft-gradient gate failed'
                gate('draft_dense_cache' if a.dense_control else 'draft_dense_chunk', check)
            caches.append({'id': item['id'], 'split': item['pilot_split'], 'cache': cache})
            del batch
            state['counts']['teacher_cached'][0] = index + 1
            status()
        dump(out / 'teacher_rows.json', teacher_rows)
        preparation['teacher_cache_seconds_sum'] = sum(x['seconds'] for x in teacher_rows)
        assert state['counts']['gates'][0] == 4 and base.digest(draft) == initial
        torch.save({'caches': caches, 'aux_layers': layers}, out / 'teacher_cache.pt')
        dump(out / 'cache_manifest.json', {'sha256': sha(out / 'teacher_cache.pt'),
             'initial_draft_sha256': initial, 'aux_layers': layers,
             'rows': teacher_rows, 'checkpoint_source_unchanged': True})
        del head, docs
        gc.collect()
        torch.cuda.empty_cache()
        preparation['total_preparation_wall_seconds'] = time.perf_counter() - started
        dump(out / 'preparation.json', preparation)
        train = [x for x in caches if x['split'] == 'train']
        validation = [x for x in caches if x['split'] == 'validation']
        curves = []
        status('validation_0')
        curves.append(evaluate(trainer, draft, validation, out, 0))
        state['counts']['validation_examples'][0] = 8
        dump(out / 'validation_curve.json', curves)
        opt = MasterAdamW(draft)
        updates = []
        train_started = time.perf_counter()
        for index, item in enumerate(train, 1):
            status('training')
            opt.zero_grad()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begun = time.perf_counter()
            kw = {k: v.cuda() for k, v in item['cache'].items()}
            res = trainer(**kw)
            assert torch.isfinite(res.loss) and res.loss > 0 and res.valid_tokens > 0
            (res.loss * LOSS_SCALE).backward()
            norm = opt.step()  # Rejects nonfinite gradients before updating.
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begun
            row = {'update': index, 'id': item['id'], 'sequence_tokens': kw['input_ids'].shape[1],
                   'supervised_positions_step0': int(kw['position_mask'].sum()),
                   'update_seconds': elapsed,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                   'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                   'grad_norm_before_clip': float(norm), **metrics(res)}
            for (_, parameter), master in zip(opt.named, opt.masters):
                st = opt.optimizer.state[master]
                assert all(bool(torch.isfinite(v).all()) for v in
                           (parameter, master, st['exp_avg'], st['exp_avg_sq'])), 'Nonfinite optimizer state'
            row['all_weights_and_moments_finite'] = True
            updates.append(row)
            dump(out / 'updates.json', updates)
            state['counts']['updates'][0] = index
            status()
            print('UPDATE', index, 'seconds', round(elapsed, 3), 'loss', row['loss'], flush=True)
            opt.zero_grad()
            del kw, res, norm
            if index % 8 == 0:
                status(f'validation_{index}')
                curves.append(evaluate(trainer, draft, validation, out, index))
                state['counts']['validation_examples'][0] += 8
                dump(out / 'validation_curve.json', curves)
                status()
        training_wall = time.perf_counter() - train_started
        status('save_checkpoint')
        torch.save({'trainable_parameters': {n: p.detach().cpu().clone() for n, p in draft.named_parameters() if p.requires_grad},
                    'draft_revision': base.DRAFT_SHA, 'target_revision': QWEN_SHA,
                    'initial_draft_sha256': initial, 'updates': 24}, out / 'draft_trainable_024.pt')
        final = base.digest(draft)
        assert final != initial, 'Weights did not change'
        frozen_after = hashlib.sha256(draft.model.embed_tokens.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        assert frozen_after == frozen_before, 'Frozen embedding changed'
        del trainer, opt, caches, train, validation
        gc.collect()
        torch.cuda.empty_cache()
        # Verify original checkpoint files remain untouched; loading.json records strict mapping.
        source_after = {p: sha(ROOT / p) for p in source_weight_hashes}
        assert source_after == source_weight_hashes, 'Original checkpoint files changed'
        dump(out / 'source_weights_after.json', source_after)
        dump(out / 'completion.json', {'initial_draft_sha256': initial, 'final_draft_sha256': final,
             'checkpoint_sha256': sha(out / 'draft_trainable_024.pt'), 'frozen_embedding_unchanged': True,
             'source_draft_checkpoint_sha256': sha(draft_path / 'pytorch_model.bin'),
             'training_wall_including_validation_after_updates_seconds': training_wall,
             'update_seconds_sum': sum(r['update_seconds'] for r in updates),
             'validation_seconds_sum': sum(r['wall_seconds'] for r in curves),
             'preparation': preparation, 'total_wall_seconds': time.perf_counter() - started,
             'finished_utc': now()})
        state.update(state='complete', stage='done', finished_utc=now())
        status()
    except Exception:
        state.update(state='failed', error=traceback.format_exc(), finished_utc=now())
        status()
        raise


if __name__ == '__main__':
    main()
