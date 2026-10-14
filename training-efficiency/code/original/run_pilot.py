"""Small greedy pilot; clean timings, exact replay, and separate profiler runs."""
import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'EAGLE'))

import torch
import transformers
from huggingface_hub import HfApi, snapshot_download
from metadata import finish_tensor_metadata
from topk_patch import make_topk
from reference_metadata import build_reference_metadata

COMMIT = 'cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b'
CHECKPOINT_REVISIONS = {
    'Qwen/Qwen3-1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
    'AngelSlim/Qwen3-1.7B_eagle3': '94441b48acc5804677ae12259617c83323b543a9',
}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def resolve_checkpoint(repo_id, explicit, needles):
    if explicit:
        return explicit, {'path_kind': 'explicit', 'source': explicit}
    # Kaggle mounts are versioned; record input source in kernel metadata too.
    if Path('/kaggle/input').exists():
        for p in sorted(Path('/kaggle/input').rglob('config.json')):
            if all(n in str(p.parent).lower() for n in needles):
                return str(p.parent), {'path_kind': 'kaggle_mount', 'source': str(p.parent)}
    info = HfApi().model_info(repo_id, revision=CHECKPOINT_REVISIONS[repo_id])
    sha = info.sha
    weights = '*.safetensors' if any(x.rfilename.endswith('.safetensors') for x in info.siblings) else '*.bin'
    path = snapshot_download(repo_id, revision=sha,
                             allow_patterns=['*.json', weights, '*.model', '*.txt', '*.jinja'])
    return path, {'path_kind': 'hf_snapshot', 'repo_id': repo_id, 'revision': sha}


def verify_vendor():
    manifest = json.loads((ROOT / 'code' / 'eagle_hashes.json').read_text())
    if manifest['commit'] != COMMIT:
        raise RuntimeError('Unexpected EAGLE commit')
    for name, sha in manifest['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Modified vendor file: ' + name)


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ROOT / 'results' / 'pilot'))
    p.add_argument('--base-path')
    p.add_argument('--draft-path')
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--prompt-limit', type=int, default=6)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--total-nodes', type=int, default=64)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--top-k', type=int, default=10)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dump(out / 'status.json', {'state': 'starting', 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU tests are separate')
    verify_vendor()
    random.seed(20261005); torch.manual_seed(20261005)
    env = dict(python=platform.python_version(), os=platform.platform(),
               torch=torch.__version__, transformers=transformers.__version__,
               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
               capability=torch.cuda.get_device_capability(0),
               cuda_free_total_bytes=torch.cuda.mem_get_info(), eagle_commit=COMMIT)
    dump(out / 'environment.json', env)
    base, base_info = resolve_checkpoint('Qwen/Qwen3-1.7B', args.base_path, ['qwen-3', '1.7b'])
    draft, draft_info = resolve_checkpoint('AngelSlim/Qwen3-1.7B_eagle3', args.draft_path, ['qwen3-1-7b-eagle3'])
    dump(out / 'config.json', dict(vars(args), seed=20261005, base=base_info, draft=draft_info,
                                  timing_scope='generation_including_prefill_and_initial_draft',
                                  stopping='upstream strict > cap; actual overshoot recorded; EOS disabled',
                                  precision='fp16', batch=1))
    print('CHECKPOINTS_RESOLVED', flush=True)
    from eagle.model.ea_model import EaModel
    import eagle.model.ea_model as ea_module
    t = time.perf_counter()
    model = EaModel.from_pretrained(use_eagle3=True, base_model_path=base, ea_model_path=draft,
                                   total_token=args.total_nodes, depth=args.depth, top_k=args.top_k,
                                   torch_dtype=torch.float16, device_map={'': 0}).eval()
    tok = model.get_tokenizer()
    upstream = model.ea_layer.topK_genrate
    env.update(load_sec=time.perf_counter()-t,
               target_attention_class=type(model.base_model.model.layers[0].self_attn).__name__,
               target_attention_source_sha256=hashlib.sha256(inspect.getsource(
                   type(model.base_model.model.layers[0].self_attn).forward).encode()).hexdigest(),
               target_attention_config=getattr(model.base_model.config, '_attn_implementation', None),
               target_attention_note='Pinned Qwen3 KV forward explicitly uses eager_attention_forward; see vendor manifest',
               target_params=sum(x.numel() for x in model.base_model.parameters()),
               draft_params=sum(x.numel() for x in model.ea_layer.parameters()))
    dump(out / 'environment.json', env)
    print('MODEL_LOADED', round(env['load_sec'], 2), 'sec', flush=True)
    prompts = json.loads((ROOT / 'prompts.json').read_text())[:args.prompt_limit]
    encoded = []
    for prompt in prompts:
        text = tok.apply_chat_template([{'role': 'user', 'content': prompt['text']}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = torch.tensor(tok([text]).input_ids, device='cuda')
        encoded.append((prompt, ids))
    dump(out / 'prompts.json', prompts)
    cache_length = max(x.shape[1] for _, x in encoded) + args.tokens + args.total_nodes + 96

    @contextlib.contextmanager
    def no_eos():
        eos = tok.eos_token_id
        tok.eos_token_id = -1
        try:
            yield
        finally:
            tok.eos_token_id = eos

    def generation(ids, mode, n_tokens=args.tokens):
        with torch.no_grad(), no_eos():
            method = model.naivegenerate if mode == 'ar' else model.eagenerate
            # AR produces cap+1; EAGLE may overshoot by several tokens.
            cap = n_tokens-1 if mode == 'ar' else n_tokens
            return method(ids, temperature=0.0, max_new_tokens=cap,
                          max_length=cache_length, log=True)

    # Capture is a separate diagnostic run. Never copy trees during clean timing.
    captured = []
    def observer(parents, result):
        if len(captured) < 24:
            captured.append({'parents': parents.detach().cpu().clone(),
                             'outputs': tuple(x.detach().cpu().clone() for x in result)})
    model.ea_layer.topK_genrate = make_topk(model.ea_layer, upstream, observer=observer)
    generation(encoded[0][1], 'upstream', n_tokens=32)
    model.ea_layer.topK_genrate = upstream
    torch.save(captured, out / 'replay.pt')
    for item in captured:
        tokens = item['outputs'][0][0]
        new = finish_tensor_metadata(tokens, item['parents'], 'cpu', None)
        for a, b in zip(new, item['outputs']):
            if not torch.equal(a, b):
                raise AssertionError('Metadata replay mismatch')
    dump(out / 'correctness.json', {'replay_trees': len(captured), 'all_metadata_exact': True,
                                   'generation_checks': []})
    print('REPLAY_EXACT', len(captured), flush=True)

    # Short replay microbenchmark, CPU inputs only; no claim about transfer cost.
    replay_methods = {'upstream': build_reference_metadata(),
                      'tensor_cpu': lambda tokens, parents: finish_tensor_metadata(tokens, parents, 'cpu', None)}
    replay_times = {name: [] for name in replay_methods}
    for rep in range(108):
        order = list(replay_methods.items())
        if rep % 2:
            order.reverse()
        for name, method in order:
            t = time.perf_counter()
            for item in captured:
                method(item['outputs'][0][0], item['parents'])
            if rep >= 8:
                replay_times[name].append((time.perf_counter()-t)/len(captured))
    dump(out / 'replay_microbenchmark.json', dict(
         median_ms={k: 1000*statistics.median(v) for k,v in replay_times.items()},
         input_device='cpu', includes_d2h=False,
         note='Author tail replay omits cleanup of previous phase tensors; not full-pipeline speedup'))
    variants = {'upstream': upstream,
                'tensor_cpu': make_topk(model.ea_layer, upstream, variant='tensor_cpu')}
    for name, method in variants.items():
        model.ea_layer.topK_genrate = method
        generation(encoded[0][1], name, n_tokens=32)
    generation(encoded[0][1], 'ar', n_tokens=32)
    checks = []
    references = {}
    rows = []
    for rep in range(args.repeats):
        for j, (prompt, ids) in enumerate(encoded):
            order = ['upstream', 'tensor_cpu', 'ar']
            if (rep+j) % 2:
                order.reverse()
            for mode in order:
                model.ea_layer.topK_genrate = variants.get(mode, upstream)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); start = time.perf_counter()
                result, new, idx = generation(ids, mode)
                torch.cuda.synchronize(); elapsed = time.perf_counter()-start
                generated = result[0, ids.shape[1]:].tolist()
                new = int(new); cycles = int(idx)+1
                assert new == len(generated)
                row = dict(prompt_id=prompt['id'], language=prompt['language'], kind=prompt['kind'],
                           rep=rep, mode=mode, input_tokens=int(ids.shape[1]), output_tokens=new,
                           requested_tokens=args.tokens, cycles=cycles, tau=new/cycles,
                           accepted_draft_tokens=(new-cycles if mode != 'ar' else 0),
                           sec=elapsed, ms_per_token=1000*elapsed/new,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
                rows.append(row)
                references[(rep, prompt['id'], mode)] = generated
                dump(out / 'rows.json', rows)
                dump(out / 'status.json', {'state': 'running', 'completed_runs': len(rows)})
                print('RUN', rep, prompt['id'], mode, round(row['ms_per_token'], 4), 'ms/token', flush=True)
            old = references[(rep, prompt['id'], 'upstream')]
            new = references[(rep, prompt['id'], 'tensor_cpu')]
            ar = references[(rep, prompt['id'], 'ar')]
            check = dict(rep=rep, prompt_id=prompt['id'], metadata_variant_full_exact=(old==new),
                         variant_first_difference=first_difference(old,new),
                         ar_comparison_tokens=min(len(old),len(ar)),
                         ar_first_difference=first_difference(old[:len(ar)], ar))
            checks.append(check)
            dump(out / 'correctness.json', {'replay_trees': len(captured), 'all_metadata_exact': True,
                                           'generation_checks': checks})
            if old != new:
                raise AssertionError('Generation changed under metadata-only intervention')
    with (out / 'rows.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    torch.save(references, out / 'generated_tokens.pt')
    summary = []
    for prompt, _ in encoded:
        speedups = []
        for rep in range(args.repeats):
            rr={r['mode']:r for r in rows if r['rep']==rep and r['prompt_id']==prompt['id']}
            speedups.append(rr['upstream']['sec']/rr['tensor_cpu']['sec'])
        summary.append(dict(prompt_id=prompt['id'], speedup_repeats=speedups,
                            median_speedup=statistics.median(speedups)))
    dump(out / 'summary.json', {'paired_by_prompt': summary,
         'note': 'Development pilot only; three repeats are not independent tasks; no holdout claim'})
    if args.profile:
        def wrap(fn, label):
            def call(*a, **kw):
                with torch.profiler.record_function(label):
                    return fn(*a, **kw)
            return call
        td, ep = ea_module.tree_decoding, ea_module.evaluate_posterior
        try:
            ea_module.tree_decoding = wrap(td, 'target_tree_verification')
            ea_module.evaluate_posterior = wrap(ep, 'posterior_selection')
            for mode in variants:
                patched = make_topk(model.ea_layer, upstream, variant=mode, ranges=True)
                model.ea_layer.topK_genrate = wrap(patched, 'draft_with_metadata')
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                     torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                     profile_memory=False, with_stack=False) as prof:
                    generation(encoded[0][1], mode, n_tokens=32)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(out / (mode + '_trace.json')))
                events=[]
                for e in prof.key_averages():
                    events.append(dict(key=e.key, count=e.count, cpu_total_us=e.cpu_time_total,
                        self_cpu_us=e.self_cpu_time_total,
                        device_total_us=getattr(e,'device_time_total',0)))
                dump(out / (mode + '_profile.json'), events)
        finally:
            ea_module.tree_decoding, ea_module.evaluate_posterior = td, ep
            model.ea_layer.topK_genrate = upstream
    dump(out / 'status.json', {'state': 'complete', 'completed_runs': len(rows)})
    print('PILOT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
