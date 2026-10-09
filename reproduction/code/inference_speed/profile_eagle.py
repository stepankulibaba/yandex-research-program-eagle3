"""Where does the time go? The authors' EAGLE-3 code, unchanged, timed and profiled from the outside.

    cd EAGLE && ../venv_eagle/bin/python ../profile_eagle.py --model ../models/llama31-8b-instruct \
        --draft ../models/eagle3-llama31-8b [--bench mt_bench] [--questions 5] [--out ../results/main/profile]

Run it when the GPU is free (not next to a running night). Same model loading, prompts and tree as the
authors' gen_ea_answer_llama3chat.py (fp16, 60 tokens, depth 7, top-k 10, T=0), first turn of the first questions.

Three measurements:
  1. plain timing of eagenerate and naivegenerate (no instrumentation): ms per EAGLE round, ms per plain token,
     tau, speed-up, and the cost of one round in plain steps;
  2. the same with a CUDA sync around each phase of a round, to split it into
        draft (ea_layer.topK_genrate) | verify (tree_decoding: target forward over the tree) |
        accept (evaluate_posterior) | update (update_inference_inputs without the draft) | rest of the loop;
     and for plain generation: target forward vs the rest of the loop;
  3. torch.profiler over one question each: how long the GPU actually computes (sum of kernel times) versus wall
     time, kernel launches per round / per token, and the top operations on CPU and GPU.
Writes summary.md, summary.json and two Chrome traces (open in https://ui.perfetto.dev).
"""
import argparse
import contextlib
import json
from pathlib import Path
import time

import torch

SYSTEM = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
          "  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
          "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make "
          "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't "
          "know the answer to a question, please don't share false information.")


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


class PhaseTimer:
    """Wraps functions so that each call is timed between two CUDA syncs; accumulates per phase."""

    def __init__(self):
        self.seconds = {}
        self.calls = {}

    def wrap(self, name, fn):
        def timed(*args, **kwargs):
            t0 = sync_time()
            try:
                return fn(*args, **kwargs)
            finally:
                self.seconds[name] = self.seconds.get(name, 0.0) + sync_time() - t0
                self.calls[name] = self.calls.get(name, 0) + 1
        return timed


@contextlib.contextmanager
def patched(module, names, timer):
    originals = {n: getattr(module, n) for n in names}
    try:
        for n in names:
            setattr(module, n, timer.wrap(n, originals[n]))
        yield
    finally:
        for n, fn in originals.items():
            setattr(module, n, fn)


def kernel_stats(prof):
    """(total kernel time in s, kernel launches) from a torch.profiler run."""
    total_us, launches = 0.0, 0
    for e in prof.key_averages():
        if str(getattr(e, 'device_type', '')).endswith('CUDA'):
            total_us += self_time_us(e, 'gpu')
            launches += e.count
    return total_us / 1e6, launches


def self_time_us(event, kind):
    """Self time of a profiler event on 'cpu' or 'gpu' (the GPU field was renamed across torch versions)."""
    if kind == 'cpu':
        return getattr(event, 'self_cpu_time_total', 0) or 0
    return getattr(event, 'self_device_time_total', None) or getattr(event, 'self_cuda_time_total', 0) or 0


def top_ops(prof, kind, n=12):
    events = sorted(prof.key_averages(), key=lambda e: self_time_us(e, kind), reverse=True)[:n]
    return [{'name': e.key[:70], 'count': e.count, 'ms': self_time_us(e, kind) / 1000} for e in events]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--draft', required=True)
    parser.add_argument('--bench', default='mt_bench')
    parser.add_argument('--questions', type=int, default=5)
    parser.add_argument('--out', default='../results/main/profile')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import eagle.model.ea_model as ea_module
    from eagle.model.ea_model import EaModel
    model = EaModel.from_pretrained(base_model_path=args.model, ea_model_path=args.draft, total_token=60, depth=7,
                                    top_k=10, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto',
                                    use_eagle3=True)
    model.eval()
    tok = model.get_tokenizer()
    rows = [json.loads(line) for line in Path(f'eagle/data/{args.bench}/question.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()][:args.questions]
    prompts = []
    for q in rows:
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q['turns'][0]}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(torch.as_tensor(tok([text], add_special_tokens=False).input_ids).cuda())

    def eagle(ids):
        return model.eagenerate(ids, temperature=0.0, log=True, is_llama3=True)

    def plain(ids):
        return model.naivegenerate(ids, temperature=0.0, log=True, is_llama3=True)

    with torch.no_grad():
        for _ in range(2):                      # warm-up, as the authors do
            eagle(prompts[0])
            plain(prompts[0])

        # 1. Plain timing.
        e_time = e_tokens = e_rounds = b_time = b_tokens = 0
        for ids in prompts:
            t0 = sync_time()
            _, new_token, idx = eagle(ids)
            e_time += sync_time() - t0
            e_tokens += int(new_token)
            e_rounds += int(idx) + 1
            t0 = sync_time()
            _, new_token, _ = plain(ids)
            b_time += sync_time() - t0
            b_tokens += int(new_token)
        round_ms = 1000 * e_time / e_rounds
        token_ms = 1000 * b_time / b_tokens
        timing = {'eagle_tok_s': e_tokens / e_time, 'plain_tok_s': b_tokens / b_time,
                  'speedup': (e_tokens / e_time) / (b_tokens / b_time), 'tau': e_tokens / e_rounds,
                  'ms_per_round': round_ms, 'ms_per_plain_token': token_ms,
                  'round_cost_in_plain_steps': round_ms / token_ms, 'rounds': e_rounds, 'plain_tokens': b_tokens}

        # 2. Phases, with a CUDA sync around each (the syncs themselves add a little).
        # The draft (topK_genrate) runs once at prefill (inside initialize_tree) and once per round (inside
        # update_inference_inputs); only the per-round calls count as the round's draft time.
        timer = PhaseTimer()
        in_update = {'flag': False}
        draft_original = model.ea_layer.topK_genrate
        update_original = ea_module.update_inference_inputs

        def update_marked(*a, **kw):
            in_update['flag'] = True
            try:
                return update_original(*a, **kw)
            finally:
                in_update['flag'] = False

        def draft_split(*a, **kw):
            name = 'draft' if in_update['flag'] else 'draft at prefill'
            return timer.wrap(name, draft_original)(*a, **kw)

        model.ea_layer.topK_genrate = draft_split
        ea_module.update_inference_inputs = update_marked
        e_total = 0.0
        with patched(ea_module, ['initialize_tree', 'tree_decoding', 'evaluate_posterior',
                                 'update_inference_inputs'], timer):
            for ids in prompts:
                t0 = sync_time()
                eagle(ids)
                e_total += sync_time() - t0
        model.ea_layer.topK_genrate = draft_original
        ea_module.update_inference_inputs = update_original
        s = timer.seconds
        rounds = timer.calls['tree_decoding']
        in_phases = s['initialize_tree'] + s['tree_decoding'] + s['evaluate_posterior'] + s['update_inference_inputs']
        per_round = {
            'draft (topK_genrate)': 1000 * s['draft'] / rounds,
            'verify (target forward over the tree)': 1000 * s['tree_decoding'] / rounds,
            'accept (evaluate_posterior)': 1000 * s['evaluate_posterior'] / rounds,
            'update without the draft': 1000 * (s['update_inference_inputs'] - s['draft']) / rounds,
            'rest of the loop (Python, stop checks)': 1000 * (e_total - in_phases) / rounds,
            'prefill per question (initialize_tree), spread over rounds': 1000 * s['initialize_tree'] / rounds,
        }

        b_timer = PhaseTimer()
        forward = model.base_model.forward
        model.base_model.forward = b_timer.wrap('target forward', forward)
        b_total = b_tok = 0
        for ids in prompts:
            t0 = sync_time()
            _, new_token, _ = plain(ids)
            b_total += sync_time() - t0
            b_tok += int(new_token)
        model.base_model.forward = forward
        per_token = {'target forward': 1000 * b_timer.seconds['target forward'] / b_tok,
                     'rest of the loop (Python, stop checks)':
                         1000 * (b_total - b_timer.seconds['target forward']) / b_tok}

        # 3. torch.profiler: GPU busy time vs wall time.
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        profiles = {}
        for name, fn in (('eagle', eagle), ('plain', plain)):
            with torch.profiler.profile(activities=activities) as prof:
                t0 = sync_time()
                _, new_token, idx = fn(prompts[0])
                wall = sync_time() - t0
            kernels_s, launches = kernel_stats(prof)
            steps = (int(idx) + 1) if name == 'eagle' else int(new_token)
            profiles[name] = {'wall_s': wall, 'gpu_kernel_s': kernels_s, 'gpu_busy_share': kernels_s / wall,
                              'steps': steps, 'ms_per_step_wall': 1000 * wall / steps,
                              'ms_per_step_gpu': 1000 * kernels_s / steps, 'kernel_launches_per_step': launches / steps,
                              'top_cpu': top_ops(prof, 'cpu'), 'top_gpu': top_ops(prof, 'gpu')}
            prof.export_chrome_trace(str(out / f'trace_{name}.json'))

    weights_gb = sum(p.numel() * p.element_size() for p in model.base_model.parameters()) / 1e9
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'bench': args.bench,
              'questions': len(prompts), 'target_weights_gb': weights_gb, 'timing': timing,
              'eagle_round_ms': per_round, 'plain_token_ms': per_token, 'profiler': profiles}
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    lines = [f"# Where the time goes: {result['gpu']}, torch {result['torch']}, {args.bench} x {len(prompts)}", '',
             f"EAGLE {timing['eagle_tok_s']:.1f} tok/s, plain {timing['plain_tok_s']:.1f} tok/s, "
             f"speed-up {timing['speedup']:.2f}x, tau {timing['tau']:.2f}; one round = {timing['ms_per_round']:.2f} ms "
             f"= {timing['round_cost_in_plain_steps']:.2f} plain steps ({timing['ms_per_plain_token']:.2f} ms).", '',
             f"Target weights {weights_gb:.1f} GB: reading them once takes ~{weights_gb / 4.8:.1f} ms at 4.8 TB/s.", '',
             '| one EAGLE round | ms |', '|---|---|']
    lines += [f'| {k} | {v:.2f} |' for k, v in per_round.items()]
    lines += ['', '| one plain token | ms |', '|---|---|']
    lines += [f'| {k} | {v:.2f} |' for k, v in per_token.items()]
    lines += ['', '| torch.profiler, 1 question | wall ms/step | GPU ms/step | GPU busy | kernel launches/step |',
              '|---|---|---|---|---|']
    for name, p in profiles.items():
        lines.append(f"| {name} | {p['ms_per_step_wall']:.2f} | {p['ms_per_step_gpu']:.2f} | "
                     f"{100 * p['gpu_busy_share']:.0f}% | {p['kernel_launches_per_step']:.0f} |")
    for name, p in profiles.items():
        for key, title in (('top_cpu', 'CPU'), ('top_gpu', 'GPU')):
            lines += ['', f'Top {title} ops, {name} (ms over the question):', '| op | calls | ms |', '|---|---|---|']
            lines += [f"| {r['name']} | {r['count']} | {r['ms']:.1f} |" for r in p[key]]
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:30]))


if __name__ == '__main__':
    main()
