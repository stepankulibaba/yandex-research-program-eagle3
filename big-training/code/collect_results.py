"""Compact copy of everything the report and the figures need, from a local copy of the server's run directory.

    python code/collect_results.py <copy of ~/data/big_training without weights>   -> results/llama31-8b/

The server copy (logs only: runs/ without checkpoints, milestones' weights and exports; data/stats.json) is a few
tens of MB, mostly answer texts. Kept here:
  <variant>/val.jsonl, metrics.jsonl, status.json   as written by train.py
  <variant>/events.jsonl                            sessions, checkpoints, validations from train.log (timestamps)
  final_eval/<draft>/sglang_<bench>.json            per question: turns, generated tokens, verification rounds,
                                                    accepted draft tokens, seconds, histogram of accepted tokens per
                                                    round (index = accepted draft tokens, 0..8); aggregate tau, speed
  final_eval/<draft>/authors_<bench>.json           the authors' code (aggregates only)
  milestones/contributions.json                     milestone_analysis.py: how much each target layer feeds fc
  data_stats.json                                   the training data (prepare_data.py)
  nemo_incidents/<run>/                             the NeMo runs (metrics, validation)
Drafts: official, baseline, ours, ours_sglang_capture_bug (before the SGLang fix), <variant>_step<N> (milestones),
plain (no draft: the speed-up denominator). Runs on a shared GPU (*_gpu_shared) are left out. Server paths are dropped.
tau = sum(tokens - 1) / sum(rounds) over all turns, as in final_eval.py.
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'llama31-8b'
BENCHES = ['mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'sum']
VARIANTS = ['baseline', 'ours']
NEMO_RUNS = ['ours_bf16_stopped', 'ours_fp32_lr2e-4_diverged', 'ours_nemo_attention_unstable']
EVENT = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj, indent=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent), encoding='utf-8')


def answers(path):
    rows = []
    for q in read(path):
        hist = [0] * 9
        for t in q['turns']:
            for i, n in enumerate((t.get('meta_info') or {}).get('spec_accept_histogram') or []):
                hist[i] += n
        rows.append({'question_id': q['question_id'], 'turns': len(q['turns']),
                     'tokens': sum(t['tokens'] for t in q['turns']),
                     'rounds': sum(t['steps'] or 0 for t in q['turns']),             # 0 for the plain server
                     'accepted': sum(t.get('accepted_draft_tokens') or 0 for t in q['turns']),
                     'seconds': sum(t['seconds'] for t in q['turns']), 'accept_hist': hist})
    return rows


def final_eval(src):
    for d in sorted(p for p in src.iterdir() if p.is_dir() and p.name not in ('logs',)
                    and not p.name.endswith('_gpu_shared')):
        for bench in BENCHES:
            if (d / f'sglang_{bench}.json').exists():
                meta = read(d / f'sglang_{bench}.meta.json')
                write(OUT / 'final_eval' / d.name / f'sglang_{bench}.json',
                      {'engine': 'sglang 0.5.9', 'tree': meta.get('tree'), 'gpu_shared': meta['gpu_shared'],
                       'questions': meta['questions'], 'tau': meta['tau'], 'tok_s': meta['tok_s'],
                       'per_question': answers(d / f'sglang_{bench}.json')}, indent=0)
            if (d / f'authors_{bench}.json').exists():
                a = read(d / f'authors_{bench}.json')
                a.pop('draft', None)
                write(OUT / 'final_eval' / d.name / f'authors_{bench}.json', a)


def events(train_log):
    rows = []
    for line in train_log.read_text(encoding='utf-8', errors='replace').splitlines():
        m = EVENT.match(line)
        if m and not m.group(2).startswith('exported draft weights'):
            rows.append({'time': m.group(1), 'event': m.group(2)})
    return rows


def main(src):
    src = Path(src)
    runs = src / 'runs'
    for v in VARIANTS:
        for name in ('val.jsonl', 'metrics.jsonl', 'status.json'):
            (OUT / v).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runs / v / name, OUT / v / name)
        (OUT / v / 'events.jsonl').write_text(
            ''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in events(runs / v / 'train.log')), encoding='utf-8')
    final_eval(runs / 'final_eval')
    if (runs / 'milestone_analysis' / 'contributions.json').exists():
        (OUT / 'milestones').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runs / 'milestone_analysis' / 'contributions.json', OUT / 'milestones' / 'contributions.json')
    stats = read(src / 'data' / 'stats.json')
    write(OUT / 'data_stats.json', stats)
    for run in NEMO_RUNS:
        for name in ('metrics.jsonl', 'val.jsonl'):
            if (runs / run / name).exists():
                (OUT / 'nemo_incidents' / run).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(runs / run / name, OUT / 'nemo_incidents' / run / name)
    print(f'written to {OUT}')


if __name__ == '__main__':
    main(sys.argv[1])
