"""Results table for 3a and 3b: the paper vs the authors' code vs SGLang.

    python summarize.py <model dir> [results dir, default results/main]  -> <results dir>/summary.md and .json

Only finished results (with an intact "done" marker) enter the table.

Authors' code (3a), computed like the authors' evaluation/speed.py:
    speed-up = mean over questions of EAGLE tokens/s  /  mean over questions of plain tokens/s
               (plain tokens counted by re-tokenizing the answer, as speed.py does)
    tau      = generated tokens / verification rounds, with idx + 1 rounds per turn
SGLang (3b), same questions over HTTP:
    speed-up = mean tokens/s with the EAGLE-3 server / mean tokens/s with the plain server, same seed
    tau      = (generated tokens - 1) / verification rounds, per turn summed: SGLang emits the first token at
               prefill, then each round emits its accepted draft tokens + 1, exactly as a round of the authors'
               code. Diagnostics: SGLang's accepted-draft counter + 1 (it can include a tail cut at a stop token)
               and generated tokens / rounds (it includes the prefill token).
"""
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

from common import BENCHES, atomic_json, atomic_text, questions, read_json, validate_answers, verified_artifact

# EAGLE-3 paper, Table 1, LLaMA-Instruct 3.1 8B: (speed-up, tau) per benchmark, in BENCHES order
PAPER = {'0': [(4.40, 6.13), (4.85, 6.74), (4.48, 6.23), (4.82, 6.70), (3.65, 5.34)],
         '1': [(3.07, 4.24), (4.13, 5.82), (3.32, 4.59), (3.90, 5.56), (2.99, 4.39)]}
SGLANG_TREES_T0 = ('paper_8_10_60', 'chain_3_1_4', 'docs_5_8_32', 'eagle2_6_10_60')
SGLANG_TREES_T1 = ('paper_8_10_60',)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def eagle_repo(ea_path, base_path, tok, expected=None):
    """3a numbers from the authors' answer files (EAGLE-3 and plain)."""
    ea_rows, base_rows = load_jsonl(ea_path), load_jsonl(base_path)
    if expected is None:
        expected = [{'question_id': r['question_id'], 'turns': r['choices'][0]['turns']} for r in base_rows]
    validate_answers(ea_path, expected, 'eagle')
    validate_answers(base_path, expected, 'eagle')
    ea = [r['choices'][0] for r in ea_rows]
    base = [r['choices'][0] for r in base_rows]
    ea_speed = [sum(c['new_tokens']) / sum(c['wall_time']) for c in ea]
    base_speed = [sum(len(tok(t).input_ids) - 1 for t in c['turns']) / sum(c['wall_time']) for c in base]
    if mean(base_speed) <= 0:
        raise ValueError('Zero baseline speed')
    tau = sum(sum(c['new_tokens']) for c in ea) / sum(sum(i + 1 for i in c['idxs']) for c in ea)
    return {'base_tok_s': mean(base_speed), 'eagle_tok_s': mean(ea_speed),
            'speedup': mean(ea_speed) / mean(base_speed), 'tau': tau,
            'questions': len(ea), 'protocol': 'unchanged-author-cuda-generation'}


def sglang_speed(path):
    """Speed and acceptance counters of one SGLang answer file."""
    rows = read_json(path)
    if not rows:
        raise ValueError('Empty SGLang benchmark')
    turns = [t for r in rows for t in r['turns']]
    speeds = [sum(t['tokens'] for t in r['turns']) / sum(t['seconds'] for t in r['turns']) for r in rows]
    steps = [t.get('steps') for t in turns]
    accepted = [t.get('accepted_draft_tokens') for t in turns]
    rounds = sum(steps) if all(s is not None for s in steps) else None
    tau = sum(t['tokens'] - 1 for t in turns) / rounds if rounds else None
    tokens_per_round = sum(t['tokens'] for t in turns) / rounds if rounds else None
    accepted_per_round = ((sum(accepted) / rounds + 1)
                          if rounds and all(a is not None for a in accepted) else None)
    return {'tok_s': mean(speeds), 'tau': tau, 'completion_over_verify': tokens_per_round,
            'verification_tokens_per_round': accepted_per_round, 'verification_rounds': rounds,
            'zero_round_turns': sum(s == 0 for s in steps), 'questions': len(rows), 'timing': 'http_round_trip'}


def author_cell(results, prefix, bench, expected, tok):
    ea = results / 'eagle' / f'ea_{prefix}{bench}.jsonl'
    base = results / 'eagle' / f'base_{prefix}{bench}.jsonl'
    ma = verified_artifact(ea, lambda p: validate_answers(p, expected, 'eagle'))
    mb = verified_artifact(base, lambda p: validate_answers(p, expected, 'eagle'))
    if ma['temperature'] != mb['temperature'] or ma['environment'] != mb['environment']:
        raise ValueError('Author pair provenance mismatch')
    return eagle_repo(ea, base, tok, expected)


def sglang_cell(results, config, suffix, bench, expected, seed):
    spec_path = results / 'sglang' / f'{config}{suffix}_{bench}.json'
    base_path = results / 'sglang' / f'base{suffix}_{bench}.json'
    ms = verified_artifact(spec_path, lambda p: validate_answers(p, expected, 'sglang'))
    mb = verified_artifact(base_path, lambda p: validate_answers(p, expected, 'sglang'))
    if ms['seed'] != mb['seed'] or ms['temperature'] != mb['temperature'] or ms['environment'] != mb['environment']:
        raise ValueError('SGLang pair provenance mismatch')
    spec, baseline = sglang_speed(spec_path), sglang_speed(base_path)
    spec.update({'seed': seed, 'speedup': spec['tok_s'] / baseline['tok_s']})
    return spec


def fmt(value):
    return f'{value:.3f}' if value is not None else '—'


def main(model_dir, output='results/main'):
    from transformers import AutoTokenizer
    results = Path(output)
    tok = AutoTokenizer.from_pretrained(model_dir)
    repo = Path(__file__).parent / 'EAGLE'
    count = 2 if results.name == 'smoke' else 80
    summary = {}
    lines = ['# Inference results', '',
             'Authors\' code: unchanged official scripts, their speed-up and tau. Tree: 60 tokens, --depth 7',
             '(draft length 8), top-k 10 — depth 8 of the EAGLE-3 paper.',
             'SGLang: HTTP round-trip time, at most 512 new / 1979 total tokens per turn, no prefix cache.',
             'SGLang tau = (generated - 1) / rounds, the same count as the authors\' tau. Compare speed-ups,',
             'not absolute tokens/s, between the two (HTTP vs CUDA timer).',
             'paper_8_10_60 has the same budget as the authors\' tree (not necessarily the same candidates);',
             'eagle2_6_10_60 is the EAGLE-2 tree.', '']
    for temperature in ('0', '1'):
        lines += [f'## T={temperature}', '',
                  '| benchmark | paper speed-up / τ | authors\' code speed-up / τ | SGLang config | SGLang speed-up '
                  '| SGLang τ | accepted + 1 (diagnostic) | T=1 seeds: speed-up mean ± sd |',
                  '|---|---|---|---|---|---|---|---|']
        prefix = '' if temperature == '0' else 't1_'
        configs = SGLANG_TREES_T0 if temperature == '0' else SGLANG_TREES_T1
        seeds = (0, 1, 2) if temperature == '1' and count == 80 else (0,)
        per_bench = {}
        for index, bench in enumerate(BENCHES):
            expected = questions(repo, bench, count)
            paper_speedup, paper_tau = PAPER[temperature][index]
            r = {'paper_speedup': paper_speedup, 'paper_tau': paper_tau, 'status': {}, 'sglang': {}}
            try:
                r['author'] = author_cell(results, prefix, bench, expected, tok)
            except (OSError, ValueError, KeyError) as exc:
                r['status']['author'] = f'incomplete: {exc}'
            author = r.get('author')
            author_text = f"{author['speedup']:.2f} / {author['tau']:.2f}" if author else 'incomplete'
            for config in configs:
                repeats = []
                for seed in seeds:
                    suffix = ('_t1' if temperature == '1' else '') + (f'_seed{seed}' if seed else '')
                    try:
                        repeats.append(sglang_cell(results, config, suffix, bench, expected, seed))
                    except (OSError, ValueError, KeyError) as exc:
                        r['status'][f'{config}_seed{seed}'] = f'incomplete: {exc}'
                if repeats:
                    r['sglang'][config] = {'repeats': repeats, 'seed_count': len(repeats),
                                           'speedup_mean': mean(v['speedup'] for v in repeats),
                                           'speedup_sd': pstdev(v['speedup'] for v in repeats)}
                first = next((x for x in repeats if x['seed'] == 0), None)
                spread = (f"{r['sglang'][config]['speedup_mean']:.3f} ± {r['sglang'][config]['speedup_sd']:.3f} "
                          f"(n={len(repeats)})" if len(repeats) > 1 else '—')
                lines.append(f"| {bench} | {paper_speedup:.2f} / {paper_tau:.2f} | {author_text} | {config} | "
                             f"{fmt(first['speedup']) if first else 'incomplete'} | "
                             f"{fmt(first['tau']) if first else '—'} | "
                             f"{fmt(first['verification_tokens_per_round']) if first else '—'} | {spread} |")
            per_bench[bench] = r
        summary['T=' + temperature] = per_bench
        lines.append('')
    atomic_json(results / 'summary.json', summary)
    atomic_text(results / 'summary.md', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(*sys.argv[1:])
