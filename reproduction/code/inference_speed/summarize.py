"""Only committed complete cohorts enter the summary. Author and HTTP protocols are separate."""
import json
from pathlib import Path
from statistics import mean, pstdev
import sys
from runtime import (BENCHES, atomic_json, atomic_text, questions, read_json, validate_answers, verified_artifact)

PAPER = {'0': [(4.40, 6.13), (4.85, 6.74), (4.48, 6.23), (4.82, 6.70), (3.65, 5.34)],
         '1': [(3.07, 4.24), (4.13, 5.82), (3.32, 4.59), (3.90, 5.56), (2.99, 4.39)]}


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def eagle_repo(ea_path, base_path, tok, expected=None):
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
    return {'base_tok_s': mean(base_speed), 'eagle_tok_s': mean(ea_speed),
            'speedup': mean(ea_speed) / mean(base_speed),
            'tau': sum(sum(c['new_tokens']) for c in ea) / sum(sum(i + 1 for i in c['idxs']) for c in ea),
            'questions': len(ea), 'protocol': 'unchanged-author-cuda-generation'}


def sglang_speed(path):
    rows = read_json(path)
    if not rows:
        raise ValueError('Empty SGLang benchmark')
    turns = [t for r in rows for t in r['turns']]
    speeds = [sum(t['tokens'] for t in r['turns']) / sum(t['seconds'] for t in r['turns']) for r in rows]
    steps = [t.get('steps') for t in turns]
    accepted = [t.get('accepted_draft_tokens') for t in turns]
    rounds = sum(steps) if all(s is not None for s in steps) else None
    raw = sum(t['tokens'] for t in turns) / rounds if rounds else None
    tau_verify = (sum(accepted) / rounds + 1) if rounds and all(a is not None for a in accepted) else None
    return {'tok_s': mean(speeds), 'completion_over_verify': raw,
            'verification_tokens_per_round': tau_verify, 'verification_rounds': rounds,
            'zero_round_turns': sum(s == 0 for s in steps), 'questions': len(rows), 'timing': 'http_round_trip'}


def main(model_dir, output='results/main'):
    from transformers import AutoTokenizer
    output = Path(output)
    tok = AutoTokenizer.from_pretrained(model_dir)
    repo = Path(__file__).parent / 'EAGLE'
    result = {}
    lines = ['# Inference results', '',
             'Author speed-up/τ follow the unchanged official code. SGLang uses HTTP round-trip latency,',
             'a hard 512-new-token / 1979-total-token budget and no prefix reuse.',
             'Its verification tokens/round and completion/round are separate counters;',
             'Author tree: 60 tokens, --depth 7 (draft length 8), top-k 10, as in the EAGLE-3 paper (depth 8);', 'SGLang paper_8_10_60 is the same budget (not an assertion of identical candidate trees); eagle2_6_10_60 is the EAGLE-2 tree.', '']
    count = 2 if output.name == 'smoke' else 80
    for temp in ('0', '1'):
        lines += [f'## T={temp}', '', '| benchmark | paper speed / τ | author speed / τ | SGLang config | HTTP speed-up | verify tokens/round | completion/round | T=1 repeat speed mean ± sd |',
                  '|---|---|---|---|---|---|---|---|']
        data = {}
        prefix = '' if temp == '0' else 't1_'
        configs = ('paper_8_10_60', 'chain_3_1_4', 'docs_5_8_32', 'eagle2_6_10_60') if temp == '0' else ('paper_8_10_60',)
        for index, bench in enumerate(BENCHES):
            expected = questions(repo, bench, count)
            r = {'paper_speedup': PAPER[temp][index][0], 'paper_tau': PAPER[temp][index][1], 'status': {}}
            ea = output / 'eagle' / f'ea_{prefix}{bench}.jsonl'
            base = output / 'eagle' / f'base_{prefix}{bench}.jsonl'
            try:
                ma = verified_artifact(ea, lambda p: validate_answers(p, expected, 'eagle'))
                mb = verified_artifact(base, lambda p: validate_answers(p, expected, 'eagle'))
                if ma['temperature'] != mb['temperature'] or ma['environment'] != mb['environment']:
                    raise ValueError('Author pair provenance mismatch')
                r['author'] = eagle_repo(ea, base, tok, expected)
            except (OSError, ValueError, KeyError) as exc:
                r['status']['author'] = f'incomplete: {exc}'
            r['sglang'] = {}
            for cfg in configs:
                repeats = []
                for seed in ((0, 1, 2) if temp == '1' and count == 80 else (0,)):
                    suffix = ('_t1' if temp == '1' else '') + (f'_seed{seed}' if seed else '')
                    a = output / 'sglang' / f'{cfg}{suffix}_{bench}.json'
                    b = output / 'sglang' / f'base{suffix}_{bench}.json'
                    try:
                        ma = verified_artifact(a, lambda p: validate_answers(p, expected, 'sglang'))
                        mb = verified_artifact(b, lambda p: validate_answers(p, expected, 'sglang'))
                        if ma['seed'] != mb['seed'] or ma['temperature'] != mb['temperature'] or ma['environment'] != mb['environment']:
                            raise ValueError('SGLang pair provenance mismatch')
                        spec, baseline = sglang_speed(a), sglang_speed(b)
                        spec.update({'seed': seed, 'speedup': spec['tok_s'] / baseline['tok_s']})
                        repeats.append(spec)
                    except (OSError, ValueError, KeyError) as exc:
                        r['status'][cfg + f'_seed{seed}'] = f'incomplete: {exc}'
                if repeats:
                    r['sglang'][cfg] = {'repeats': repeats, 'seed_count': len(repeats),
                        'speedup_mean': mean(v['speedup'] for v in repeats),
                        'speedup_sd': pstdev(v['speedup'] for v in repeats)}
                psp, ptau = PAPER[temp][index]
                author = r.get('author')
                atext = f"{author['speedup']:.2f} / {author['tau']:.2f}" if author else 'incomplete'
                value = next((x for x in repeats if x['seed'] == 0), None)
                def fmt(v):
                    return f'{v:.3f}' if v is not None else '—'
                variation = (f"{r['sglang'][cfg]['speedup_mean']:.3f} ± {r['sglang'][cfg]['speedup_sd']:.3f} (n={len(repeats)})"
                             if len(repeats) > 1 else '—')
                lines.append(f"| {bench} | {psp:.2f} / {ptau:.2f} | {atext} | {cfg} | "
                             f"{fmt(value['speedup']) if value else 'incomplete'} | "
                             f"{fmt(value['verification_tokens_per_round']) if value else '—'} | "
                             f"{fmt(value['completion_over_verify']) if value else '—'} | {variation} |")
            data[bench] = r
        result['T=' + temp] = data
        lines.append('')
    atomic_json(output / 'summary.json', result)
    atomic_text(output / 'summary.md', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(*sys.argv[1:])
