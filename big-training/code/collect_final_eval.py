"""Compact copy of the final evaluation (final_eval.py output) for the repository.

    python code/collect_final_eval.py <runs/final_eval copied from the server>   -> results/llama31-8b/final_eval/

final_eval.py keeps every answer (texts, token ids) — several MB per draft. Here only what the numbers need is kept:
per question and benchmark the generated tokens, verification rounds, accepted draft tokens and turns (SGLang), the
answer time, plus the aggregate tau and speed; the authors' code gives aggregates only. Server paths are dropped.
tau = sum(tokens - 1) / sum(rounds) over all turns, as in final_eval.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'llama31-8b' / 'final_eval'
DRAFTS = ['official', 'baseline', 'ours', 'ours_sglang_capture_bug', 'plain']
BENCHES = ['mt_bench', 'humaneval', 'gsm8k', 'alpaca', 'sum']


def main(src):
    src = Path(src)
    for draft in DRAFTS:
        d = OUT / draft
        d.mkdir(parents=True, exist_ok=True)
        for bench in BENCHES:
            answers = src / draft / f'sglang_{bench}.json'
            if answers.exists():
                meta = json.loads((src / draft / f'sglang_{bench}.meta.json').read_text(encoding='utf-8'))
                rows = [{'question_id': q['question_id'], 'turns': len(q['turns']),
                         'tokens': sum(t['tokens'] for t in q['turns']),
                         'rounds': sum(t['steps'] or 0 for t in q['turns']),           # 0 for the plain server
                         'accepted': sum(t.get('accepted_draft_tokens') or 0 for t in q['turns']),
                         'seconds': sum(t['seconds'] for t in q['turns'])}
                        for q in json.loads(answers.read_text(encoding='utf-8'))]
                out = {'engine': 'sglang 0.5.9', 'tree': meta.get('tree'), 'gpu_shared': meta['gpu_shared'],
                       'questions': meta['questions'], 'tau': meta['tau'], 'tok_s': meta['tok_s'], 'per_question': rows}
                (d / f'sglang_{bench}.json').write_text(json.dumps(out, indent=0), encoding='utf-8')
            authors = src / draft / f'authors_{bench}.json'
            if authors.exists():
                a = json.loads(authors.read_text(encoding='utf-8'))
                a.pop('draft', None)
                (d / f'authors_{bench}.json').write_text(json.dumps(a, indent=1), encoding='utf-8')
    print(f'written to {OUT}')


if __name__ == '__main__':
    main(sys.argv[1])
