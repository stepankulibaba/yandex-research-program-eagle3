"""Real tau of a draft: the authors' own evaluation script (tree 60 / depth 8 / top-k 10), on our patched copy.

    python eval_tau.py <draft dir> <name> [--bench mt_bench] [--questions 20] [--temperature 0]

<draft dir>: a milestone of train.py (config.json + pytorch_model.bin) or the official draft. Our drafts carry
eagle_capture_layers / fc_norm in config.json; the patched ea_model.py then uses those target layers and refuses
weights that do not load completely. Runs eagle.evaluation.gen_ea_answer_llama3chat with the eval environment
(venv_eagle of the reproduction) and prints tau = generated tokens / drafting rounds, as in the reproduction.
"""
import argparse
import json
import subprocess
from pathlib import Path

import settings as S


def tau_of(answer_file):
    rows = [json.loads(line) for line in Path(answer_file).read_text(encoding='utf-8').splitlines() if line.strip()]
    choices = [r['choices'][0] for r in rows]
    tokens = sum(sum(c['new_tokens']) for c in choices)
    rounds = sum(sum(i + 1 for i in c['idxs']) for c in choices)
    seconds = sum(sum(c['wall_time']) for c in choices)
    return {'questions': len(rows), 'tau': tokens / rounds, 'tokens_per_s': tokens / seconds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('draft')
    parser.add_argument('name')
    parser.add_argument('--bench', default='mt_bench')
    parser.add_argument('--questions', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0)
    args = parser.parse_args()
    out = S.RUNS / 'eval' / f'{args.name}_{args.bench}_t{args.temperature:g}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)                 # the script appends
    cmd = [str(S.EVAL_PYTHON), '-m', 'eagle.evaluation.gen_ea_answer_llama3chat',
           '--ea-model-path', str(Path(args.draft).resolve()), '--base-model-path', str(S.TARGET),
           '--model-id', args.name, '--bench-name', args.bench, '--question-begin', '0',
           '--question-end', str(args.questions), '--temperature', str(args.temperature),
           '--total-token', '60', '--depth', '7', '--top-k', '10', '--use_eagle3', '--answer-file', str(out)]
    subprocess.run(cmd, cwd=S.WORK / 'eval', check=True)
    result = {'draft': str(args.draft), 'bench': args.bench, 'temperature': args.temperature, **tau_of(out)}
    (out.with_suffix('.json')).write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
