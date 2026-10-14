"""The official draft scored by train.py's validation on the same held-out dialogues -> runs/reference.json.

    python reference_eval.py [--data DIR] [--runs DIR] [--val-eval 400]

Gives the level a fully trained baseline reaches with the authors' data and budget, in the same units as the
validation curves (per-position accuracy and the tau estimate), so monitor.py can say how far the runs are from it.
The official draft keeps its own 32K vocabulary (d2t / t2d come from its checkpoint).
"""
import argparse
import json
import os
from pathlib import Path

import settings as S
from train import Examples, build_model, deepspeed_config, tau_estimate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(S.DATA))
    parser.add_argument('--runs', default=str(S.RUNS))
    parser.add_argument('--val-eval', type=int, default=S.VAL_EVAL)
    args = parser.parse_args()
    import torch
    from datasets import load_from_disk
    data = Path(args.data)
    model, _ = build_model('baseline', data, deepspeed_config(1000))
    weights = torch.load(S.OFFICIAL_DRAFT / 'pytorch_model.bin', map_location='cpu')
    result = model.load_state_dict(weights, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [k for k in result.missing_keys if not k.startswith(('target_model.', 'embed_tokens.'))
               and not k.endswith('inv_freq')]
    if missing or unexpected:
        raise ValueError(f'Official draft does not fit the training model: missing {missing}, unexpected {unexpected}')
    model = model.half().cuda().eval()
    val = load_from_disk(str(data / 'val'))
    examples = Examples(val, list(range(min(args.val_eval, len(val)))))
    n, acc, ploss = 0, [0.0] * model.length, [0.0] * model.length
    with torch.no_grad():
        for ex in examples:
            p, _, a = model(input_ids=ex['input_ids'].cuda(), attention_mask=ex['attention_mask'].cuda(),
                            loss_mask=ex['loss_mask'])
            n += 1
            acc = [x + y for x, y in zip(acc, a)]
            ploss = [x + float(y) for x, y in zip(ploss, p)]
    acc, ploss = [x / n for x in acc], [x / n for x in ploss]
    out = {'draft': str(S.OFFICIAL_DRAFT), 'examples': n, 'acc': acc, 'ploss': ploss, 'tau_estimate': tau_estimate(acc)}
    Path(args.runs).mkdir(parents=True, exist_ok=True)
    (Path(args.runs) / 'reference.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out))
    os._exit(0)


if __name__ == '__main__':
    main()
