"""Extra measurements on the saved milestones (every 10,000 steps) of both variants, for the report.

    cd ~/data/big_training && nohup setsid ../nemo_speed/venv_orig/bin/python milestone_analysis.py > milestone_analysis.out 2>&1 &

1. contributions  -> runs/milestone_analysis/contributions.json
   How much each captured target layer feeds the draft's fc. On 64 validation dialogues the target's hidden states
   x_k (input of layer k, as the authors' code captures them) go through fc_norm (ours) and the matching block W_k of
   fc; per milestone we keep the mean norm of W_k x_k over answer tokens, its share of the total, the norm of x_k
   itself and the fc_norm scales.
2. real tau       -> runs/final_eval/<variant>_step<NNNNNNN>/ (final_eval.py's layout)
   SGLang (with the sglang_eagle31 fix) on MT-bench, HumanEval, GSM8K, Alpaca, 80 questions each, for every milestone.
"""
import json
import time

import torch

import final_eval as FE
import settings as S

OUT = S.RUNS / 'milestone_analysis'
VARIANTS = ('baseline', 'ours')
BENCHES = ['mt_bench', 'humaneval', 'gsm8k', 'alpaca']
N_DIALOGUES = 64
MAX_LEN = 1024


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def milestones(variant):
    return sorted(p for p in (S.RUNS / variant / 'milestones').iterdir() if p.name.startswith('step'))


def rms_norm(x, weight, eps):
    x32 = x.float()
    return weight.float() * x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)


@torch.no_grad()
def contributions():
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM
    target = AutoModelForCausalLM.from_pretrained(S.TARGET, torch_dtype=torch.bfloat16).cuda().eval()
    val = load_from_disk(str(S.DATA / 'val')).select(range(N_DIALOGUES))
    layers = sorted({2, 16, 22, 29})
    states = {k: [] for k in layers}                      # answer-token hidden states, input of layer k
    for row in val:
        ids = torch.tensor(row['input_ids'][:MAX_LEN]).cuda()[None]
        mask = torch.tensor(row['loss_mask'][:MAX_LEN]).bool()
        hs = target(ids, output_hidden_states=True).hidden_states
        for k in layers:
            states[k].append(hs[k][0][mask.cuda()].float().cpu())
    states = {k: torch.cat(v) for k, v in states.items()}
    del target
    torch.cuda.empty_cache()
    log(f'hidden states: {len(states[2])} answer tokens from {N_DIALOGUES} dialogues')
    result = {'tokens': len(states[2]), 'state_norm': {k: float(v.norm(dim=-1).mean()) for k, v in states.items()},
              'variants': {}}
    for variant in VARIANTS:
        rows = []
        for m in milestones(variant):
            cfg = json.loads((m / 'config.json').read_text())
            w = torch.load(m / 'pytorch_model.bin', map_location='cpu')
            fc = w['fc.weight'].float().cuda()
            h = fc.shape[0]
            row = {'step': cfg.get('trained_steps', int(m.name[4:])), 'layers': cfg['eagle_capture_layers'],
                   'contrib_norm': [], 'block_frobenius': [], 'fc_norm_scale': []}
            outs = []
            for i, k in enumerate(cfg['eagle_capture_layers']):
                x = states[k].cuda()
                if cfg.get('fc_norm'):
                    g = w[f'fc_norm.{i}.weight']
                    x = rms_norm(x, g.cuda(), cfg['rms_norm_eps'])
                    row['fc_norm_scale'].append({'mean': float(g.float().mean()), 'std': float(g.float().std()),
                                                 'abs_mean': float(g.float().abs().mean())})
                block = fc[:, i * h:(i + 1) * h]
                y = x @ block.T
                outs.append(y)
                row['contrib_norm'].append(float(y.norm(dim=-1).mean()))
                row['block_frobenius'].append(float(block.norm()))
            total = sum(outs)
            sq = [float((o.pow(2).sum(-1)).mean()) for o in outs]
            row['contrib_share'] = [s / sum(sq) for s in sq]
            row['cos_to_total'] = [float(torch.nn.functional.cosine_similarity(o, total, dim=-1).mean()) for o in outs]
            row['total_norm'] = float(total.norm(dim=-1).mean())
            rows.append(row)
            log(f'{variant} {m.name}: share {[round(s, 3) for s in row["contrib_share"]]}')
            del fc, w
        result['variants'][variant] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'contributions.json').write_text(json.dumps(result, indent=1))


def real_tau():
    import export
    for step_dir in [m for m in milestones('baseline') if m.name != 'step0068325']:
        for variant in VARIANTS:
            src = S.RUNS / variant / 'milestones' / step_dir.name
            if not src.exists():
                continue
            out = S.RUNS / 'exports' / f'{variant}_{src.name}'
            if not (out / 'sglang' / 'config.json').exists():
                export.main(src, out)
            name = f'{variant}_{src.name}'
            try:
                FE.run_sglang(name, out / 'sglang', BENCHES, 80, 0.8)
            except Exception as e:  # noqa: BLE001 - one failed milestone must not stop the rest
                log(f'{name}: failed: {e}')


if __name__ == '__main__':
    if not (OUT / 'contributions.json').exists():
        contributions()
    real_tau()
    log('done')
