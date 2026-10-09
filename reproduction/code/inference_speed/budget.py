"""Task 5: GPU hours to rebuild the paper's training data and train the draft, compared with a budget (100 h).

    python budget.py --probe results/main/regen_probe.json --training ../nemo_speed/artifacts/main/summary.json
                     [--gpu-hours 100] [--overhead-hours 10] [--already-regenerated] [--out results/main/budget.md]

For every measured trainer and number of epochs:
    regeneration h = sum over split and length group: dialogues x measured seconds per dialogue / 3600
    h per epoch    = training tokens per epoch / the trainer's measured tokens/s / 3600
    total h        = reserve + regeneration + epochs x h per epoch
    data share     = which fraction of the corpus fits the budget (regeneration shrinks with it, unless
                     --already-regenerated: then it is paid in full first)
A sensitivity range (x0.75 .. x1.5 of the measured cost) is shown; it is not a confidence interval.

Inputs are pilots: dialogue counts are 68K ShareGPT (the paper's) and the real UltraChat-200K train splits (208K +
256K = the paper's 464K); group weights come from a bounded scan of each split; the regeneration policy is ours
(the paper gives none); trainer speeds were measured on another corpus. Use the "as-is" and packed NeMo rows for
realistic costs: the matched rows pad every example to 2048 tokens on purpose.
"""
import argparse
from pathlib import Path

from common import atomic_json, atomic_text, positive, read_json, validate_speed, verified_artifact
from regen_probe import validate_probe

EPOCHS = (1, 2, 5, 10, 20, 40)      # the authors' trainer is configured for 40


def data_share(budget, regen, epoch_cost, epochs, overhead=0, already_regenerated=False):
    available = max(0.0, budget - overhead)
    if already_regenerated:
        return min(1.0, max(0.0, available - regen) / (epochs * epoch_cost))
    return min(1.0, available / (regen + epochs * epoch_cost))


def estimate(probe, tokens_per_second):
    """(regeneration hours, hours per epoch, training tokens per epoch) for the full corpus.

    Group weights include dialogues the length policy rejects: they cost regeneration time but add no
    training tokens. All assistant turns are regenerated, as in the paper.
    """
    regen_seconds = epoch_tokens = 0.0
    for source in probe['sources'].values():
        for split in source['splits'].values():
            # only the share of dialogues our policy can use at all (ShareGPT: ~63 %: odd turn counts, a model
            # turn first, other roles or empty turns are skipped); the group weights are within that share
            usable = split.get('valid', 1) / split.get('scanned', 1)
            for group in split['groups'].values():
                count = split['dialogues'] * usable * group['weight']
                regen_seconds += count * group['seconds_per_dialogue']
                epoch_tokens += count * group['mean_training_tokens']
    positive(tokens_per_second, 'training throughput')
    return regen_seconds / 3600, epoch_tokens / tokens_per_second / 3600, epoch_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-hours', type=float, default=100)
    parser.add_argument('--overhead-hours', type=float, default=10)
    parser.add_argument('--slow-factor', type=float, default=1.5)
    parser.add_argument('--fast-factor', type=float, default=0.75)
    parser.add_argument('--already-regenerated', action='store_true')
    parser.add_argument('--probe', default='results/main/regen_probe.json')
    parser.add_argument('--training', default='../nemo_speed/artifacts/main/summary.json')
    parser.add_argument('--out', default='results/main/budget.md')
    args = parser.parse_args()
    for name in ('gpu_hours', 'slow_factor', 'fast_factor'):
        positive(getattr(args, name), name)
    positive(args.overhead_hours, 'overhead', zero=True)
    if not args.fast_factor <= 1 <= args.slow_factor:
        raise ValueError('Sensitivity factors must bracket 1')

    verified_artifact(args.probe, validate_probe)
    probe = read_json(args.probe)
    summary = read_json(args.training)
    if not summary['runs']:
        raise ValueError('No measured trainers')

    lines = ['# Preliminary GPU-hour scenarios', '',
             f'**This pilot does not establish that {args.gpu_hours:g} GPU-hours suffice for full paper reproduction.**',
             'Regeneration (our policy; the paper gives none): all assistant turns, greedy, the authors\' system',
             'prompt, <=512 tokens per turn, <=1900 per example; dialogues that do not fit are dropped.',
             f'Sampling: {probe["sampling"]}. Dialogues: ShareGPT 68 623 (the V4.3 file = the paper 68K), UltraChat 208K + 256K, times the share our policy can use.',
             f'Reserve: {args.overhead_hours:g} GPU-hours for evaluation, checkpoints and retries.',
             f'Sensitivity: {args.fast_factor:g}x to {args.slow_factor:g}x of the measured cost; NOT a confidence interval.',
             'Trainer speed: measured on another corpus. Realistic rows: author-as-is and packed NeMo; matched rows',
             'pad every example to 2048 tokens on purpose and overstate the cost.',
             'Regeneration is paid once. --already-regenerated counts it in full before training.', '',
             '| trainer | protocol | regen h | h/epoch | epochs | central total h | sensitivity total h '
             '| data share (central / slow) |',
             '|---|---|---|---|---|---|---|---|']
    rows = []
    for run in summary['runs']:
        result_path = Path(args.training).parent / (run['name'] + '.speed.json')
        verified_artifact(result_path, validate_speed)
        measured = read_json(result_path)
        regen, h_epoch, _ = estimate(probe, measured['nonpad_tokens'] / measured['seconds'])
        if h_epoch <= 0:
            raise ValueError('No retained training tokens')
        for epochs in EPOCHS:
            cost = regen + epochs * h_epoch
            total = args.overhead_hours + cost
            low = args.overhead_hours + args.fast_factor * cost
            high = args.overhead_hours + args.slow_factor * cost
            share = data_share(args.gpu_hours, regen, h_epoch, epochs, args.overhead_hours,
                               args.already_regenerated)
            slow_share = data_share(args.gpu_hours, regen * args.slow_factor, h_epoch * args.slow_factor,
                                    epochs, args.overhead_hours, args.already_regenerated)
            rows.append({'run': run['name'], 'epochs': epochs, 'regen_h': regen, 'epoch_h': h_epoch,
                         'central_h': total, 'sensitivity_h': [low, high], 'data_share': share,
                         'slow_scenario_share': slow_share})
            lines.append(f"| {run['name']} | {measured['protocol']} | {regen:.2f} | {h_epoch:.2f} | {epochs} | {total:.2f} | "
                         f"{low:.2f}–{high:.2f} | {share:.1%} / {slow_share:.1%} |")

    first = summary['runs'][0]
    epoch_tokens = estimate(probe, first['nonpad_tokens'] / first['seconds'])[2]
    lines += ['', f'Caching the target\'s hidden states instead of running it every epoch: '
                  f'{epoch_tokens * 3 * 4096 * 2 / 1e12:.2f} TB (three BF16 states per token, nothing else).',
              'For a firm budget: count lengths/turns over the full corpora and time training on regenerated data.']
    atomic_text(args.out, '\n'.join(lines) + '\n')
    atomic_json(Path(args.out).with_suffix('.json'), {'scope': 'preliminary sensitivity scenarios',
                                                      'budget_h': args.gpu_hours, 'reserve_h': args.overhead_hours,
                                                      'rows': rows})
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
