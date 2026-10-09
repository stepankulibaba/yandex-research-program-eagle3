"""Dimensionally checked pilot scenarios, with explicit reserve and cache policy."""
import argparse
from pathlib import Path
from runtime import atomic_json, atomic_text, positive, read_json, validate_speed, verified_artifact
from regen_probe import validate_probe

EPOCHS = (1, 2, 5, 10, 20, 40)


def data_share(budget, regen, epoch_cost, epochs, overhead=0, already_regenerated=False):
    available = max(0.0, budget - overhead)
    if already_regenerated:
        return min(1.0, max(0.0, available - regen) / (epochs * epoch_cost))
    return min(1.0, available / (regen + epochs * epoch_cost))


def estimate(probe, run):
    # Weights include policy-rejected examples: they cost regeneration time but
    # contribute zero retained training tokens. Multiple assistant turns are measured.
    regen_seconds, epoch_tokens = 0.0, 0.0
    for source in probe['sources'].values():
        population = source['paper_assumed_dialogues']
        for group in source['groups'].values():
            count = population * group['weight']
            regen_seconds += count * group['seconds_per_dialogue']
            epoch_tokens += count * group['mean_training_tokens']
    positive(run['nonpad_tokens_per_s'], 'training throughput')
    return regen_seconds / 3600, epoch_tokens / run['nonpad_tokens_per_s'] / 3600, epoch_tokens


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
    rows = []
    lines = ['# Preliminary GPU-hour scenarios', '',
        '**This pilot does not establish that 100 GPU-hours suffice for full paper reproduction.**',
        f'Regeneration: both corpora, measured multi-turn answers, shared system prompt and 1900-token retention policy.',
        f'Sampling: {probe["sampling"]}. Source counts (68K/464K) are paper assumptions, not measured retained sizes.',
        f'Operational reserve: {args.overhead_hours:g} GPU-hours; adjust it for evaluation/checkpoint/retry costs.',
        f'Sensitivity: {args.fast_factor:g}× to {args.slow_factor:g}× measured cost; this is NOT a confidence interval.',
        'Training uses the adapted BF16 throughput recipe and another corpus; padding/length/packing effects remain a transfer assumption.',
        'One-off target regeneration is counted once. Already-regenerated mode treats its full cost as sunk in the same budget.', '',
        '| trainer | regen h | h/epoch | epochs | central total h | sensitivity total h | data share (central / slow) |',
        '|---|---|---|---|---|---|---|']
    for run in summary['runs']:
        artifact = Path(args.training).parent / (run['name'] + '.speed.json')
        verified_artifact(artifact, validate_speed)
        actual = read_json(artifact)
        run = {**actual, 'name': run['name'],
               'nonpad_tokens_per_s': actual['nonpad_tokens'] / actual['seconds']}
        regen, h_epoch, tokens = estimate(probe, run)
        if h_epoch <= 0:
            raise ValueError('No retained training tokens')
        for e in EPOCHS:
            total = args.overhead_hours + regen + e * h_epoch
            low = args.overhead_hours + args.fast_factor * (regen + e * h_epoch)
            high = args.overhead_hours + args.slow_factor * (regen + e * h_epoch)
            share = data_share(args.gpu_hours, regen, h_epoch, e, args.overhead_hours, args.already_regenerated)
            conservative = data_share(args.gpu_hours, regen * args.slow_factor, h_epoch * args.slow_factor,
                                      e, args.overhead_hours, args.already_regenerated)
            rows.append({'run': run['name'], 'epochs': e, 'regen_h': regen, 'epoch_h': h_epoch,
                         'central_h': total, 'sensitivity_h': [low, high], 'data_share': share,
                         'slow_scenario_share': conservative})
            lines.append(f"| {run['name']} | {regen:.2f} | {h_epoch:.2f} | {e} | {total:.2f} | "
                         f"{low:.2f}–{high:.2f} | {share:.1%} / {conservative:.1%} |")
    tokens = estimate(probe, summary['runs'][0])[2]
    lines += ['', f'Hidden-state cache only: {tokens * 3 * 4096 * 2 / 1e12:.2f} TB for three BF16 states/token.',
              'This excludes target probability/logit supervision, vocab maps, embeddings, metadata and I/O.',
              'Required follow-up for a firm budget: full-corpus length/turn census and throughput validation on regenerated samples.']
    atomic_text(args.out, '\n'.join(lines) + '\n')
    atomic_json(Path(args.out).with_suffix('.json'), {'scope': 'preliminary sensitivity scenarios',
        'budget_h': args.gpu_hours, 'reserve_h': args.overhead_hours, 'rows': rows})
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
