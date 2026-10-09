"""Summarize exact speed-window counters; never infer samples from log cadence."""
import json
from pathlib import Path
import sys
import time
from runtime import atomic_json, atomic_text, read_json, validate_speed, verified_artifact


def stamp():
    for line in sys.stdin:
        sys.stdout.write(f'{time.time():.3f} {line}')
        sys.stdout.flush()


def summary(directory='artifacts/main'):
    directory = Path(directory)
    rows, rejected = [], {}
    for p in sorted(directory.glob('*.speed.json')):
        try:
            manifest = verified_artifact(p, validate_speed)
            r = read_json(p)
            dt, steps = r['seconds'], r['optimizer_steps']
            r.update({'name': p.name.removesuffix('.speed.json'), 'manifest': manifest,
                'optimizer_ms': 1000 * dt / steps, 'nonpad_tokens_per_s': r['nonpad_tokens'] / dt,
                'padded_tokens_per_s': r['padded_tokens'] / dt,
                'supervised_tokens_per_s': r['supervised_tokens'] / dt,
                'documents_per_s': r['documents'] / dt})
            rows.append(r)
        except (OSError, ValueError, KeyError) as exc:
            rejected[p.name] = str(exc)
    if not rows:
        raise ValueError('No committed complete training runs')
    # Refuse to pool different corpora, warm-ups, or mappings into a comparison.
    common = {(r['manifest']['data'], r['manifest']['mapping'], r['warmup_optimizer_steps']) for r in rows}
    if len(common) != 1:
        raise ValueError('Training cohort mismatch')
    lines = ['# Adapted BF16 training throughput', '',
        'Both adapters consume the same author-tokenized input IDs, masks and 32K mapping.',
        'Common: fixed 2048 rows, eager target, BF16, constant LR, TTT=7, no activation checkpointing.',
        'This is a speed-only protocol; it excludes preprocessing, evaluation and checkpoint writes.',
        'Packing changes row composition; mb4 changes effective optimizer batch from 2 to 4 rows.',
        'FA2 labels apply to draft attention. Target attention remains eager.', '',
        '| run | actual draft attention | batch × accumulation | packing | ms/optimizer step | documents/s | nonpad tok/s | padded tok/s | supervised tok/s | peak allocated / reserved GiB | measured steps |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        c = r['config']
        lines.append(f"| {r['name']} | {c['draft_attention']} | {c['micro_batch']} × {c['accumulation']} | "
                     f"{c['packing']} | {r['optimizer_ms']:.2f} | {r['documents_per_s']:.3f} | "
                     f"{r['nonpad_tokens_per_s']:.1f} | {r['padded_tokens_per_s']:.1f} | "
                     f"{r['supervised_tokens_per_s']:.1f} | {r['peak_allocated_gb']:.2f} / "
                     f"{r['peak_reserved_gb']:.2f} | {r['optimizer_steps']} |")
    atomic_json(directory / 'summary.json', {'schema': 3, 'runs': rows, 'rejected': rejected,
                                            'scope': 'adapted speed-only, not convergence or paper reproduction'})
    atomic_text(directory / 'summary.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if sys.argv[1] == 'stamp':
        stamp()
    elif sys.argv[1] == 'summary':
        summary(*sys.argv[2:])
    else:
        raise SystemExit('stamp | summary [artifact_directory]')
