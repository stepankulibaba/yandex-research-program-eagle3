"""Task 4 table: training step speed of every finished run (the authors' trainer and the NeMo variants).

    python stamp_and_summarize.py summary [artifacts dir, default artifacts/main]   -> summary.txt and summary.json
    ... | python stamp_and_summarize.py stamp                                       prefix lines with the time

Each run's <name>.speed.json comes from training_common.SpeedWindow: exact time and token counts over the
measured optimizer steps. Runs on different data, vocabulary or warm-up are never put in one table.
"""
from pathlib import Path
import sys
import time

from common import SCHEMA, atomic_json, atomic_text, read_json, validate_speed, verified_artifact


def stamp():
    for line in sys.stdin:
        sys.stdout.write(f'{time.time():.3f} {line}')
        sys.stdout.flush()


def summary(directory='artifacts/main'):
    directory = Path(directory)
    runs, rejected = [], {}
    for path in sorted(directory.glob('*.speed.json')):
        try:
            manifest = verified_artifact(path, validate_speed)
            r = read_json(path)
            seconds, steps = r['seconds'], r['optimizer_steps']
            r.update({'name': path.name.removesuffix('.speed.json'), 'manifest': manifest,
                      'optimizer_ms': 1000 * seconds / steps,
                      'nonpad_tokens_per_s': r['nonpad_tokens'] / seconds,
                      'padded_tokens_per_s': r['padded_tokens'] / seconds,
                      'supervised_tokens_per_s': r['supervised_tokens'] / seconds,
                      'documents_per_s': r['documents'] / seconds})
            runs.append(r)
        except (OSError, ValueError, KeyError) as exc:
            rejected[path.name] = str(exc)
    if not runs:
        raise ValueError('No committed complete training runs')
    if len({(r['manifest']['data'], r['manifest']['mapping'], r['warmup_optimizer_steps']) for r in runs}) != 1:
        raise ValueError('Training cohort mismatch')

    lines = ['# Training step speed', '',
             'All runs get the same tokenized examples, masks and 32K vocabulary, TTT=7.',
             'matched: 2048-token rows, eager target, BF16, constant LR, no checkpointing (identical work).',
             'author-as-is: the authors\' trainer with its own settings (fp16, checkpointing, no padding, its lr schedule).',
             'The trainers still differ in loss details, draft RoPE config, optimizer precision and library versions:',
             'this compares the speed of two software stacks, not identical training.',
             'Not timed: preprocessing, evaluation, checkpoints.',
             'Packing changes what a row holds; mb4 makes an optimizer step 4 rows instead of 2.',
             'flash-attn 2 applies to the draft only; the target attention stays eager.', '',
             '| run | protocol | draft attention | batch × accumulation | packing | ms / optimizer step | documents/s '
             '| non-pad tok/s | padded tok/s | supervised tok/s | peak allocated / reserved GiB | measured steps |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in runs:
        c = r['config']
        lines.append(f"| {r['name']} | {r['protocol']} | {c['draft_attention']} | "
                     f"{c['micro_batch']} × {c['accumulation']} | "
                     f"{c['packing']} | {r['optimizer_ms']:.2f} | {r['documents_per_s']:.3f} | "
                     f"{r['nonpad_tokens_per_s']:.1f} | {r['padded_tokens_per_s']:.1f} | "
                     f"{r['supervised_tokens_per_s']:.1f} | {r['peak_allocated_gb']:.2f} / "
                     f"{r['peak_reserved_gb']:.2f} | {r['optimizer_steps']} |")
    atomic_json(directory / 'summary.json', {'schema': SCHEMA, 'runs': runs, 'rejected': rejected,
                                             'scope': 'matched speed-only, not convergence or paper reproduction'})
    atomic_text(directory / 'summary.txt', '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if sys.argv[1:2] == ['stamp']:
        stamp()
    elif sys.argv[1:2] == ['summary']:
        summary(*sys.argv[2:])
    else:
        raise SystemExit('stamp | summary [artifacts dir]')
