"""Smoke test verdict: what worked, the measured speed, and what to do next.  python smoke_report.py"""
import json
from pathlib import Path

import settings as S

RUNS = S.HERE / 'runs_smoke'


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def main():
    ok, lines = True, []

    def check(condition, text):
        nonlocal ok
        ok &= bool(condition)
        lines.append(('OK   ' if condition else 'FAIL ') + text)

    for v in S.VARIANTS:
        st = read(RUNS / v / 'status.json') or {}
        check(st.get('phase') == 'done' and st.get('step') == 150, f'{v}: finished 150 steps (phase {st.get("phase")})')
        log = (RUNS / v / 'train.log').read_text(encoding='utf-8', errors='replace') if (RUNS / v / 'train.log').exists() else ''
        if v == 'baseline':
            check('resumed from' in log, 'baseline: interrupted and resumed from its checkpoint')
        check(st.get('nonfinite', 1) == 0, f'{v}: no non-finite losses ({st.get("nonfinite")})')
        vals = [json.loads(x) for x in (RUNS / v / 'val.jsonl').read_text().splitlines()] if (RUNS / v / 'val.jsonl').exists() else []
        if vals:
            check(vals[-1]['tau_estimate'] > vals[0]['tau_estimate'],
                  f'{v}: validation tau estimate grew {vals[0]["tau_estimate"]:.3f} -> {vals[-1]["tau_estimate"]:.3f}')
        rows = [json.loads(x) for x in (RUNS / v / 'metrics.jsonl').read_text().splitlines()] if (RUNS / v / 'metrics.jsonl').exists() else []
        if rows:
            speed = sorted(r['samples_per_s'] for r in rows[len(rows) // 3:])
            lines.append(f'     {v}: {speed[len(speed) // 2]:.2f} dialogues/s, '
                         f'{sorted(r["tokens_per_s"] for r in rows)[len(rows) // 2]:,.0f} tok/s (both running together), '
                         f'GPU memory {st.get("gpu_memory_gb", 0):.0f} GB')
    ref = read(RUNS / 'reference.json')
    check(ref is not None, f'official draft scored on validation: tau estimate {ref and round(ref["tau_estimate"], 3)}')
    for name in ('smoke_baseline', 'smoke_ours', 'smoke_official'):
        r = read(S.RUNS / 'eval' / f'{name}_mt_bench_t0.json')
        good = r is not None and r['tau'] >= (5 if name == 'smoke_official' else 1)
        check(good, f'real tau {name}: {r and round(r["tau"], 2)} ({"expect ~6" if name == "smoke_official" else "expect >= 1"})')
    lines.append('')
    lines.append('ALL GOOD: python plan.py "YYYY-MM-DD HH:MM", then bash session.sh in each session' if ok
                 else 'SOMETHING FAILED: send me smoke.out and runs_smoke/*/train.log')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
