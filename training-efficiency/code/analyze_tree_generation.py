"""Recompute the October 6 generation result, with grouped paired bootstrap."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results/qwen3-1.7b'

def analyze():
    rows = json.loads((RESULTS / 'tree_generation_measurements.json').read_text())
    checks = json.loads((RESULTS / 'tree_generation_exact_pairs.json').read_text())
    saved = json.loads((RESULTS / 'tree_generation_analysis.json').read_text())
    metadata = json.loads((RESULTS / 'tree_generation_metadata.json').read_text())
    assert len(rows) == 192 and len(checks) == 96
    indexed = {(r['rep'], r['prompt_id'], r['mode']): r for r in rows}
    assert len(indexed) == 192
    for check in checks:
        assert check['tokens_exact'] and check['cycles_exact']
        assert check['upstream_tokens_sha256'] == check['candidate_tokens_sha256']
        a = indexed[check['rep'], check['prompt_id'], 'upstream']
        b = indexed[check['rep'], check['prompt_id'], 'tensor_cpu']
        for key in ['input_tokens', 'output_tokens', 'cycles', 'group_id', 'language']:
            assert a[key] == b[key]
        assert a['order_index'] != b['order_index']
        for row in [a, b]:
            assert row['sec'] > 0 and row['cycles'] > 0
            assert row['tau'] == row['output_tokens'] / row['cycles']
    groups = sorted({r['group_id'] for r in rows})
    prompts = {r['prompt_id'] for r in rows}
    assert len(groups) == 24 and len(prompts) == 48
    for group in groups:
        current = [r for r in rows if r['group_id'] == group]
        assert len(current) == 8
        assert {r['language'] for r in current} == {'en', 'ru'}
        assert {r['rep'] for r in current} == {0, 1}
    totals = {mode: sum(r['sec'] for r in rows if r['mode'] == mode) for mode in ['upstream', 'tensor_cpu']}
    ratio = totals['upstream'] / totals['tensor_cpu']
    group_totals = np.array([[sum(r['sec'] for r in rows if r['group_id'] == group and r['mode'] == mode)
                             for mode in ['upstream', 'tensor_cpu']] for group in groups])
    rng = np.random.default_rng(20261006)
    samples = group_totals[rng.integers(0, len(groups), size=(10000, len(groups)))].sum(axis=1)
    ci = np.quantile(samples[:, 0] / samples[:, 1], [.025, .975]).tolist()
    assert np.allclose([ratio, *ci], [saved['heldout_speedup'], *saved['heldout_group_bootstrap_95_interval']], atol=1e-12, rtol=1e-12)
    assert all(group_totals[:, 0] > group_totals[:, 1])
    tokens = {mode: sum(r['output_tokens'] for r in rows if r['mode'] == mode) for mode in totals}
    cycles = {mode: sum(r['cycles'] for r in rows if r['mode'] == mode) for mode in totals}
    assert set(tokens.values()) == {12486} and set(cycles.values()) == {6746}
    assert metadata['config']['stopping'].startswith('EOS disabled')
    return {'ratio': ratio, 'ci95': ci, 'time_reduction_percent': (1 - 1 / ratio) * 100,
            'seconds': totals, 'tokens_per_variant': 12486, 'cycles_per_variant': 6746,
            'pairs_checked': 96, 'groups': 24, 'prompts': 48}

if __name__ == '__main__':
    print(json.dumps(analyze(), indent=2))
