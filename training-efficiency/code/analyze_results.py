"""Recompute published paired statistics and budgets from compact real counters.

No CUDA, torch, model weights, original dataset text, or private machine access.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results/qwen3-1.7b'
SEED = 20261008
RESAMPLES = 20000
CONDITIONS = ['baseline', 'train24hi', 'train24lo', 'train96hi', 'train96lo']

def read(name):
    return json.loads((RESULTS / name).read_text())

def quantile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

def effect(ratios, primary=False, seed=SEED):
    rng = random.Random(seed)
    logs = [math.log(value) for value in ratios]
    samples = [math.exp(sum(logs[rng.randrange(len(logs))] for _ in logs) / len(logs)) for _ in range(RESAMPLES)]
    alpha = .05 / 3 if primary else .05
    interval = [quantile(samples, alpha / 2), quantile(samples, 1 - alpha / 2)]
    better = sum(value > 1 for value in ratios)
    return {'geometric_ratio': math.exp(statistics.mean(logs)), 'bootstrap_interval': interval,
            'confidence': 1 - alpha, 'better': better,
            'worse': sum(value < 1 for value in ratios), 'ties': sum(value == 1 for value in ratios),
            'positive_signal': primary and interval[0] > 1 and better >= 24, 'ratios': ratios}

def assert_close(actual, expected):
    if isinstance(expected, dict):
        for key in expected:
            assert key in actual, key
            assert_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_close(a, b)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)
    else:
        assert actual == expected, (actual, expected)

def verify_export_integrity():
    manifest = json.loads((ROOT / 'code/export_manifest.json').read_text())
    for name, info in manifest['source_files'].items():
        assert hashlib.sha256((ROOT / 'code/original' / name).read_bytes()).hexdigest() == info['export_sha256'], name
    for name, info in manifest['result_files'].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == info['export_sha256'], name
    return len(manifest['source_files']) + len(manifest['result_files'])

def qualified_repeats(rows, keys):
    groups = {}
    for row in rows:
        assert row['cycles'] > 0 and 0 < row['useful_tokens'] <= 128
        assert row['tau_fixed'] == row['useful_tokens'] / row['cycles']
        assert row['draft_accepted'] == row['useful_tokens'] - row['cycles']
        identity = tuple(row[k] for k in keys)
        assert row['repeat'] not in groups.setdefault(identity, {})
        groups[identity][row['repeat']] = row
    for pair in groups.values():
        assert set(pair) == {0, 1}
        for key in ['cycles', 'useful_tokens', 'tau_fixed', 'draft_accepted', 'trace_sha256']:
            assert pair[0][key] == pair[1][key], key
    return [pair[0] for pair in groups.values()], len(groups)

def factorial():
    saved = read('factorial_analysis.json')
    ordered_ids = [r['id'] for r in saved['primary']['queries']]
    assert len(ordered_ids) == len(set(ordered_ids)) == 32
    maps = {}
    repeats = 0
    for condition in CONDITIONS:
        rows = read(f'factorial_{condition}_replay.json')
        assert len(rows) == 80
        unique, count = qualified_repeats(rows, ['id'])
        assert count == 40
        repeats += count
        maps[condition] = {r['id']: r for r in unique if r['split'] == 'validation'}
        assert set(maps[condition]) == set(ordered_ids)
    conditions = {}
    for condition in CONDITIONS:
        current = maps[condition]
        ratios = [current[i]['tau_fixed'] / maps['baseline'][i]['tau_fixed'] for i in ordered_ids]
        cycles = sum(r['cycles'] for r in current.values())
        conditions[condition] = {'aggregate_tau': sum(r['useful_tokens'] for r in current.values()) / cycles,
                                 'relative_initial': effect(ratios), 'cycles_sum': cycles}
    primary = {}
    values = {'H27': [], 'H28': [], 'H29': [], 'interaction': []}
    for i in ordered_ids:
        t = {key: maps[key][i]['tau_fixed'] for key in CONDITIONS}
        values['H27'].append(math.sqrt((t['train24lo'] / t['train24hi']) * (t['train96lo'] / t['train96hi'])))
        values['H28'].append(math.sqrt((t['train96hi'] / t['train24hi']) * (t['train96lo'] / t['train24lo'])))
        values['H29'].append(t['train96lo'] / t['baseline'])
        values['interaction'].append((t['train96lo'] / t['train96hi']) / (t['train24lo'] / t['train24hi']))
    for key, ratios in values.items():
        primary[key] = effect(ratios, primary=key != 'interaction')
        assert_close(primary[key], saved['primary'][key])
    assert_close(conditions, saved['primary']['conditions'])
    assert repeats == saved['exact_repeat_checks'] == 200
    total = 0
    for condition in CONDITIONS[1:]:
        updates = read(f'factorial_{condition}_updates.json')
        assert len(updates) == (384 if condition.startswith('train24') else 381)
        assert [row['update'] for row in updates] == list(range(1, len(updates) + 1))
        assert all(row['finite'] and math.isfinite(row['loss']) for row in updates)
        tokens = sum(row['input_tokens'] for row in updates)
        assert tokens == 79712
        total += tokens
    assert total == saved['all_input_tokens'] == 318848
    return {'primary': primary, 'conditions': conditions, 'input_tokens': total, 'exact_repeats': repeats}

def learning_curve():
    saved = read('learning_curve_analysis.json')
    unique, repeats = qualified_repeats(read('learning_curve_replay.json'), ['id', 'update', 'teacher'])
    assert repeats == 160
    curve = {}
    for update in [0, 24, 96, 384]:
        curve[str(update)] = {}
        for split, count in [('train', 24), ('validation', 8)]:
            rows = [r for r in unique if r['update'] == update and r['teacher'] == 'full_teacher_cache' and r['split'] == split]
            assert len(rows) == count
            tau = sum(r['useful_tokens'] for r in rows) / sum(r['cycles'] for r in rows)
            assert_close(tau, saved['summaries'][f'{update}/{split}/full_teacher_cache']['aggregate_tau'])
            curve[str(update)][split] = tau
    assert saved['H25'] is True and saved['H26'] is False
    return curve

def memory():
    saved = read('memory_analysis.json')
    rows = read('memory_measurements.json')
    assert len(rows) == 96 and saved['executions'] == '96/96'
    cfg = read('memory_protocol_before_probe.json')
    for expected in saved['timing_pairs']:
        ratios = []
        for entry in expected['entries']:
            def get(mode):
                found = [r for r in rows if r['kind'] == 'timing' and r['length'] == expected['length'] and r['order'] == entry['block'] and r['mode'] == mode]
                assert len(found) == 1
                return found[0]['ms']
            baseline = get(expected['baseline'])
            candidate = get(expected['candidate'])
            assert_close(baseline, entry['baseline_ms']); assert_close(candidate, entry['candidate_ms'])
            ratios.append(baseline / candidate)
        assert_close(math.exp(statistics.mean(map(math.log, ratios))), expected['geometric_ratio'])
        assert expected['predeclared_engineering_speedup_passed'] == (expected['geometric_ratio'] >= 1.02 and expected['technical_pair_ci95'][0] > 1)
    maxima = {}
    for mode in ['dense', 'split', 'stride']:
        passing = []
        for length in cfg['capacity_lengths']:
            found = [r for r in rows if r['kind'] == 'capacity' and r['length'] == length and r['mode'] == mode]
            assert len(found) == 2
            if all(r['outcome'] == 'complete' and r['updates_done'] == 3 and r['weights_restored'] for r in found):
                passing.append(length)
        maxima[mode] = max(passing)
        assert maxima[mode] == next(r['maximum_passing_grid_length'] for r in saved['capacity'] if r['mode'] == mode)
    checks = read('memory_checks.json')
    trajectory = [r for r in checks if r['kind'] == 'trajectory']
    assert len(trajectory) == 48 and all(r['passed'] and r['fingerprints_exact'] for r in trajectory)
    assert sum(r['loss_hash_exact'] for r in trajectory) == saved['trajectory_loss_hashes_exact'] == 41
    assert saved['prediction']['hypothesis_passed'] is False
    return {'maximum_passing_grid': maxima, 'trajectory_exact': 48, 'loss_hash_exact': 41, 'prediction_passed': False}

def analyze():
    integrity = verify_export_integrity()
    result = {'integrity_files': integrity, 'factorial': factorial(), 'learning_curve': learning_curve(), 'memory': memory()}
    gate = read('generation_correctness.json')
    assert gate['checks_done'] == 16 and gate['checks_passed'] == 14 and not gate['timings_qualified']
    result['generation_correctness'] = '14/16; generation timing not qualified'
    return result

if __name__ == '__main__':
    result = analyze()
    print('Export integrity, paired intervals, token budgets, exact repeats, learning curve and memory counters: PASS')
    for condition, row in result['factorial']['conditions'].items():
        print(f'{condition}: {(row["relative_initial"]["geometric_ratio"] - 1) * 100:+.4f}%')
    print('Fine-tuning generation equality: 14/16; no fine-tuned generation speed claim.')
