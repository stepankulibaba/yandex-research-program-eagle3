"""Exact greedy *chain* metrics and paired, prompt-level uncertainty.

This module does not generate tokens and does not validate target-model replay.
The caller must provide each target's own greedy continuation, already trimmed
at EOS or its generation budget. Teacher-forced predictions are valid until the
first mismatch; predictions following a mismatch never count as accepted.

These are single-chain metrics, not the tau of EAGLE's branching tree. No warmup
time, GPU throughput, or wall-clock speed is inferred from these measurements.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def score_chain(pred: np.ndarray, ids: np.ndarray, prompt_len: int) -> dict:
    """Traverse verification cycles, including short tails and an EOS token.

    ``ids`` has shape [T] and includes the prompt and its greedy continuation.
    ``pred`` has shape [K, T]. Its element ``pred[s - 1, t]`` predicts
    ``ids[t + s + 1]`` for depth s >= 1. The known target/root token at a cycle
    start is ``ids[t + 1]``. Start at ``t = prompt_len - 1``; after accepting L
    draft proposals, the next start is ``t + 1 + L``.

    The root counts toward tokens per cycle, but never toward draft acceptance.
    Depth opportunities are conditional: a depth is reached only after all
    previous proposals were accepted AND its target token exists. Consequently
    short tails do not enter denominators at unavailable depths.

    Returned values are JSON serializable. An empty continuation has zero
    cycles/tokens and ``tau=None``; it is not a valid row for bootstrap.
    """
    pred, ids = np.asarray(pred), np.asarray(ids)
    prompt_len = _integer(prompt_len, "prompt_len")
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("ids must be a one-dimensional integer array")
    if pred.ndim != 2 or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("pred must be a two-dimensional integer array")
    if pred.shape[0] < 1 or pred.shape[1] != ids.size:
        raise ValueError("pred must have shape [K, len(ids)] with K >= 1")
    if not 1 <= prompt_len <= ids.size:
        raise ValueError("prompt_len must be between 1 and len(ids)")

    depth, total = pred.shape
    accepted = np.zeros(depth, dtype=np.int64)
    opportunities = np.zeros(depth, dtype=np.int64)
    cycle_lengths, cycle_starts = [], []
    t = prompt_len - 1
    while t + 1 < total:
        # The target/root token is always known, including a final EOS root.
        length = 1
        cycle_starts.append(t)
        for k in range(depth):
            target_pos = t + k + 2
            if target_pos >= total:
                break
            opportunities[k] += 1
            if pred[k, t] != ids[target_pos]:
                break
            accepted[k] += 1
            length += 1
        cycle_lengths.append(length)
        t += length

    tokens, cycles = sum(cycle_lengths), len(cycle_lengths)
    assert tokens == total - prompt_len, "every completion token must count once"
    assert int(accepted.sum()) == tokens - cycles, "roots are not draft proposals"
    return {
        "tokens": int(tokens),
        "cycles": cycles,
        "tau": tokens / cycles if cycles else None,
        "accepted_draft_tokens": int(accepted.sum()),
        "accepted_by_depth": accepted.tolist(),
        "opportunities_by_depth": opportunities.tolist(),
        "conditional_acceptance_by_depth": [
            int(a) / int(n) if n else None for a, n in zip(accepted, opportunities)
        ],
        "cycle_lengths": cycle_lengths,
        "cycle_starts": cycle_starts,
    }


def pooled_tau(rows: Iterable[Mapping]) -> float:
    """Return sum(completion tokens)/sum(verification cycles) for given rows.

    Call separately per training seed. Pooling differently sized seed samples
    gives seeds different weights, unlike the seed mean used by paired_bootstrap.
    """
    tokens = cycles = 0
    for row in rows:
        n = _integer(row["tokens"], "tokens")
        c = _integer(row["cycles"], "cycles")
        if n < 0 or c < 0 or c > n or (c == 0) != (n == 0):
            raise ValueError("rows require 0 <= cycles <= tokens; both zero together")
        tokens += n
        cycles += c
    if cycles == 0:
        raise ValueError("tau is undefined without verification cycles")
    return tokens / cycles


def _index_rows(rows: Sequence[Mapping], label: str) -> tuple[dict, dict, list]:
    if not rows:
        raise ValueError(f"{label}: no rows supplied")
    indexed, domains, edits = {}, {}, set()
    for row in rows:
        if "prompt_id" not in row:
            raise ValueError(f"{label}: each row needs prompt_id")
        prompt = str(row["prompt_id"])
        seed = _integer(row.get("seed", 0), "seed")
        domain = str(row.get("domain", "all"))
        tokens = _integer(row["tokens"], "tokens")
        cycles = _integer(row["cycles"], "cycles")
        if not 1 <= cycles <= tokens:
            raise ValueError(f"{label}: bootstrap rows need 1 <= cycles <= tokens")
        key = (seed, prompt)
        if key in indexed:
            raise ValueError(f"{label}: duplicate row for seed={seed}, prompt={prompt}")
        if prompt in domains and domains[prompt] != domain:
            raise ValueError(f"{label}: prompt {prompt} has inconsistent domains")
        domains[prompt] = domain
        indexed[key] = (tokens, cycles)
        if "edit" in row:
            edits.add(str(row["edit"]))
    if len(edits) > 1:
        raise ValueError(f"{label}: filter rows to one target edit before bootstrap")
    seeds = sorted({key[0] for key in indexed})
    expected = {(seed, prompt) for seed in seeds for prompt in domains}
    if set(indexed) != expected:
        raise ValueError(f"{label}: every seed must have the same complete prompt set")
    return indexed, domains, seeds


def paired_bootstrap(
    rows_base: Sequence[Mapping],
    rows_candidate: Sequence[Mapping],
    *,
    seed: int = 12345,
    n: int = 2000,
    confidence: float = 0.95,
) -> dict:
    """Stratified paired bootstrap of the difference in mean-seed pooled tau.

    Rows contain ``prompt_id, domain, seed, tokens, cycles``; ``edit`` and
    ``method`` can be present. Compare ONE target edit and ONE method per side.
    Resample whole prompts, with replacement within each domain, preserving its
    observed prompt count. Use the SAME resampled prompts for both methods and
    all training seeds. Compute pooled tau within each seed, then average those
    ratios. Never bootstrap overlapping token positions as independent samples.

    Seed IDs must match, unless one method has exactly one deterministic seed;
    that single row set is then broadcast across the other method's seeds. This
    is appropriate for an unchanged deterministic baseline versus refit seeds.
    It does not estimate uncertainty over training seeds: the CI is conditional
    on the observed trained models. Seed variability is reported separately.
    """
    n, seed = _integer(n, "n"), _integer(seed, "seed")
    if n < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("n >= 1 and 0 < confidence < 1 are required")
    base, domains_b, seeds_b = _index_rows(rows_base, "baseline")
    candidate, domains_c, seeds_c = _index_rows(rows_candidate, "candidate")
    if domains_b != domains_c:
        raise ValueError("baseline and candidate must have identical prompts/domains")
    edits_b = {str(r["edit"]) for r in rows_base if "edit" in r}
    edits_c = {str(r["edit"]) for r in rows_candidate if "edit" in r}
    if edits_b and edits_c and edits_b != edits_c:
        raise ValueError("compare methods on the same target edit")
    for rows, label in ((rows_base, "baseline"), (rows_candidate, "candidate")):
        if len({str(r["method"]) for r in rows if "method" in r}) > 1:
            raise ValueError(f"{label}: filter rows to one method before bootstrap")
    if seeds_b == seeds_c:
        pairs = list(zip(seeds_b, seeds_c))
    elif len(seeds_b) == 1:
        pairs = [(seeds_b[0], s) for s in seeds_c]
    elif len(seeds_c) == 1:
        pairs = [(s, seeds_c[0]) for s in seeds_b]
    else:
        raise ValueError("training seed sets must match, or one side must have one seed")

    prompts = sorted(domains_b)
    arrays_b = np.asarray([[base[(s_b, p)] for p in prompts] for s_b, _ in pairs], dtype=np.float64)
    arrays_c = np.asarray([[candidate[(s_c, p)] for p in prompts] for _, s_c in pairs], dtype=np.float64)
    strata = defaultdict(list)
    for idx, prompt in enumerate(prompts):
        strata[domains_b[prompt]].append(idx)

    def ratios(arrays: np.ndarray, indices: np.ndarray) -> np.ndarray:
        totals = arrays[:, indices, :].sum(axis=1)
        return totals[:, 0] / totals[:, 1]

    all_indices = np.arange(len(prompts))
    base_ratios = ratios(arrays_b, all_indices)
    candidate_ratios = ratios(arrays_c, all_indices)
    seed_deltas = candidate_ratios - base_ratios
    rng = np.random.default_rng(seed)
    sampled_delta = np.empty(n, dtype=np.float64)
    for i in range(n):
        indices = np.concatenate([
            rng.choice(group, size=len(group), replace=True)
            for _, group in sorted(strata.items())
        ])
        sampled_delta[i] = np.mean(ratios(arrays_c, indices) - ratios(arrays_b, indices))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(sampled_delta, [tail, 1.0 - tail])
    return {
        "base_tau": float(base_ratios.mean()),
        "candidate_tau": float(candidate_ratios.mean()),
        "delta_tau": float(seed_deltas.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "bootstrap_repeats": n,
        "bootstrap_seed": seed,
        "n_prompts": len(prompts),
        "domains": {key: len(value) for key, value in sorted(strata.items())},
        "seed_pairs": [[a, b] for a, b in pairs],
        "delta_tau_by_seed": seed_deltas.tolist(),
        "seed_delta_std": float(seed_deltas.std(ddof=1)) if len(seed_deltas) > 1 else None,
        "ci_scope": "prompt uncertainty conditional on the observed training seeds",
    }
