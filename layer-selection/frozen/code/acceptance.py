"""Measuring tau with real EAGLE generation (temperature 0).

Every verification cycle the target accepts some of the draft's tokens and adds one of its own, so
    tau = (accepted draft tokens + cycles) / cycles.
We hook EAGLE's `evaluate_posterior` to record how many draft tokens were accepted in each cycle.

Comparing two variants (e.g. the official draft and an ablation): once a variant generates a different token,
the texts diverge and later cycles run on different text. `prefix_tau_pair` therefore also reports tau over the
cycles that start before the first divergent token, where both run on identical text.
"""
import time

import numpy as np
import torch


@torch.no_grad()
def measure_acceptance(model, prompts, max_new_tokens=128, reference=None, gen_kwargs=None):
    """Runs EAGLE generation for every prompt.

    Returns (rows, outputs): one row per prompt with the accepted-token counts per cycle, and the generated ids.
    `reference` = outputs of another run (prompt id -> ids): adds where this run's text first differs from it.
    """
    import eagle.model.ea_model as ea_module
    rows, outputs = [], {}
    original = ea_module.evaluate_posterior
    for prompt in prompts:
        accepted = []

        def traced(*args, **kwargs):
            result = original(*args, **kwargs)
            accepted.append(int(result[1]))          # result[1] = number of accepted draft tokens in this cycle
            return result

        ea_module.evaluate_posterior = traced
        try:
            ids = prompt["ids"].to("cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model.eagenerate(ids.clone(), temperature=0.0, max_new_tokens=max_new_tokens, **(gen_kwargs or {}))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        finally:
            ea_module.evaluate_posterior = original
        gen = out[0, ids.shape[1]:].cpu()
        outputs[prompt["id"]] = gen
        row = {"id": prompt["id"], "bench": prompt["bench"], "cycles": len(accepted),
               "accepted_sum": int(sum(accepted)), "new_tokens": int(gen.numel()), "seconds": seconds,
               "tau": float(np.mean([a + 1 for a in accepted])) if accepted else float("nan"),
               "accepted": accepted}
        if reference is not None:
            row.update(_divergence(accepted, reference[prompt["id"]], gen))
        rows.append(row)
    return rows, outputs


def _divergence(accepted, ref, gen):
    """Where this generation first differs from the reference, and the cycles that ran before that point."""
    n = min(len(ref), len(gen))
    diff = (ref[:n] != gen[:n]).nonzero()
    first = int(diff[0]) if len(diff) else (n if len(ref) != len(gen) else None)
    cycles, acc = _cycles_before(accepted, first)
    return {"mismatch": first is not None, "first_mismatch": first, "prefix_cycles": cycles, "prefix_accepted": acc}


def _cycles_before(accepted, first_mismatch):
    """Cycles that start before token `first_mismatch` (position 0 comes from the prefill)."""
    start, cycles, acc = 1, 0, 0
    for a in accepted:
        if first_mismatch is not None and start > first_mismatch:
            break
        cycles += 1
        acc += a
        start += a + 1
    return cycles, acc


def tau_of(rows):
    """Pooled tau of a set of rows: all accepted tokens and cycles summed first."""
    cycles = sum(r["cycles"] for r in rows)
    return (sum(r["accepted_sum"] for r in rows) + cycles) / max(cycles, 1)


def tau_summary(rows):
    out = {"tau": tau_of(rows), "cycles": sum(r["cycles"] for r in rows),
           "new_tokens": sum(r["new_tokens"] for r in rows), "seconds": sum(r["seconds"] for r in rows)}
    for bench in sorted({r["bench"] for r in rows}):
        out[f"tau_{bench}"] = tau_of([r for r in rows if r["bench"] == bench])
    if rows and "mismatch" in rows[0]:
        out["mismatched_prompts"] = int(sum(r["mismatch"] for r in rows))
        pc = sum(r["prefix_cycles"] for r in rows)
        out["tau_prefix"] = (sum(r["prefix_accepted"] for r in rows) + pc) / max(pc, 1)
        out["prefix_cycles"] = pc
    return out


def prefix_tau_pair(ref_rows, rows):
    """tau of the reference and of a variant over the same identical-text prefix of every prompt."""
    ref = {r["id"]: r for r in ref_rows}
    ref_cycles = ref_acc = var_cycles = var_acc = 0
    for r in rows:
        cycles, acc = _cycles_before(ref[r["id"]]["accepted"], r.get("first_mismatch"))
        ref_cycles, ref_acc = ref_cycles + cycles, ref_acc + acc
        var_cycles, var_acc = var_cycles + r["prefix_cycles"], var_acc + r["prefix_accepted"]
    return (ref_acc + ref_cycles) / max(ref_cycles, 1), (var_acc + var_cycles) / max(var_cycles, 1)


def paired_bootstrap(rows_a, rows_b, n=2000, seed=0):
    """95 % interval of tau(b) - tau(a), resampling prompts (the same prompts for both)."""
    a = {r["id"]: r for r in rows_a}
    b = {r["id"]: r for r in rows_b}
    ids = [r["id"] for r in rows_b if r["id"] in a]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.choice(len(ids), len(ids), replace=True)
        diffs.append(tau_of([b[ids[i]] for i in pick]) - tau_of([a[ids[i]] for i in pick]))
    return [float(x) for x in np.quantile(diffs, [0.025, 0.975])]
