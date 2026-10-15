"""What the fc correction (fixfc.py) did, from the per-prompt logs only (no GPU).

    py -3 analysis_fix.py [results_dir]

Greedy lossless decoding: with and without the correction the edited target writes the SAME text, so every prompt is
a paired observation. Per edited target:
  A  paired bootstrap CI (over prompts) of the pooled tau change on the English domains, share of prompts that improve
  B  where the gain comes from: distribution of accepted lengths per cycle (0..depth), early vs late part of the answer
  C  speed-up implied by tau (cost per cycle unchanged: the correction only adds a bias to fc)
"""
import json
import sys
from pathlib import Path

import numpy as np

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results" / "llama31-8b"
EN = ["general", "math", "code", "medicine", "tool"]
rng = np.random.default_rng(0)


def load(name):
    return json.loads((DIR / name).read_text(encoding="utf-8"))


def pooled(rows):
    c = sum(r["cycles"] for r in rows)
    return (sum(r["accepted_sum"] for r in rows) + c) / c


for e in ("nemotron_nano", "r1_distill", "toolace", "abliterated"):
    o, f = load(f"{e}__inst.json"), load(f"{e}__inst_fix.json")
    ro = {r["id"]: r for r in o["rows"]}
    rf = {r["id"]: r for r in f["rows"]}
    ids = [i for i in ro if ro[i]["domain"] in EN]
    same_text = np.mean([ro[i]["answer"] == rf[i]["answer"] for i in ids])
    po, pf = pooled([ro[i] for i in ids]), pooled([rf[i] for i in ids])
    boots = []
    for _ in range(4000):
        s = rng.choice(ids, len(ids))
        boots.append(pooled([rf[i] for i in s]) / pooled([ro[i] for i in s]) - 1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    per = np.array([rf[i]["tau"] / ro[i]["tau"] - 1 for i in ids])
    print(f"\n=== {e}: identical text {same_text:.2f} | English tau {po:.3f} -> {pf:.3f} ({(pf / po - 1) * 100:+.1f}%, "
          f"95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]) | prompts improved {np.mean(per > 0):.2f}, worse {np.mean(per < 0):.2f}")
    # B1: accepted-length histogram per cycle
    def hist(rows):
        acc = np.concatenate([np.array(r["accepted"]) for r in rows])
        return np.bincount(acc, minlength=8)[:8] / len(acc)
    ho, hf = hist([ro[i] for i in ids]), hist([rf[i] for i in ids])
    print("   accepted draft tokens per cycle 0..7 (share of cycles):")
    print("     without", " ".join(f"{v:.3f}" for v in ho))
    print("     with   ", " ".join(f"{v:.3f}" for v in hf))
    # B2: early vs late part of the answer (by cycle index quartile)
    parts = []
    for q in range(4):
        so = sf = co = cf = 0
        for i in ids:
            a, b = ro[i]["accepted"], rf[i]["accepted"]
            na, nb = len(a), len(b)
            sa = a[q * na // 4:(q + 1) * na // 4]; sb = b[q * nb // 4:(q + 1) * nb // 4]
            so += sum(sa) + len(sa); co += len(sa); sf += sum(sb) + len(sb); cf += len(sb)
        parts.append((so / max(co, 1), sf / max(cf, 1)))
    print("   tau by quarter of the answer (cycles):", "  ".join(f"Q{q + 1} {a:.2f}->{b:.2f} ({(b / a - 1) * 100:+.0f}%)"
                                                            for q, (a, b) in enumerate(parts)))
    # B3: per domain
    print("   per domain:", "  ".join(f"{d} {pooled([ro[i] for i in ids if ro[i]['domain'] == d]):.2f}->"
                                     f"{pooled([rf[i] for i in ids if rf[i]['domain'] == d]):.2f}" for d in EN))
    # C: implied speed-up: speed-up = tau / (cycle cost in plain steps); cycle cost from the map run
    s = o["speed"]
    cycle_cost = pooled([ro[i] for i in ids]) / s["speedup"] if s["speedup"] else float("nan")
    print(f"   measured speed-up (12 prompts) {o['speed']['speedup']:.2f} -> {f['speed']['speedup']:.2f}; "
          f"implied by English tau at constant cycle cost: {po / cycle_cost:.2f} -> {pf / cycle_cost:.2f}")
