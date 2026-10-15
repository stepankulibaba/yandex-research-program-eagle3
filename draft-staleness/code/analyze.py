"""Tables of the staleness map from results/llama31-8b/: tau per domain vs the original target, decomposition, and how well
the shift of the captured hidden states predicts the loss.

    py -3 analyze.py [results_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results" / "llama31-8b"
DOMAINS = ["general", "math", "code", "medicine", "tool", "ru"]


def load(name):
    p = DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


base = load("base__inst.json")
runs = sorted(DIR.glob("*__*.json"), key=lambda p: p.stat().st_mtime)
print("tree tau per domain (change vs original target) | speedup")
print("target".ljust(22) + "".join(d.rjust(16) for d in DOMAINS) + "   speed")
points = []
for p in runs:
    r = json.loads(p.read_text(encoding="utf-8"))
    cells = []
    for d in DOMAINS:
        t = r["domains"][d]["tau"]
        b = base["domains"][d]["tau"]
        cells.append(f"{t:5.2f} ({(t / b - 1) * 100:+4.0f}%)".rjust(16) if r["target"] != "base" or r["draft"] != "inst" else f"{t:5.2f}".rjust(16))
    print(f"{r['target']}+{r['draft']}".ljust(22) + "".join(cells) + f"   {r['speed']['speedup']:.2f}")
    dec = load(f"decomp_{r['target']}.json")
    if dec and r["draft"] == "inst":
        en = [d for d in DOMAINS if d != "ru"]
        loss = float(np.mean([1 - r["domains"][d]["tau"] / base["domains"][d]["tau"] for d in en]))
        points.append((r["target"], float(np.mean(dec["all"]["feature_shift"])), loss, dec["all"]["free_transfer"]))

print("\ndecomposition (chain tau, same text): T0 own text > T1 text with T0 features > T1 text with T1 features")
for p in sorted(DIR.glob("decomp_*.json"), key=lambda p: p.stat().st_mtime):
    r = json.loads(p.read_text(encoding="utf-8"))
    a = r["all"]
    print(f"{r['target']:14s} {a['tau_base']:.2f} > {a['tau_old_features']:.2f} > {a['tau_new_features']:.2f}  "
          f"free {a['free_transfer']:+.2f}  shift {[round(s, 2) for s in a['feature_shift']]}  oov-stops {a['oov_stop_share']:.2f}")

if len(points) >= 3:
    x = np.array([p[1] for p in points])
    y = np.array([p[2] for p in points])
    rank = lambda v: np.argsort(np.argsort(v))
    print(f"\nmean feature shift vs mean tau loss on 5 English domains (n={len(points)}): "
          f"pearson {np.corrcoef(x, y)[0, 1]:.2f}, spearman {np.corrcoef(rank(x), rank(y))[0, 1]:.2f}")
    for name, s, l, f in sorted(points, key=lambda p: p[1]):
        print(f"  {name:14s} shift {s:.3f}  loss {l * 100:5.1f}%  free {f:+.2f}")
