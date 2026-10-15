"""How the PARAMETERS of the edited targets change, and whether that predicts the draft's tau loss / the state shift.

    py -3 analysis_weights.py [results_dir]

Uses geo1b_<edit>.json (relative change and concentration of every matrix) and geo1b_<edit>.npz (top-8 left singular
vectors of the change of o_proj / down_proj = directions the edit writes into the residual stream), geo1_<edit>.npz
(mean state shift at the captured layers 2, 16, 29), and the staleness map for the tau loss.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results" / "llama31-8b"
EN = ["general", "math", "code", "medicine", "tool"]
ORDER = ["limo_lora", "toolace", "supernova", "vikhr", "ultramedical", "huatuo_o1", "abliterated", "r1_distill",
         "nemotron_nano"]
H = 4096


def load(n):
    return json.loads((DIR / n).read_text(encoding="utf-8"))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rk = lambda v: np.argsort(np.argsort(v))
    return np.corrcoef(x, y)[0, 1], np.corrcoef(rk(x), rk(y))[0, 1]


base = load("base__inst.json")
edits = [e for e in ORDER if (DIR / f"geo1b_{e}.json").exists()]
g1 = {e: load(f"geo1_{e}.json") for e in edits}
g1b = {e: load(f"geo1b_{e}.json") for e in edits}
loss = np.array([np.mean([1 - load(f"{e}__inst.json")["domains"][d]["tau"] / base["domains"][d]["tau"] for d in EN])
                 for e in edits])
shift = np.array([np.mean(g1[e]["shift_rel"]) for e in edits])


def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


print("1. How much do the weights move (relative Frobenius change, median over matrices of each kind)")
kinds = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head", "norm"]
print(f"{'edit':14s}" + "".join(f"{k[:8]:>9s}" for k in kinds) + "   tau loss")
agg = {}
for e in edits:
    rel = g1b[e]["weights_rel"]
    row = []
    for k in kinds:
        vals = [v for n, v in rel.items() if k in n and (k != "norm" or "layernorm" in n or n.endswith("norm.weight"))]
        row.append(np.median(vals) if vals else np.nan)
    agg[e] = row
    print(f"{e:14s}" + "".join(f"{v:9.4f}" for v in row) + f"   {loss[edits.index(e)] * 100:5.1f}%")

print("\n2. Weight-only predictors of the tau loss (no forward pass, no text)")
cands = {}
for e in edits:
    rel = g1b[e]["weights_rel"]
    mats = {n: v for n, v in rel.items() if layer_of(n) is not None and n.endswith("proj.weight")}
    cands.setdefault("mean rel change, all layer matrices", []).append(np.mean(list(mats.values())))
    cands.setdefault("mean rel change, layers 0-28 (up to the last captured)", []).append(
        np.mean([v for n, v in mats.items() if layer_of(n) <= 28]))
    cands.setdefault("mean rel change, residual writers (o, down)", []).append(
        np.mean([v for n, v in mats.items() if "o_proj" in n or "down_proj" in n]))
    cands.setdefault("embed_tokens rel change", []).append(rel.get("model.embed_tokens.weight", np.nan))
for name, x in cands.items():
    p, s = corr(x, loss)
    ps, ss = corr(x, shift)
    print(f"  {name:55s} vs tau loss: pearson {p:5.2f} spearman {s:5.2f} | vs state shift: pearson {ps:5.2f}")

print("\n3. How concentrated are the weight changes (energy of top-1 / top-8 / top-32 singular directions, median)")
for e in edits:
    top = g1b[e]["weights_top_energy"]
    arr = np.array(list(top.values()))
    w = np.array([v for n, v in top.items() if "o_proj" in n or "down_proj" in n])
    print(f"  {e:14s} all: {np.median(arr[:, 0]):.3f} / {np.median(arr[:, 1]):.3f} / {np.median(arr[:, 2]):.3f}   "
          f"residual writers: {np.median(w[:, 0]):.3f} / {np.median(w[:, 1]):.3f} / {np.median(w[:, 2]):.3f}")

print("\n4. Do the weights write into the residual stream along the directions the states actually move?")
print("   share of the mean state shift (layers 16 and 29) inside the span of the top-8 left singular vectors of the")
print("   o_proj/down_proj changes of all earlier layers, vs a random subspace of the same dimension")
rng = np.random.default_rng(0)
for e in edits:
    if not (DIR / f"geo1b_{e}.npz").exists():
        print("  (needs the full geometry1b npz files, ~34 MB each, not in git; numbers: results/llama31-8b/analysis/analysis_weights_v1.txt)")
        break
    z = np.load(DIR / f"geo1b_{e}.npz")
    mu = np.load(DIR / f"geo1_{e}.npz")["mu"].astype(np.float64)
    out = []
    for j, layer in ((1, 16), (2, 29)):
        vecs = [z[k].astype(np.float64) for k in z.files if k.startswith("left::") and (layer_of(k) or 99) < layer]
        if not vecs:
            continue
        U = np.concatenate(vecs, 1)
        q, _ = np.linalg.qr(U)
        m = mu[j * H:(j + 1) * H]
        share = float(np.sum((q.T @ m) ** 2) / np.sum(m ** 2))
        r, _ = np.linalg.qr(rng.standard_normal((H, q.shape[1])))
        rnd = float(np.sum((r.T @ m) ** 2) / np.sum(m ** 2))
        out.append(f"layer {layer}: {share:.3f} (random {rnd:.3f}, dims {q.shape[1]})")
    print(f"  {e:14s} " + "   ".join(out))
