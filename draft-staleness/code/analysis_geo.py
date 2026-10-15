"""Offline analysis of the staleness map + geometry step 1 (no GPU).

    py -3 analysis_geo.py [results_dir]   -> prints a report; figures to logs/fig_*.png

Q1  predictor of the tau loss: bootstrap CI over edits, leave-one-edit-out prediction error, which captured layer
    carries the signal, per-domain points
Q2  geometry of the state shift across edits: cosine of the mean shift, overlap of the top shift subspaces
    (mean squared cosine of principal angles), against random subspaces of the same size
Q3  geometry of "what the draft must learn": overlap of the rank-32 output subspaces of the logit residual
"""
import json
import sys
from pathlib import Path

import numpy as np

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results" / "llama31-8b"
FIG = Path(__file__).resolve().parent.parent / "figures"
EN = ["general", "math", "code", "medicine", "tool"]
ORDER = ["limo_lora", "toolace", "supernova", "vikhr", "ultramedical", "huatuo_o1", "abliterated", "r1_distill",
         "nemotron_nano"]
H = 4096
rng = np.random.default_rng(0)


def load(name):
    p = DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


base = load("base__inst.json")
edits = [e for e in ORDER if (DIR / f"geo1_{e}.json").exists()]
maps = {e: load(f"{e}__inst.json") for e in edits}
geo = {e: load(f"geo1_{e}.json") for e in edits}
dec = {e: load(f"decomp_{e}.json") for e in edits}
npz = {e: np.load(DIR / f"geo1_{e}.npz") for e in edits}


def loss(e, domains=EN):
    return float(np.mean([1 - maps[e]["domains"][d]["tau"] / base["domains"][d]["tau"] for d in domains]))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rk = lambda v: np.argsort(np.argsort(v))
    return np.corrcoef(x, y)[0, 1], np.corrcoef(rk(x), rk(y))[0, 1]


print(f"edits with geometry: {len(edits)}: {edits}\n")
# ------------------------------------------------------------------------------------------------ Q1
print("Q1. Does the shift of the captured states predict the tau loss?")
y = np.array([loss(e) for e in edits])
preds = {
    "shift mean of 3 layers": [np.mean(geo[e]["shift_rel"]) for e in edits],
    "shift layer 2": [geo[e]["shift_rel"][0] for e in edits],
    "shift layer 16": [geo[e]["shift_rel"][1] for e in edits],
    "shift layer 29": [geo[e]["shift_rel"][2] for e in edits],
    "shift through W_fc": [geo[e]["shift_fc_rel"] for e in edits],
    "target argmax change": [geo[e]["target_argmax_change"] for e in edits],
    "target KL": [geo[e]["target_kl"] for e in edits],
    "draft KL old->new": [geo[e]["draft_kl_old_new"] for e in edits],
}
for name, x in preds.items():
    x = np.array(x)
    p, s = corr(x, y)
    boots = []
    for _ in range(4000):
        idx = rng.integers(0, len(edits), len(edits))
        if len(set(idx)) > 2:
            boots.append(np.corrcoef(x[idx], y[idx])[0, 1])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    # leave one edit out: linear fit on the others, predict the left-out loss
    err = []
    for i in range(len(edits)):
        m = np.ones(len(edits), bool); m[i] = False
        a, b = np.polyfit(x[m], y[m], 1)
        err.append(abs(a * x[i] + b - y[i]))
    print(f"  {name:24s} pearson {p:5.2f} [{lo:5.2f},{hi:5.2f}]  spearman {s:5.2f}  LOO |error| {np.mean(err) * 100:4.1f} pp")
print(f"  (mean tau loss over edits {y.mean() * 100:.1f}%, a constant predictor errs {np.mean(np.abs(y - y.mean())) * 100:.1f} pp)")
# without the two reasoning models: is it only the two large points?
small = [i for i, e in enumerate(edits) if e not in ("r1_distill", "nemotron_nano")]
x = np.array(preds["shift mean of 3 layers"])
p, s = corr(x[small], y[small])
print(f"  without R1-Distill and Nemotron (n={len(small)}): pearson {p:.2f} spearman {s:.2f}")
# per domain points
xs, ys = [], []
for e in edits:
    for d in EN:
        xs.append(np.mean(geo[e]["shift_rel"])); ys.append(1 - maps[e]["domains"][d]["tau"] / base["domains"][d]["tau"])
p, s = corr(xs, ys)
print(f"  per domain points (n={len(xs)}): pearson {p:.2f} spearman {s:.2f}")

# ------------------------------------------------------------------------------------------------ Q2
print("\nQ2. Do different edits move the states in the same directions?")
mus = {e: npz[e]["mu"].astype(np.float64) for e in edits}
for j, layer in enumerate((2, 16, 29)):
    M = np.array([mus[e][j * H:(j + 1) * H] for e in edits])
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    C = Mn @ Mn.T
    off = C[~np.eye(len(edits), dtype=bool)]
    print(f"  layer {layer:2d}: cosine of mean shifts between edits: median {np.median(off):.2f}, min {off.min():.2f}, "
          f"max {off.max():.2f}  (random directions in {H}d: ~{1 / np.sqrt(H):.3f})")
print("  cosine matrix of mean shifts, layer 16:")
M = np.array([mus[e][H:2 * H] for e in edits]); Mn = M / np.linalg.norm(M, axis=1, keepdims=True); C = Mn @ Mn.T
print("  " + " " * 14 + "".join(f"{e[:6]:>7s}" for e in edits))
for i, e in enumerate(edits):
    print(f"  {e:14s}" + "".join(f"{C[i, j]:7.2f}" for j in range(len(edits))))


def overlap(A, B):
    """Mean squared cosine of principal angles between row spaces of A and B (k x d), 1 = same subspace."""
    qa, _ = np.linalg.qr(A.T); qb, _ = np.linalg.qr(B.T)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(s ** 2))


for k in (4, 16):
    P = {e: npz[e]["shift_pca"][:k].astype(np.float64) for e in edits}
    vals = [overlap(P[a], P[b]) for i, a in enumerate(edits) for b in edits[i + 1:]]
    rnd = [overlap(rng.standard_normal((k, 3 * H)), rng.standard_normal((k, 3 * H))) for _ in range(20)]
    print(f"  top-{k:2d} shift subspaces (3 layers, {3 * H}d): overlap between edits median {np.median(vals):.3f} "
          f"[{min(vals):.3f}, {max(vals):.3f}], random {np.mean(rnd):.4f}")
print("  offset share of the shift (constant part):", {e: [round(v, 2) for v in geo[e]["offset_share"]] for e in edits})
print("  shift energy in the top-k input directions of W_fc vs the states' own energy there:")
for e in edits:
    g = geo[e]
    print(f"    {e:14s} " + "  ".join(f"k={k}: shift {g['energy_in_fc_top'][k]:.3f} / states {g['energy_in_fc_top_base_h'][k]:.3f}"
                                      for k in ("16", "64", "256")))

# ------------------------------------------------------------------------------------------------ Q3
print("\nQ3. Is there a shared subspace of what the draft has to learn (logit residual, rank-32 output basis)?")
B = {e: npz[e]["head_out_basis"].astype(np.float64) for e in edits}
for k in (4, 16, 32):
    vals = [overlap(B[a][:k], B[b][:k]) for i, a in enumerate(edits) for b in edits[i + 1:]]
    print(f"  top-{k:2d} output directions: overlap between edits median {np.median(vals):.3f} "
          f"[{min(vals):.3f}, {max(vals):.3f}]   random in 32000d ~ {k / 32000:.4f}")
print("  residual spectrum (energy share of top-16 / top-64):",
      {e: (round(geo[e]["residual_spectrum_share"]["16"], 2), round(geo[e]["residual_spectrum_share"]["64"], 2)) for e in edits})

# ------------------------------------------------------------------------------------------------ figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.array(preds["shift mean of 3 layers"])
    ax[0].scatter(x, y * 100, color="#2F6DB5")
    for e, xi, yi in zip(edits, x, y):
        ax[0].annotate(e, (xi, yi * 100), fontsize=8, xytext=(4, 2), textcoords="offset points")
    a, b = np.polyfit(x, y, 1)
    g = np.linspace(0, x.max() * 1.05, 50)
    ax[0].plot(g, (a * g + b) * 100, ":", color="#888")
    ax[0].set_xlabel("relative shift of the draft's input states")
    ax[0].set_ylabel("tau loss of the official draft, %")
    ax[0].set_title(f"Staleness predictor (pearson {corr(x, y)[0]:.2f}, n={len(edits)})", loc="left")
    acc = [(geo[e]["acc_old"], geo[e]["acc_new"]) for e in edits]
    idx = np.arange(len(edits))
    ax[1].bar(idx - 0.2, [a for a, _ in acc], 0.4, label="old features", color="#8C96A3")
    ax[1].bar(idx + 0.2, [b for _, b in acc], 0.4, label="new features", color="#2F6DB5")
    ax[1].set_xticks(idx, [e[:8] for e in edits], rotation=45, fontsize=8)
    ax[1].set_ylim(0.45, 0.72)
    ax[1].set_ylabel("depth-1 agreement with the edited target")
    ax[1].set_title("Draft reading old vs new target states", loc="left")
    ax[1].legend(fontsize=8, frameon=False)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_predictor_and_gap.png", dpi=130)
    print("\nfigure: figures/fig_predictor_and_gap.png")
except Exception as error:     # noqa: BLE001
    print("figure failed:", error)
