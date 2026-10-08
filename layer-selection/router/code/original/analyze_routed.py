"""Analysis of the unfrozen routed-draft runs: tau over training, how the router's layer choice evolves,
what it depends on, and how it compares with the frozen-draft study.

py -3.12 analyze_routed.py <pair>      (expects results/h200/<pair>/routed/history.json)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
pair = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
H = json.loads((HERE / f"results/h200/{pair}/routed/history.json").read_text())
FIG = HERE / f"figures/routed/{pair}"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
evals = H["evals"]
names = list(evals[0]["res"])
steps = [e["step"] for e in evals]
epochs = [e["epoch"] for e in evals]
COL = {"fixed_raw": "#7A7A7A", "fixed_norm": "#2F6DB5", "fixed_alt": "#5FA8D3", "static_top3": "#2E7D4F",
       "static_top2": "#8BC79A", "token_top3": "#B23A48", "token_top2": "#E89A52"}
LABEL = {"fixed_raw": "оригинал (2,N/2,N−3)", "fixed_norm": "оригинал + RMSNorm", "fixed_alt": "(N/2,0.7N,N−3) + RMSNorm",
         "static_top3": "статический top-3", "static_top2": "статический top-2", "token_top3": "роутер top-3",
         "token_top2": "роутер top-2"}

# 1. tau over training
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
for n in names:
    t = [e["res"][n]["tau_chain"] for e in evals]
    ax[0].plot(epochs, t, marker="o", ms=3, color=COL.get(n), label=LABEL.get(n, n))
    ax[1].plot(epochs[1:], [x - y for x, y in zip(t[1:], [e["res"]["fixed_raw"]["tau_chain"] for e in evals[1:]])],
               marker="o", ms=3, color=COL.get(n), label=LABEL.get(n, n))
ax[0].set(xlabel="эпоха", ylabel="цепочечная τ (до 6 шагов)", title="Качество драфта по ходу обучения")
ax[1].axhline(0, color="black", lw=.8)
ax[1].set(xlabel="эпоха", ylabel="Δτ к оригиналу", title="Разница с оригинальной тройкой слоёв")
ax[0].legend(frameon=False, fontsize=8)
for a in ax:
    a.grid(alpha=.2)
fig.tight_layout()
fig.savefig(FIG / "tau_curves.png", dpi=150)

# 2. layer choice over training for routed variants
routed = [n for n in names if "layer_share" in evals[-1]["res"][n]]
if routed:
    fig, axes = plt.subplots(len(routed), 1, figsize=(12, 2.2 * len(routed) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for a, n in zip(axes, routed):
        M = np.array([e["res"][n]["layer_share"] for e in evals[1:]]).T
        im = a.imshow(M, aspect="auto", origin="lower", cmap="magma", vmin=0,
                      extent=[epochs[1] - 0.05, epochs[-1] + 0.05, -0.5, M.shape[0] - 0.5])
        a.set(ylabel="слой target", title=f"{LABEL.get(n, n)}: доля выборов слоя")
        fig.colorbar(im, ax=a, fraction=0.02)
    axes[-1].set_xlabel("эпоха")
    fig.tight_layout()
    fig.savefig(FIG / "layer_choice_over_training.png", dpi=150)

# 3. final layer choice by category
for n in [x for x in routed if x.startswith("token")]:
    by = evals[-1]["res"][n]["by"]
    keys = ["bench", "token", "conf", "pos"]
    fig, axes = plt.subplots(1, len(keys), figsize=(16, 3.8), sharey=True)
    for a, k in zip(axes, keys):
        groups = sorted(by[k])
        M = np.array([by[k][g] for g in groups]).T
        a.imshow(M, aspect="auto", origin="lower", cmap="magma", vmin=0)
        a.set_xticks(range(len(groups)), groups, rotation=30, ha="right", fontsize=8)
        a.set_title({"bench": "задача", "token": "тип токена", "conf": "уверенность target", "pos": "позиция"}[k])
    axes[0].set_ylabel("слой target")
    fig.suptitle(f"{LABEL.get(n, n)}: какие слои выбираются (конец обучения)")
    fig.tight_layout()
    fig.savefig(FIG / f"by_category_{n}.png", dpi=150)

# 4. text summary
print("final tau:")
for n in names:
    r = evals[-1]["res"][n]
    print(f"  {n:12s} tau_chain={r['tau_chain']:.3f} tau_cycles={r['tau_cycles']:.3f} "
          f"d1={r['accept_by_depth'][0]:.3f} d6={r['accept_by_depth'][-1]:.3f}")
for n in routed:
    share = np.array(evals[-1]["res"][n]["layer_share"])
    top = np.argsort(share)[::-1][:6]
    print(f"{n}: top layers " + ", ".join(f"{int(i)}:{share[i]:.2f}" for i in top))
    if n.startswith("token"):
        by = evals[-1]["res"][n]["by"]
        for k, groups in by.items():
            for g, v in groups.items():
                v = np.array(v)
                t3 = np.argsort(v)[::-1][:3]
                print(f"    {k}={g}: " + ", ".join(f"{int(i)}:{v[i]:.2f}" for i in t3))
