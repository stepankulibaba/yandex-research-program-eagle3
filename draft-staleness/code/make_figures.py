"""Four pitch figures from the staleness results (no GPU): figures/fig_pitch_*.png

  1 ladder      tau loss of the official draft per edit and domain
  2 predictor   tau loss vs weight change (weight-only) and vs state shift
  3 mechanism   transfer on changed tokens vs damage on unchanged tokens
  4 fix         tree tau with / without the fc correction, paired CI on identical text
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = Path(__file__).resolve().parent.parent / "results" / "llama31-8b"
OUT = Path(__file__).resolve().parent.parent / "figures"
EN = ["general", "math", "code", "medicine", "tool"]
ORDER = ["limo_lora", "toolace", "supernova", "vikhr", "ultramedical", "huatuo_o1", "abliterated", "r1_distill",
         "nemotron_nano"]
LABEL = {"limo_lora": "LIMO (LoRA)", "toolace": "ToolACE", "supernova": "SuperNova", "vikhr": "Vikhr",
         "ultramedical": "UltraMedical", "huatuo_o1": "HuatuoGPT-o1", "abliterated": "abliterated",
         "r1_distill": "R1-Distill", "nemotron_nano": "Nemotron-Nano"}
BLUE, GREY, RED, GREEN = "#2F6DB5", "#8C96A3", "#B23A48", "#2E7D4F"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(n):
    return json.loads((DIR / n).read_text(encoding="utf-8"))


base = load("base__inst.json")
loss = {e: [1 - load(f"{e}__inst.json")["domains"][d]["tau"] / base["domains"][d]["tau"] for d in EN] for e in ORDER}

# 1 ladder
fig, ax = plt.subplots(figsize=(8, 4.2))
mean = [np.mean(loss[e]) * 100 for e in ORDER]
ax.barh(range(len(ORDER)), mean, color=BLUE)
for i, e in enumerate(ORDER):
    ax.scatter([v * 100 for v in loss[e]], [i] * len(EN), color="black", s=8, zorder=3)
ax.set_yticks(range(len(ORDER)), [LABEL[e] for e in ORDER])
ax.set_xlabel("tau loss of the official Llama-3.1-Instruct draft, % (bar: mean of 5 domains, dots: domains)")
ax.set_title("Fine-tuning the target makes the EAGLE-3 draft stale", loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_pitch_1_ladder.png", dpi=140); plt.close(fig)

# 2 predictor
g1 = {e: load(f"geo1_{e}.json") for e in ORDER}
g1b = {e: load(f"geo1b_{e}.json") for e in ORDER}
wchange = [np.mean([v for n, v in g1b[e]["weights_rel"].items() if "o_proj" in n or "down_proj" in n]) * 100 for e in ORDER]
shift = [np.mean(g1[e]["shift_rel"]) for e in ORDER]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, x, xl in ((axes[0], wchange, "relative weight change of o_proj / down_proj, %  (weights only)"),
                  (axes[1], shift, "relative shift of the draft's input states  (forward passes)")):
    ax.scatter(x, mean, color=BLUE)
    for e, xi, yi in zip(ORDER, x, mean):
        ax.annotate(LABEL[e], (xi, yi), fontsize=8, xytext=(4, 2), textcoords="offset points")
    r = np.corrcoef(x, mean)[0, 1]
    ax.set_xlabel(xl); ax.set_ylabel("tau loss, %")
    ax.set_title(f"pearson {r:.2f}", loc="left")
fig.suptitle("Staleness is predictable without generating text", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "fig_pitch_2_predictor.png", dpi=140); plt.close(fig)

# 3 mechanism: transfer vs damage, from geo1b npz
top0 = np.load(DIR / "geo1b_base_ids.npz")["top0"]
transfer, damage = [], []
for e in ORDER:
    z = np.load(DIR / f"geo1b_{e}_ids.npz")
    t1, dn, do = z["top1"], z["draft_new"], z["draft_old"]
    valid = (t1 >= 0) & (top0 >= 0)
    U, C = valid & (t1 == top0), valid & (t1 != top0)
    damage.append(((do[U] == t1[U]).mean() - (dn[U] == t1[U]).mean()) * 100)
    transfer.append(((dn[C] == t1[C]).mean() - (do[C] == t1[C]).mean()) * 100)
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(ORDER))
ax.bar(x - 0.2, transfer, 0.4, color=GREEN, label="transfer: +pp where the target changed its answer")
ax.bar(x + 0.2, [-d for d in damage], 0.4, color=RED, label="damage: -pp where the target did not change")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, [LABEL[e] for e in ORDER], rotation=35, ha="right")
ax.set_ylabel("depth-1 agreement, new vs old features (pp)")
ax.set_title("Reading the new states helps and hurts at the same time", loc="left")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_pitch_3_mechanism.png", dpi=140); plt.close(fig)

# 4 fix
fix = {"nemotron_nano": (8.6, 7.4, 10.0), "r1_distill": (9.6, 8.4, 10.8), "toolace": (0.2, -0.5, 1.0),
       "abliterated": (-9.7, -11.0, -8.3)}            # logs/analysis_fix_prefix.txt (identical text, 95% CI)
fig, ax = plt.subplots(figsize=(7, 3.6))
names = list(fix)
vals = [fix[n][0] for n in names]
err = np.array([[fix[n][0] - fix[n][1] for n in names], [fix[n][2] - fix[n][0] for n in names]])
ax.bar(range(len(names)), vals, yerr=err, color=[GREEN if v > 1 else (RED if v < -1 else GREY) for v in vals], capsize=4)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(names)), [LABEL[n] for n in names])
ax.set_ylabel("tree tau change, % (identical text)")
ax.set_title("Rank-2 fc correction, fitted in 1 min, no generation, no training", loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_pitch_4_fix.png", dpi=140); plt.close(fig)
print("figures:", sorted(p.name for p in OUT.glob("fig_pitch_*.png")))
