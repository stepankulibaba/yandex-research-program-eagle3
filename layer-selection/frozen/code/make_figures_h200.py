"""Figures for the H200 layer study (DSL-8B official EAGLE-3 draft vs Qwen3-1.7B AngelSlim draft)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results/h200"
FIG = HERE / "figures/h200"
FIG.mkdir(parents=True, exist_ok=True)
PAIRS = {"dsl-8b": ("DeepSeek-R1-Distill-Llama-8B (оф. драфт)", "#2F6DB5"),
         "qwen3-1.7b": ("Qwen3-1.7B (драфт AngelSlim)", "#D9822B")}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load(pair, name):
    return json.loads((RES / pair / name).read_text(encoding="utf-8"))


# 1. Official-draft ablations, relative change of tau
labels = [("drop_low", "без нижнего (2)"), ("drop_mid", "без среднего (N/2)"), ("only_high", "только N−3"),
          ("only_low_mid", "только 2 + N/2"), ("mid", "N/2 → 0,7N"), ("high_last", "N−3 → N (последний)"),
          ("high_prev", "N−3 → N−1"), ("rank_half", "fc ранг H/2"), ("rank_quarter", "fc ранг H/4")]
fig, ax = plt.subplots(figsize=(11, 5.2))
y = np.arange(len(labels))
for k, (pair, (title, color)) in enumerate(PAIRS.items()):
    e = load(pair, "eval_official.json")
    N, H = e["n_layers"], e["hidden"]
    tri = e["triplet"]
    q = round(0.7 * N)
    keymap = {"mid": f"mid_{tri[1]}to{q}", "high_last": f"high_{tri[2]}to{N}", "high_prev": f"high_{tri[2]}to{N - 1}",
              "rank_half": f"fc_rank{H // 2}", "rank_quarter": f"fc_rank{H // 4}"}
    rows = [r for r in load(pair, "rows_baseline.json") if r["bench"] in ("mt_bench", "gsm8k")]
    c = sum(r["cycles"] for r in rows)
    base = (sum(r["accepted_sum"] for r in rows) + c) / c
    vals, lo, hi = [], [], []
    for key, _ in labels:
        a = e["ablations"][keymap.get(key, key)]
        vals.append(100 * (a["tau"] - base) / base)
        lo.append(100 * a["delta_ci95"][0] / base)
        hi.append(100 * a["delta_ci95"][1] / base)
    vals, lo, hi = map(np.array, (vals, lo, hi))
    ax.barh(y + (k - 0.5) * 0.38, vals, height=0.38, color=color, label=f"{title}, τ={base:.2f}",
            xerr=[vals - lo, hi - vals], ecolor="#444", capsize=2)
ax.set_yticks(y, [l for _, l in labels])
ax.invert_yaxis()
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("изменение τ относительно официального драфта, %")
ax.set_title("Официальные драфты без переобучения: что будет, если убрать или подменить слой")
ax.legend(loc="lower left", frameon=False)
ax.grid(axis="x", alpha=.25)
fig.tight_layout()
fig.savefig(FIG / "1_official_ablations.png", dpi=150)

# 2. Layer sweep (step-0 accuracy, frozen draft) + future-token probes, depth normalised
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for pair, (title, color) in PAIRS.items():
    s = load(pair, "sweep.json")
    N, tri = s["n_layers"], s["triplet"]
    x = np.arange(N + 1) / N
    single = np.array([s["step0"][f"single_{i}"]["acc"] for i in range(N + 1)])
    axes[0].plot(x, single, marker="o", ms=3, color=color, label=title)
    axes[0].scatter([x[np.argmax(single)]], [single.max()], s=90, facecolors="none", edgecolors=color, lw=2)
    gain = [s["step0"][f"pair_{i}_{tri[2]}"]["acc"] - single[tri[2]] if i != tri[2] else np.nan for i in range(N + 1)]
    axes[1].plot(x, 100 * np.array(gain), marker="o", ms=3, color=color, label=title)
    a = load(pair, "analysis.json")
    for k, ls in [(1, ":"), (3, "-")]:
        acc = np.array([a["future_probes"][f"{i}:{k}"]["probe_acc"] for i in range(N + 1)])
        axes[2].plot(x, acc / acc.max(), ls=ls, color=color, label=f"{title.split(' ')[0]}, k={k}")
for ax in axes:
    for v, lab in [(2 / 32, "2"), (0.5, "N/2"), (0.7, "0,7N"), (29 / 32, "N−3")]:
        ax.axvline(v, color="#999", ls="--", lw=.8)
        ax.text(v, 0.01, lab, fontsize=8, color="#666", ha="center", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlabel("глубина слоя, доля от N")
    ax.grid(alpha=.2)
axes[0].set(title="Драфт видит только один слой\n(step-0 accuracy, драфт заморожен)", ylabel="accuracy")
axes[1].set(title="Прирост от второго слоя к N−3", ylabel="Δ accuracy, п.п.")
axes[1].axhline(0, color="black", lw=.8)
axes[2].set(title="Пробы: токен через k шагов\n(нормировано на максимум)", ylabel="доля от лучшего слоя")
axes[0].legend(frameon=False, fontsize=9)
axes[2].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "2_layer_sweep.png", dpi=150)

# 3. Where the official fc takes its signal from + residual norm growth
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for k, (pair, (title, color)) in enumerate(PAIRS.items()):
    a = load(pair, "analysis.json")
    N, tri = a["n_layers"], a["triplet"]
    share = a["fc_decomposition"]["contribution_energy_share"]
    axes[0].bar(np.arange(3) + (k - 0.5) * 0.38, np.array(share) * 100, width=0.38, color=color, label=title)
    rms = np.array(a["norms"]["rms_mean"])
    axes[1].semilogy(np.arange(N + 1) / N, rms / rms[tri[0]], color=color, marker="o", ms=3, label=title)
axes[0].set_xticks(range(3), ["нижний (2)", "средний (N/2)", "верхний (N−3)"])
axes[0].set(title="Откуда официальный fc берёт сигнал\n(доля энергии ‖W_k h_k‖² в смешанном векторе)", ylabel="%")
axes[0].legend(frameon=False, fontsize=9)
axes[1].set(title="Рост нормы residual stream\n(относительно слоя 2)", xlabel="глубина слоя, доля от N", ylabel="RMS / RMS(слой 2)")
axes[1].grid(alpha=.2)
fig.tight_layout()
fig.savefig(FIG / "3_fc_sources.png", dpi=150)

# 4. Fusion variants on Qwen (frozen draft, refit of the input layer only)
r = load("qwen3-1.7b", "fusion/phase_b.json")
off = r["tau"]["official"]["tau"]
names = {"fc3_scratch": "(2,14,25) как upstream", "fc3_norm": "(2,14,25) + RMSNorm", "fc3_last": "(2,14,28) + RMSNorm",
         "tri_2_20_25": "(2,20,25) + RMSNorm", "tri_14_20_25": "(14,20,25) + RMSNorm", "pair_20_25": "пара (20,25)",
         "single_25": "только 25 (N−3)", "single_28": "только 28 (последний)", "fc4_norm": "4 слоя (2,11,20,25)",
         "mix3_init": "смесь всех слоёв", "router_mix": "роутер по слоям (MoE)", "lowrank_1536": "low-rank, те же парам.",
         "moe_lowrank": "low-rank MoE", "moe_lowrank_bal": "low-rank MoE, баланс", "fc3_official_ft": "оф. fc дообучен"}
order = sorted(names, key=lambda n: r["tau"][n]["tau"])
fig, ax = plt.subplots(figsize=(10, 6.4))
yy = np.arange(len(order))
t = np.array([r["tau"][n]["tau"] for n in order])
lo = np.array([r["tau"][n]["delta_ci95"][0] for n in order]) + off
hi = np.array([r["tau"][n]["delta_ci95"][1] for n in order]) + off
colors = ["#7A7A7A" if n == "fc3_official_ft" else ("#B04A5A" if "moe" in n or "router" in n else "#D9822B") for n in order]
ax.barh(yy, t, xerr=[t - lo, hi - t], color=colors, capsize=2)
ax.axvline(off, color="black", ls="--", lw=1)
ax.text(off, len(order) - 0.3, f" официальный fc τ={off:.2f}", fontsize=9)
ax.set_yticks(yy, [f"{names[n]} ({r['variants'][n]['params'] / 1e6:.1f}M)" for n in order])
ax.set_xlim(2.8, 4.6)
ax.set_xlabel("τ (MT-bench + GSM8K, протокол статьи)")
ax.set_title("Qwen3-1.7B: переобучен только входной слой fc, драфт заморожен")
ax.grid(axis="x", alpha=.25)
fig.tight_layout()
fig.savefig(FIG / "4_qwen_fusion.png", dpi=150)
print(sorted(p.name for p in FIG.iterdir()))
