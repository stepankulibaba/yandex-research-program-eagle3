"""Report figures from the downloaded Kaggle results (results/kaggle_a, kaggle_b, kaggle_b2)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
BLUE, GREY, RED = "#4C78A8", "#9AA0A6", "#D1495B"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


a = load(RES / "kaggle_a/out/phase_a.json")
N, tri = a["n_layers"], a["triplet"]
L = np.arange(N + 1)

# 1. Layer analysis
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
ax[0].semilogy(L, a["norms_train"]["rms_mean"], marker="o", color=BLUE)
ax[0].set(title="RMS residual stream", xlabel="слой (вход)", ylabel="RMS, log")
r2 = a["ridge_r2_to_fused"]
ax[1].plot(L, [r2[str(i)] for i in L], marker="o", color=BLUE)
ax[1].set(title=f"R² восстановления официального g из одного слоя\n(вся тройка: {r2['triplet']:.2f})", xlabel="слой", ylim=(0, 1))
for k, c in zip(range(1, 4), [BLUE, "#F58518", "#54A24B"]):
    ax[2].plot(L, [a["future_probes"][f"{i}:{k}"]["probe_acc"] for i in L], marker="o", color=c, label=f"k={k}")
ax[2].set(title="Проба: top-1 greedy-токена target через k шагов", xlabel="слой", ylabel="accuracy")
ax[2].legend()
for x in ax:
    for t in tri:
        x.axvline(t, color=RED, ls="--", alpha=.5)
    x.grid(alpha=.2)
fig.tight_layout()
fig.savefig(FIG / "a_layers.png", dpi=140)

# 2. CKA
fig, x = plt.subplots(figsize=(6.2, 5.4))
im = x.imshow(np.array(a["cka"]), cmap="viridis", vmin=0, vmax=1)
fig.colorbar(im)
x.set(title="Linear CKA между слоями Qwen3-1.7B", xlabel="слой", ylabel="слой")
fig.tight_layout()
fig.savefig(FIG / "a_cka.png", dpi=140)

# 3. Ablations
abl = {k: v for k, v in a["ablations"].items() if k != "baseline"}
base = a["baseline"]["tau"]
names = list(abl)
d = np.array([abl[n]["tau"] - base for n in names])
lo = np.array([abl[n]["delta_ci95"][0] for n in names])
hi = np.array([abl[n]["delta_ci95"][1] for n in names])
fig, x = plt.subplots(figsize=(10, 0.3 * len(names) + 1.4))
y = np.arange(len(names))
x.barh(y, d, xerr=[d - lo, hi - d], color=[RED if h < 0 else GREY for h in hi])
x.set_yticks(y, names)
x.invert_yaxis()
x.axvline(0, color="black", lw=1)
x.set(title=f"Абляции официального драфта: Δτ к upstream (τ={base:.2f}), 95% paired bootstrap", xlabel="Δτ")
x.grid(axis="x", alpha=.2)
fig.tight_layout()
fig.savefig(FIG / "a_ablations.png", dpi=140)


def variant_plot(run, fname, title):
    r = load(RES / f"kaggle_{run}" / "out/phase_b.json")
    off = r["tau"]["official"]["tau"]
    names = [n for n in r["variants"]]
    order = sorted(names, key=lambda n: r["tau"][n]["tau"])
    fig, ax = plt.subplots(1, 2, figsize=(15, 0.36 * len(order) + 1.6), sharey=True)
    y = np.arange(len(order))
    t = np.array([r["tau"][n]["tau"] for n in order])
    lo = np.array([r["tau"][n]["delta_ci95"][0] for n in order]) + off
    hi = np.array([r["tau"][n]["delta_ci95"][1] for n in order]) + off
    ax[0].barh(y, t, xerr=[t - lo, hi - t], color=[BLUE if "official" not in n else "#F58518" for n in order])
    ax[0].axvline(off, color="black", ls="--", label=f"официальный fc τ={off:.2f}")
    ax[0].set_yticks(y, [f"{n} ({r['variants'][n]['params'] / 1e6:.1f}M)" for n in order])
    ax[0].set(title="τ на 48 промптах (усы: 95% CI)", xlabel="τ")
    ax[0].legend(loc="lower right")
    s = np.array([r["step0_eval"][n]["acc"] for n in order])
    ax[1].barh(y, s, color=[BLUE if "official" not in n else "#F58518" for n in order])
    ax[1].axvline(r["step0_eval"]["official"]["acc"], color="black", ls="--")
    ax[1].set(title="step-0 accuracy на ответах target (фиксированный текст)", xlabel="acc", xlim=(0.4, 0.8))
    for x in ax:
        x.grid(axis="x", alpha=.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=140)
    return r


rb = variant_plot("b", "b_variants.png", "Раунд 1: обучение fusion при замороженном драфте (одинаковые 8000 последовательностей)")
mixes = [(n, v["mix_weights_static"]) for n, v in rb["variants"].items() if n in ("mix3_uniform", "mix3_init")]
mixes += [("router_mix: среднее по токенам", rb["variants"]["router_mix"]["router_mean_weights"])]
fig, axes = plt.subplots(len(mixes), 1, figsize=(11, 1.5 * len(mixes) + 0.8), sharex=True)
for x, (n, w) in zip(axes, mixes):
    im = x.imshow(np.array(w), aspect="auto", cmap="magma", vmin=0)
    x.set(title=n, ylabel="слот", yticks=[0, 1, 2])
    fig.colorbar(im, ax=x)
axes[-1].set(xlabel="слой target", xticks=range(N + 1))
fig.tight_layout()
fig.savefig(FIG / "b_mix_weights.png", dpi=140)

if (RES / "kaggle_b2/out/phase_b.json").exists():
    variant_plot("b2", "b2_variants.png", "Раунд 2: фиксированные наборы слоёв (другие 8000 последовательностей)")
print(sorted(p.name for p in FIG.iterdir()))
