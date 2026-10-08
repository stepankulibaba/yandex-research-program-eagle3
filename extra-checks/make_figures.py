"""Extra figures for the summary report: languages and the public-draft zoo."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COL = {"dsl-8b": "#2F6DB5", "qwen3-1.7b": "#D9822B"}
NAME = {"dsl-8b": "DSL-8B (официальный драфт)", "qwen3-1.7b": "Qwen3-1.7B (AngelSlim)"}
LANGS = ["en", "de", "zh", "ja", "ru", "uk"]
LNAME = {"en": "англ.", "de": "нем.", "zh": "кит.", "ja": "яп.", "ru": "рус.", "uk": "укр."}

# 1. Languages: tau and OOV share, with the 1/p ceiling
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
x = np.arange(len(LANGS))
for k, p in enumerate(["dsl-8b", "qwen3-1.7b"]):
    r = json.loads((HERE / f"results/lang/{p}.json").read_text())["langs"]
    tau = [r[l]["all"]["tau"] for l in LANGS]
    sp = [r[l]["speed_subset"]["speedup"] for l in LANGS]
    oov = [np.mean([r[l][b]["oov_share"] for b in ("arena", "mgsm") if b in r[l]]) for l in LANGS]
    ax[0].bar(x + (k - 0.5) * 0.38, tau, 0.38, color=COL[p], label=NAME[p])
    for xi, s in zip(x + (k - 0.5) * 0.38, sp):
        ax[0].text(xi, 0.1, f"{s:.2f}×", ha="center", va="bottom", fontsize=8, color="white", rotation=90)
    ax[1].scatter(oov, tau, color=COL[p], s=45, label=NAME[p], zorder=3)
    for o, t, l in zip(oov, tau, LANGS):
        ax[1].annotate(LNAME[l], (o, t), textcoords="offset points", xytext=(4, 4), fontsize=9, color=COL[p])
pp = np.linspace(0.08, 0.7, 50)
ax[1].plot(pp, 1 / pp, color="#888", ls="--", label="потолок τ ≤ 1/p")
ax[0].set_xticks(x, [LNAME[l] for l in LANGS])
ax[0].set(ylabel="τ", title="Длина принятия по языкам\n(в столбцах — реальное ускорение)")
ax[0].legend(frameon=False, fontsize=9)
ax[1].set(xlabel="p — доля токенов ответа вне словаря драфта", ylabel="τ", ylim=(1, 6),
          title="Чем больше токенов вне словаря, тем ниже τ")
ax[1].legend(frameon=False, fontsize=9)
for a in ax:
    a.grid(alpha=.2)
fig.tight_layout()
fig.savefig(FIG / "5_languages.png", dpi=150)

# 2. Public drafts for Qwen3-8B: nothink vs think
z = json.loads((HERE / "results/zoo/zoo_eval.json").read_text())
show = {"deepseek_ttt7_plus1": "DeepSeek (5 слоёв)", "io_e31_llamaarch": "IO EAGLE-3.1 llama", "io_e31_qwen3arch": "IO EAGLE-3.1 qwen3",
        "io_e3_qwen3arch": "IO EAGLE-3 qwen3", "io_e31_qwen3arch_3e4": "IO EAGLE-3.1 lr 3e-4", "io_e31_fcnorm": "IO EAGLE-3.1 fc_norm",
        "redhat_thinking": "RedHat Thinking", "tengyunw": "Tengyunw", "thoughtworks": "thoughtworks", "redhat": "RedHat",
        "angelslim": "AngelSlim"}
names = sorted(show, key=lambda n: -z["modes"]["nothink"][n]["all"]["tau_chain"])
a = [z["modes"]["nothink"][n]["all"]["tau_chain"] for n in names]
b = [z["modes"]["think"][n]["all"]["tau_chain"] for n in names]
fig, ax = plt.subplots(figsize=(11, 4.6))
x = np.arange(len(names))
ax.bar(x - 0.2, a, 0.4, color="#2F6DB5", label="без рассуждений")
ax.bar(x + 0.2, b, 0.4, color="#D9822B", label="с рассуждениями")
for xi, u, v in zip(x, a, b):
    ax.text(xi, max(u, v) + 0.05, f"{100 * (v - u) / u:+.0f}%", ha="center", fontsize=8)
ax.set_xticks(x, [show[n] for n in names], rotation=30, ha="right")
ax.set(ylabel="цепочечная τ (до 6 шагов)", title="11 публичных драфтов для одного target (Qwen3-8B): один target, два режима")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=.2)
fig.tight_layout()
fig.savefig(FIG / "6_zoo_modes.png", dpi=150)

# 3. Read-subspace overlap (layer 33, top-64) and data-weighted fc energy
g = json.loads((HERE / "results/zoo/zoo_geometry.json").read_text())
gn = [n for n in names if n in g["blocks"] and n != "deepseek_ttt7_plus1"] + ["deepseek_ttt7"]
lab = [show.get(n, "DeepSeek (5 слоёв)") for n in gn]
M = np.eye(len(gn))
for i, p in enumerate(gn):
    for j, q in enumerate(gn):
        if i != j:
            key = f"{p}|{q}" if f"{p}|{q}" in g["pairs"] else f"{q}|{p}"
            M[i, j] = g["pairs"].get(key, {}).get("fc_subspace_L33_r64", np.nan)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
im = ax[0].imshow(M, cmap="viridis", vmin=0, vmax=1)
ax[0].set_xticks(range(len(gn)), lab, rotation=60, ha="right", fontsize=8)
ax[0].set_yticks(range(len(gn)), lab, fontsize=8)
ax[0].set_title("Что fc читает из слоя 33 target:\nперекрытие топ-64 направлений (случайное ≈ 0,016)")
fig.colorbar(im, ax=ax[0], fraction=0.046)
tri = [n for n in gn if g["blocks"][n]["layer_ids"] == [2, 18, 33]]
E = np.array([g["blocks"][n]["data_energy_share_nothink"] for n in tri])
W = np.array([g["blocks"][n]["weight_energy_share"] for n in tri])
y = np.arange(len(tri))
for k, (lbl, c) in enumerate(zip(["слой 2", "слой 18", "слой 33"], ["#9AA0A6", "#D9822B", "#2F6DB5"])):
    ax[1].barh(y, E[:, k], left=E[:, :k].sum(1), color=c, label=lbl)
    if k < 2:
        ax[1].scatter(W[:, :k + 1].sum(1), y, marker="|", s=260, linewidths=2.5, color="black", zorder=5,
                      label="граница по весам" if k == 0 else None)
ax[1].set_yticks(y, [show[n] for n in tri], fontsize=8)
ax[1].set(xlabel="доля сигнала fc на реальных данных", title="Откуда fc берёт сигнал на данных\n(чёрные метки — та же доля по весам)")
ax[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
fig.tight_layout()
fig.savefig(FIG / "7_zoo_geometry.png", dpi=150)
print(sorted(p.name for p in FIG.iterdir()))
