"""Builds the two Kaggle notebooks (phase A analysis, phase B fusion training) with embedded sources."""
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
SOURCES = ["layer_lab.py", "phase_a.py", "phase_b.py"]
EAGLE_COMMIT = "cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"
MODELS = {"base": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
          "draft": ("AngelSlim/Qwen3-1.7B_eagle3", "94441b48acc5804677ae12259617c83323b543a9")}
USER = "<kaggle-user>"

files = {name: (HERE / name).read_bytes() for name in SOURCES}
manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for name, data in files.items():
        zf.writestr(name, data)
payload = base64.b64encode(buf.getvalue()).decode()

SETUP = f'''import os, sys, subprocess, json, hashlib, base64, io, zipfile
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.53.2", "accelerate>=1.1,<2",
                "huggingface_hub>=0.31,<1", "sentencepiece>=0.1.99"], check=True)
EAGLE_DIR = Path("/tmp/EAGLE")
if not EAGLE_DIR.exists():
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "https://github.com/SafeAILab/EAGLE.git", str(EAGLE_DIR)], check=True)
subprocess.run(["git", "-C", str(EAGLE_DIR), "checkout", "-q", "--detach", "{EAGLE_COMMIT}"], check=True)
assert subprocess.check_output(["git", "-C", str(EAGLE_DIR), "rev-parse", "HEAD"], text=True).strip() == "{EAGLE_COMMIT}"

SRC = Path("/kaggle/working/src")
SRC.mkdir(parents=True, exist_ok=True)
expected = {manifest!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode("{payload}"))) as zf:
    for name in zf.namelist():
        (SRC / name).write_bytes(zf.read(name))
for name, digest in expected.items():
    assert hashlib.sha256((SRC / name).read_bytes()).hexdigest() == digest, name
sys.path.insert(0, str(SRC))

import torch
from importlib.metadata import version
assert torch.cuda.is_available(), "GPU is required"
env = {{"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "transformers": version("transformers"),
       "python": sys.version.split()[0], "eagle_commit": "{EAGLE_COMMIT}", "sources_sha256": expected}}
print(json.dumps(env, indent=2))

from huggingface_hub import snapshot_download
BASE = snapshot_download("{MODELS['base'][0]}", revision="{MODELS['base'][1]}")
DRAFT = snapshot_download("{MODELS['draft'][0]}", revision="{MODELS['draft'][1]}")
OUT = Path("/kaggle/working/out")
OUT.mkdir(exist_ok=True)
(OUT / "environment.json").write_text(json.dumps(env, indent=2))
'''

COMMON_CFG = '''COMMON = {"eagle_dir": str(EAGLE_DIR), "base": BASE, "draft": DRAFT, "seed": 0,
          "total_token": 60, "depth": 7, "top_k": 10, "max_new_tokens": 128,
          "prompt_counts": [["mt_bench", 32], ["gsm8k", 8], ["humaneval", 8]], "max_len": 1024}
'''


def notebook(cells):
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [nbformat.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbformat.v4.new_code_cell(c[1])
                for c in cells]
    nbformat.validate(nb)
    return nb


A_CELLS = [
    ("md", """# EAGLE-3: какие слои target нужны draft-модели (фаза A)

Пара моделей: `Qwen/Qwen3-1.7B` (28 слоёв, H=2048) + `AngelSlim/Qwen3-1.7B_eagle3`. Код EAGLE закреплён на commit `cb7e084`.
Upstream передаёт драфту residual stream на входе слоёв **(2, N/2, N−3) = (2, 14, 25)** и смешивает их одним `Linear(3H→H)` без bias.
Индекс i ниже = hidden state на входе слоя i (0 — эмбеддинги, 28 — выход последнего слоя до финальной нормы).

Что считается без обучения:
1. RMS-норма residual stream по слоям.
2. Разложение официального fc: g = W_low·h_2 + W_mid·h_14 + W_high·h_25 — доля энергии, косинусы, эффективный ранг.
3. Linear CKA между всеми слоями.
4. Ridge R² восстановления официального g из каждого отдельного слоя.
5. Tuned-lens пробы: насколько слой i предсказывает greedy-токен target на k = 1, 2, 3 шага вперёд (словарь драфта).
6. Абляции на реальном спекулятивном декодировании (temperature 0): слот заменён средним, сдвинут на соседний слой (с выравниванием RMS), fc урезан по рангу. Метрика — acceptance length τ (с bonus-токеном), 95% интервал Δτ — paired bootstrap по промптам.

Ограничения: одна неофициальная пара 1.7B, 48 промптов × 128 токенов, обучающие тексты для проб — UltraChat."""),
    ("code", SETUP),
    ("code", COMMON_CFG + '''import phase_a
CFG_A = dict(COMMON, out=str(OUT), n_train_seqs=400, train_positions=40000, test_positions=100000,
             cka_tokens=4096, horizons=3, probe_epochs=3, probe_layers=None,
             shifts=[[0, 1, 3, 4, 6], [8, 11, 17, 20], [20, 22, 24, 26, 27, 28]],
             unscaled_shifts={"0": [1], "2": [26, 28]}, fc_ranks=[128, 256, 512, 1024])
result = phase_a.run(CFG_A)
'''),
    ("md", "## Результаты"),
    ("code", '''import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
r = json.loads((OUT / "phase_a.json").read_text())
N, tri = r["n_layers"], r["triplet"]
layers = np.arange(N + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
axes[0].semilogy(layers, r["norms_train"]["rms_mean"], marker="o", label="UltraChat")
axes[0].semilogy(layers, r["norms_test"]["rms_mean"], marker=".", label="ответы target")
for t in tri: axes[0].axvline(t, color="red", ls="--", alpha=.5)
axes[0].set(title="RMS residual stream по слоям", xlabel="слой (вход)", ylabel="RMS"); axes[0].legend()
r2 = r["ridge_r2_to_fused"]
axes[1].plot(layers, [r2[str(i)] for i in layers], marker="o")
for t in tri: axes[1].axvline(t, color="red", ls="--", alpha=.5)
axes[1].set(title=f"R² восстановления g из одного слоя (тройка: {r2['triplet']:.3f})", xlabel="слой", ylim=(min(0, min(r2[str(i)] for i in layers)), 1))
for k in range(1, 4):
    axes[2].plot(layers, [r["future_probes"][f"{i}:{k}"]["probe_acc"] for i in layers], marker="o", label=f"k={k}")
for t in tri: axes[2].axvline(t, color="red", ls="--", alpha=.5)
axes[2].set(title="Tuned-lens проба: top-1 к greedy-токену target через k шагов", xlabel="слой", ylabel="accuracy"); axes[2].legend()
fig.tight_layout(); fig.savefig(OUT / "layers.png", dpi=150); plt.show()

d = r["fc_decomposition"]
display(pd.DataFrame({"доля энергии W_k h_k": d["contribution_energy_share"], "‖W_k h_k‖²/‖g‖²": d["contribution_over_g_energy"]},
                     index=[f"low ({tri[0]})", f"mid ({tri[1]})", f"high ({tri[2]})"]).round(3))
display(pd.DataFrame(d["mean_cosine"], index=["low", "mid", "high"], columns=["low", "mid", "high"]).round(3))
display(pd.DataFrame(d["effective_rank"]).T)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(np.array(r["cka"]), cmap="viridis", vmin=0, vmax=1); fig.colorbar(im)
ax.set(title="Linear CKA между слоями", xlabel="слой", ylabel="слой"); fig.savefig(OUT / "cka.png", dpi=150); plt.show()
'''),
    ("code", '''abl = pd.DataFrame(r["ablations"]).T
abl["delta"] = abl["tau"] - r["baseline"]["tau"]
abl["ci_low"] = [c[0] if isinstance(c, list) else 0 for c in abl.get("delta_ci95", [None] * len(abl))]
abl["ci_high"] = [c[1] if isinstance(c, list) else 0 for c in abl.get("delta_ci95", [None] * len(abl))]
cols = [c for c in ["capture", "tau", "delta", "ci_low", "ci_high", "tau_mt_bench", "tau_gsm8k", "tau_humaneval", "mismatched_prompts"] if c in abl]
display(abl[cols])
abl.to_csv(OUT / "ablations.csv")
view = abl.drop(index="baseline")
fig, ax = plt.subplots(figsize=(12, 0.32 * len(view) + 1.5))
y = np.arange(len(view))
ax.barh(y, view["delta"].astype(float), xerr=[view["delta"] - view["ci_low"], view["ci_high"] - view["delta"]], color="#4C78A8")
ax.set_yticks(y, view.index); ax.axvline(0, color="black", lw=1); ax.invert_yaxis()
ax.set(title=f"Δτ относительно upstream (τ={r['baseline']['tau']:.3f}); усы — 95% paired bootstrap", xlabel="Δτ")
fig.tight_layout(); fig.savefig(OUT / "ablations.png", dpi=150); plt.show()
'''),
]

B_CELLS = [
    ("md", """# EAGLE-3: обучение альтернативных fusion-модулей при замороженном драфте (фаза B)

Пара: `Qwen/Qwen3-1.7B` + `AngelSlim/Qwen3-1.7B_eagle3`, EAGLE commit `cb7e084`. Декодер, нормы и lm_head драфта **заморожены**; обучается только модуль, который превращает hidden states target в вход драфта (в upstream — `Linear(3H→H)` над слоями 2, 14, 25).

Все варианты видят одни и те же последовательности UltraChat в одном порядке; target считается один раз на последовательность. Цель — step-0 loss EAGLE-3: soft cross-entropy к распределению target на словаре драфта. Оценка: step-0 accuracy на ответах target и реальная acceptance length τ при спекулятивной генерации (temperature 0), Δτ к официальному fc с paired bootstrap.

Варианты: контроль (тот же fc с нуля и дообучение официального), per-source RMSNorm, замена верхнего слоя на последний, 4 слоя, обучаемая смесь всех слоёв (3 слота × softmax по 29 слоям, +87 параметров), token-wise роутер по слоям (soft MoE, +0,18M), low-rank и low-rank MoE с тем же числом параметров, что у fc.

Ограничение: от официального драфта унаследованы TTT-обучение и латентное пространство, а бюджет здесь — несколько миллионов токенов. Поэтому сравнивать надо варианты между собой и с `fc3_scratch`, а не с абсолютным τ официального fc."""),
    ("code", SETUP),
    ("code", COMMON_CFG + '''import phase_b
CFG_B = dict(COMMON, out=str(OUT), n_train_seqs=8000, skip=2000, epochs=1, accum=8, lr=1e-3,
             fast_lr_mult=10.0, lr_mult={"fc3_official_ft": 0.1}, warmup_frac=0.03, clip=1.0,
             aux_coef=0.01, log_every=25, variants=None, max_train_seconds=5 * 3600)
result = phase_b.run(CFG_B)
'''),
    ("md", "## Результаты"),
    ("code", '''import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
r = json.loads((OUT / "phase_b.json").read_text())
off = r["tau"]["official"]["tau"]
rows = []
for name in ["official", "official_via_adapter"] + list(r["variants"]):
    t = r["tau"].get(name, {})
    s = r["step0_eval"].get(name, {})
    v = r["variants"].get(name, {})
    rows.append({"variant": name, "params": v.get("params", r["official_fc_params"]), "step0_acc": s.get("acc"),
                 "step0_loss": s.get("loss"), "tau": t.get("tau"), "delta_tau": (t.get("tau") or np.nan) - off,
                 "ci95": t.get("delta_ci95"), "mismatch": t.get("mismatched_prompts"), "note": v.get("note", "")})
table = pd.DataFrame(rows).set_index("variant")
display(table)
table.to_csv(OUT / "summary.csv")
print("optimizer steps:", r["optimizer_steps"], "/", r["planned_steps"], "stopped early:", r["stopped_early"],
      "train s:", round(r["train_seconds"]), "target fwd s:", round(r["target_forward_seconds"]))
fig, ax = plt.subplots(figsize=(11, 5))
for name, v in r["variants"].items():
    c = v["curve"]
    ax.plot([p["step"] for p in c], [p["acc"] for p in c], label=name)
ax.axhline(r["step0_eval"]["official"]["acc"], color="black", ls="--", label="official (held-out)")
ax.set(title="Step-0 accuracy на обучении", xlabel="optimizer step", ylabel="acc"); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "curves.png", dpi=150); plt.show()
mixes = [(n, v["mix_weights_static"]) for n, v in r["variants"].items() if "mix_weights_static" in v]
mixes += [(n + " (token avg)", v["router_mean_weights"]) for n, v in r["variants"].items() if "router_mean_weights" in v]
if mixes:
    fig, axes = plt.subplots(len(mixes), 1, figsize=(12, 1.6 * len(mixes) + 1))
    for ax, (n, w) in zip(np.atleast_1d(axes), mixes):
        im = ax.imshow(np.array(w), aspect="auto", cmap="magma", vmin=0)
        ax.set(title=n, ylabel="слот", xticks=range(len(w[0]))); fig.colorbar(im, ax=ax)
    axes[-1].set_xlabel("слой target")
    fig.tight_layout(); fig.savefig(OUT / "mix_weights.png", dpi=150); plt.show()
for n, v in r["variants"].items():
    if "expert_load" in v: print(n, "expert load:", np.round(v["expert_load"], 3))
'''),
]

B2_CELLS = [
    ("md", """# EAGLE-3 fusion: раунд 2 — проверка находок раунда 1

Тот же рецепт, что в `eagle3-fusion-training`, но **другая порция UltraChat** (skip=12000). Вопросы:
1. Выученные смеси слоёв ушли от слоёв 2 и 14 к зоне 18–22 и слою 25. Дают ли фиксированные тройки с теми же параметрами, что у upstream, такой же выигрыш: (2,20,25), (14,20,25), (11,20,25), (19,22,25)?
2. Нужен ли нижний слой вообще: пара (20,25) и один слой 25.
3. Воспроизводится ли +0.8 τ от дообучения официального fc на других данных.
4. Run-to-run детерминизм официального τ и τ на общем префиксе текста (до первого расхождения с baseline)."""),
    ("code", SETUP),
    ("code", COMMON_CFG + '''import phase_b
CFG_B = dict(COMMON, out=str(OUT), n_train_seqs=8000, skip=12000, epochs=1, accum=8, lr=1e-3,
             fast_lr_mult=10.0, lr_mult={"fc3_official_ft": 0.1}, warmup_frac=0.03, clip=1.0,
             aux_coef=0.01, log_every=25, max_train_seconds=5 * 3600, repeat_official=True,
             variants=["fc3_norm", "fc3_official_ft", "mix3_init"],
             custom_variants=[["tri_2_20_25", [2, 20, 25]], ["tri_14_20_25", [14, 20, 25]],
                              ["tri_11_20_25", [11, 20, 25]], ["tri_19_22_25", [19, 22, 25]],
                              ["pair_20_25", [20, 25]], ["single_25", [25]]])
result = phase_b.run(CFG_B)
'''),
    ("md", "## Результаты"),
    B_CELLS[-1],
    ("code", '''for name, t in r["tau"].items():
    print(f"{name:22s} tau={t['tau']:.3f} prefix_tau={t.get('tau_prefix', float('nan')):.3f} "
          f"mismatch={t.get('mismatched_prompts')} prefix_pair={t.get('prefix_pair_official_vs_variant')}")
'''),
]

for slug, title, cells, folder in [
    ("eagle3-fusion-layer-analysis", "EAGLE3 Fusion Layer Analysis", A_CELLS, "kaggle_a"),
    ("eagle3-fusion-training", "EAGLE3 Fusion Training", B_CELLS, "kaggle_b"),
    ("eagle3-fusion-training-round2", "EAGLE3 Fusion Training Round2", B2_CELLS, "kaggle_b2"),
]:
    out = HERE / folder
    out.mkdir(exist_ok=True)
    nb = notebook(cells)
    nbformat.write(nb, out / f"{slug}.ipynb")
    meta = {"id": f"{USER}/{slug}", "title": title, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
            "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(out / f"{slug}.ipynb", (out / f"{slug}.ipynb").stat().st_size)
(HERE / "payload_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
