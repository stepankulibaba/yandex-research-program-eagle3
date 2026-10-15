"""Tables and figures from results/<run>/final.json, history.json and results/head_speed.json.

    python analyze.py en          -> results/en/summary.md, results/en/fig_*.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import K_VALUES, LANGS, RESULTS

COLORS = {"official": "#8a8a8a", "static32k": "#eb6834", "full": "#1baf7a", "specvocab": "#2a78d6"}
NAMES = {"official": "официальный драфт (без дообучения)", "static32k": "static32k: EAGLE-3, словарь 32K",
         "full": "full: весь словарь", "specvocab": "SpecVocab, k=2048"}


def tau(res, key, group, metric="tau_chain"):
    return (res.get(key, {}).get(group) or {}).get(metric)


def main(run):
    out = RESULTS / run
    final = json.loads((out / "final.json").read_text())
    res = final["res"]
    variants = [v for v in ["official", "static32k", "full", "specvocab"] if f"{v}|exact" in res]
    keys = {v: (f"{v}|k=2048" if v == "specvocab" else f"{v}|exact") for v in variants}
    groups = ["en/mt_bench", "en/gsm8k", "en/humaneval"] + LANGS
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1. tau by language and benchmark
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        ys = [tau(res, keys[v], g) or 0 for g in groups]
        ax.bar([x + (i - (len(variants) - 1) / 2) * width for x in range(len(groups))], ys, width,
               color=COLORS[v], label=NAMES[v])
    ax.set_xticks(range(len(groups)), [g.replace("en/", "") for g in groups])
    ax.set_ylabel("τ (цепочка, 6 шагов)")
    ax.set_title("τ по бенчмаркам и языкам", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_tau_by_language.png", dpi=140)

    # 2. SpecVocab: tau and recall vs k, per language
    if "specvocab|exact" in res:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        for g in LANGS:
            a1.plot(K_VALUES, [tau(res, f"specvocab|k={k}", g) for k in K_VALUES], "o-", label=g)
            a2.plot(K_VALUES, [(res.get(f"specvocab|k={k}", {}).get(g) or {}).get("recall_by_step", [None])[0]
                               for k in K_VALUES], "o-", label=g)
        for a in (a1, a2):
            a.set_xscale("log", base=2)
            a.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
            a.set_xlabel("k — кандидатов на шаг")
        a1.set_ylabel("τ")
        a1.set_title("SpecVocab: τ от k", loc="left")
        a2.set_ylabel("доля шагов, где нужный токен среди кандидатов")
        a2.set_title("Полнота кандидатов (шаг 1)", loc="left")
        a2.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_specvocab_k.png", dpi=140)

    # 3. training curves on the watch set
    hist_path = out / "history.json"
    if hist_path.exists():
        hist = json.loads(hist_path.read_text())["evals"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for a, g in zip(axes, ["en", "ru"]):
            for v in variants:
                key = keys[v]
                pts = [(e["step"], tau(e["res"], key, g)) for e in hist if tau(e["res"], key, g) is not None]
                if pts:
                    a.plot(*zip(*pts), "o-", color=COLORS[v], label=NAMES[v], ms=3)
            a.set_title(f"τ во время обучения: {g}", loc="left")
            a.set_xlabel("шаг")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_training.png", dpi=140)

    # summary.md
    lines = [f"# SpecVocab на Qwen3-1.7B: результаты прогона `{run}`", "",
             f"Текстов в оценке: {final['texts']}. τ — средняя длина принятой цепочки из 6 шагов + 1 (жадная генерация).", "",
             "## τ по бенчмаркам и языкам", "",
             "| вариант | " + " | ".join(g.replace("en/", "") for g in groups) + " |",
             "|---" * (len(groups) + 1) + "|"]
    rows = [(NAMES[v], keys[v]) for v in variants]
    if "specvocab|exact" in res:
        rows.append(("SpecVocab без отбора (вся голова)", "specvocab|exact"))
    for name, key in rows:
        lines.append(f"| {name} | " + " | ".join(f"{tau(res, key, g):.2f}" if tau(res, key, g) else "—" for g in groups) + " |")
    lines += ["", "## Доля токенов ответа вне 32K-словаря официального драфта", "",
              "| " + " | ".join(LANGS) + " |", "|---" * len(LANGS) + "|",
              "| " + " | ".join(f"{final['oov_share'].get(g, 0):.1%}" for g in LANGS) + " |"]
    if "specvocab|exact" in res:
        lines += ["", "## SpecVocab: τ и полнота кандидатов (шаг 1) от k", "",
                  "| k | " + " | ".join(f"τ {g}" for g in LANGS) + " | " + " | ".join(f"полнота {g}" for g in LANGS) + " |",
                  "|---" * (2 * len(LANGS) + 1) + "|"]
        for k in K_VALUES:
            key = f"specvocab|k={k}"
            rec = [((res.get(key, {}).get(g) or {}).get("recall_by_step") or [0])[0] for g in LANGS]
            lines.append(f"| {k} | " + " | ".join(f"{tau(res, key, g):.2f}" for g in LANGS) + " | " +
                         " | ".join(f"{r:.1%}" for r in rec) + " |")
    speed = RESULTS / "head_speed.json"
    if speed.exists():
        sp = json.loads(speed.read_text())
        lines += ["", "## Стоимость головы драфта (GPU, обычный PyTorch, без fused-ядра статьи)", "",
                  "| голова | умножений на токен | B=1, мкс | B=60, мкс |", "|---|---|---|---|"]
        for name, macs in sp["multiply_adds"].items():
            lines.append(f"| {name} | {macs / 1e6:.1f} млн | {sp['results']['B=1'][name]:.0f} | {sp['results']['B=60'][name]:.0f} |")
    lines += ["", "![τ по языкам](fig_tau_by_language.png)", "", "![SpecVocab от k](fig_specvocab_k.png)", "",
              "![Обучение](fig_training.png)"]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "en")
