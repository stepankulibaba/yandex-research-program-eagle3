"""Figures and numbers for the router report.

    python code/make_report_figures.py qwen3-1.7b | dsl-8b
needs results/<pair>/{history.json,final_eval.pt}; writes figures/[<pair>/]r*_*.png and results/<pair>/report_stats.json
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent.parent      # the router folder
import sys
PAIR = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
R = HERE / "results" / PAIR
FIG = HERE / "figures" / ("" if PAIR == "qwen3-1.7b" else PAIR)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = {"fixed_raw": "#7A7A7A", "fixed_norm": "#2F6DB5", "fixed_alt": "#5FA8D3", "static_top3": "#2E7D4F",
       "static_top2": "#8BC79A", "token_top3": "#B23A48", "token_top2": "#E89A52"}
TRI = {"qwen3-1.7b": ("(2, 14, 25)", "(14, 20, 25)"), "dsl-8b": ("(2, 16, 29)", "(16, 22, 29)")}[PAIR]
LAB = {"fixed_raw": f"оригинал {TRI[0]}", "fixed_norm": "оригинал + норм.", "fixed_alt": f"{TRI[1]} + норм.",
       "static_top3": "статический top-3", "static_top2": "статический top-2", "token_top3": "роутер top-3",
       "token_top2": "роутер top-2"}
SUF = "" if PAIR == "qwen3-1.7b" else "_s"
for k in ("token_top3", "token_top2"):
    COL[k + SUF], LAB[k + SUF] = COL[k], LAB[k]
ORDER = ["fixed_raw", "fixed_norm", "fixed_alt", "static_top3", "static_top2", "token_top3" + SUF, "token_top2" + SUF]
NOISE_END = 0.9      # epochs: noise decays over the first 30% of 3 epochs

H = json.loads((R / "history.json").read_text())["evals"]
F = torch.load(R / "final_eval.pt", weights_only=False)
per_seq, routing, N = F["per_seq"], F["routing"], F["n_layers"]
L = N + 1
stats = {}


# ------------------------------------------------------------------ tau with paired bootstrap on 240 questions
def tau(rows, idx=None):
    rows = rows if idx is None else [rows[i] for i in idx]
    return 1 + sum(r["lead_sum"] for r in rows) / sum(r["n"] for r in rows)


def boot_diff(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ls_a = np.array([r["lead_sum"] for r in a]); n_a = np.array([r["n"] for r in a])
    ls_b = np.array([r["lead_sum"] for r in b]); n_b = np.array([r["n"] for r in b])
    idx = rng.integers(0, len(a), (n, len(a)))
    d = ls_a[idx].sum(1) / n_a[idx].sum(1) - ls_b[idx].sum(1) / n_b[idx].sum(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


names = ORDER + [n for n in per_seq if "@" in n]
final = {}
for n in names:
    rows = per_seq[n]
    cyc = sum(r["cyc_tokens"] for r in rows) / sum(r["cyc_n"] for r in rows)
    by_bench = {b: tau([r for r in rows if r["bench"] == b]) for b in ("mt_bench", "gsm8k", "humaneval")}
    depth = (np.array([r["by_depth"] for r in rows]).sum(0) / sum(r["n"] for r in rows)).tolist()
    final[n] = {"tau": tau(rows), "tau_cycles": cyc, "by_bench": by_bench, "accept_by_depth": depth,
                "d_raw": tau(rows) - tau(per_seq["fixed_raw"]), "ci_raw": boot_diff(rows, per_seq["fixed_raw"]),
                "d_alt": tau(rows) - tau(per_seq["fixed_alt"]), "ci_alt": boot_diff(rows, per_seq["fixed_alt"])}
for n in routing:
    m = f"{n}@modal"
    final[m]["d_self"] = final[m]["tau"] - final[n]["tau"]
    final[m]["ci_self"] = boot_diff(per_seq[m], per_seq[n])
stats["final"] = final
stats["modal"] = {k: list(v) for k, v in F["modal"].items()}

# ------------------------------------------------------------------ token routers: how token-dependent is the choice
ent = lambda p: float(-(p[p > 0] * np.log2(p[p > 0])).sum())  # noqa: E731
rstats = {}
for n, recs in routing.items():
    sets = [tuple(s.tolist()) for r in recs for s in r["sets"]]
    kinds = [k for r in recs for k in r["kind"]]
    conf = np.concatenate([r["conf"].numpy() for r in recs])
    pos = np.concatenate([r["pos"].numpy() for r in recs])
    bench = [r["bench"] for r in recs for _ in range(len(r["kind"]))]
    acc1 = np.concatenate([r["acc1"].numpy() for r in recs])
    cnt = Counter(sets)
    total = len(sets)
    p = np.array(list(cnt.values())) / total
    cats = {"kind": kinds, "conf": ["hi" if c > 0.9 else ("mid" if c > 0.5 else "lo") for c in conf],
            "pos": ["<64" if q < 64 else ("<256" if q < 256 else ">=256") for q in pos], "bench": bench}
    mi = {}
    for cname, labels in cats.items():
        h_cond = 0.0
        for g, c in Counter(labels).items():
            sub = Counter(s for s, lab in zip(sets, labels) if lab == g)
            h_cond += c / total * ent(np.array(list(sub.values())) / c)
        mi[cname] = ent(p) - h_cond
    # per-set acceptance of the first guess
    acc_by_set = {}
    for s, c in cnt.most_common(6):
        mask = np.array([x == s for x in sets])
        acc_by_set[str(s)] = {"share": c / total, "acc1": float(acc1[mask].mean())}
    layer_share = np.bincount(np.array(sets).ravel(), minlength=L) / (total * len(sets[0]))
    rstats[n] = {"n_tokens": total, "n_sets": len(cnt), "top_sets": [[list(s), c / total] for s, c in cnt.most_common(8)],
                 "entropy_bits": ent(p), "max_entropy_bits": float(np.log2(min(len(cnt), total))),
                 "mutual_info_bits": mi, "acc_by_set": acc_by_set, "layer_share": layer_share.tolist(),
                 "by_cat": {cname: {g: (np.bincount(np.array([s for s, lab in zip(sets, labels) if lab == g]).ravel(),
                                                     minlength=L) / (c * len(sets[0]))).tolist()
                                    for g, c in Counter(labels).items()} for cname, labels in cats.items()}}
stats["routing"] = rstats

# ------------------------------------------------------------------ figure 1: tau over training
ep = [e["epoch"] for e in H]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
base = np.array([e["res"]["fixed_raw"]["tau_chain"] for e in H])
for n in ORDER:
    t = np.array([e["res"][n]["tau_chain"] for e in H])
    ax[0].plot(ep[1:], t[1:], marker="o", ms=2.5, color=COL[n], label=LAB[n])
    ax[1].plot(ep[1:], (t - base)[1:], marker="o", ms=2.5, color=COL[n])
for a in ax:
    a.axvspan(0, NOISE_END, color="#999", alpha=.08)
    a.grid(alpha=.2)
    a.set_xlabel("эпоха")
ax[0].text(0.05, ax[0].get_ylim()[1] - 0.05, "шум роутера", fontsize=8, color="#666", va="top")
ax[0].set(ylabel="τ (цепочка до 6 шагов, 96 вопросов)", title="Качество драфта по ходу обучения")
ax[1].axhline(0, color="black", lw=.8)
ax[1].set(ylabel="Δτ к оригиналу", title="Разница с оригинальной тройкой слоёв")
ax[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "r1_tau_training.png", dpi=150)

# ------------------------------------------------------------------ figure 2: static choice over training
fig, ax = plt.subplots(2, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1]})
for row, n in enumerate(["static_top3", "static_top2"]):
    probs = np.array([e["res"][n]["static_probs"] for e in H[1:]])        # [evals, L]
    logit = np.log(probs) - np.log(probs).mean(1, keepdims=True)
    im = ax[row, 0].imshow(logit.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-2, vmax=2,
                           extent=[ep[1] - .02, ep[-1] + .02, -.5, L - .5])
    ax[row, 0].set(ylabel="слой target", title=f"{LAB[n]}: оценка слоя (центрированная)")
    fig.colorbar(im, ax=ax[row, 0], fraction=0.03)
    k = 3 if n.endswith("3") else 2
    chosen = np.argsort(-probs, 1)[:, :k]
    for j in range(k):
        ax[row, 1].scatter(ep[1:], chosen[:, j], color=COL[n], s=14)
    ax[row, 1].set(ylim=(-0.5, L - .5), ylabel="выбранные слои", title=f"{LAB[n]}: какие слои выбраны")
    ax[row, 1].axhspan(N * 0.65, N * 0.75, color="#E89A52", alpha=.12)
    ax[row, 1].axhline(N - 3, color="#2F6DB5", ls=":", lw=1)
    for a in ax[row]:
        a.axvline(NOISE_END, color="#666", ls="--", lw=.8)
        a.set_xlabel("эпоха")
    ax[row, 1].grid(alpha=.2)
ax[0, 1].text(ep[-1], N - 3 + .4, "N−3", color="#2F6DB5", fontsize=8, ha="right")
ax[0, 1].text(ep[-1], N * 0.7 - 1.6, "≈0,7N", color="#B0662B", fontsize=8, ha="right")
fig.tight_layout()
fig.savefig(FIG / "r2_static_choice.png", dpi=150)

# ------------------------------------------------------------------ figure 3: token routers over training
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for a, n in zip(ax, [f"token_top3{SUF}", f"token_top2{SUF}"]):
    share = np.array([e["res"][n]["layer_share"] for e in H[1:]]).T
    im = a.imshow(share, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=.5,
                  extent=[ep[1] - .02, ep[-1] + .02, -.5, L - .5])
    a.axvline(NOISE_END, color="white", ls="--", lw=.8)
    a.set(xlabel="эпоха", ylabel="слой target", title=f"{LAB[n]}: доля выборов слоя")
    fig.colorbar(im, ax=a, fraction=0.03)
fig.tight_layout()
fig.savefig(FIG / "r3_token_choice.png", dpi=150)

# ------------------------------------------------------------------ figure 4: token routers by category (final)
CAT = [("kind", "тип токена", ["space", "punct", "digit", "word_start", "word_cont"]),
       ("conf", "уверенность target", ["hi", "mid", "lo"]), ("pos", "позиция", ["<64", "<256", ">=256"]),
       ("bench", "задача", ["mt_bench", "gsm8k", "humaneval"])]
fig, ax = plt.subplots(2, 4, figsize=(15, 6.4), sharey=True)
for row, n in enumerate([f"token_top3{SUF}", f"token_top2{SUF}"]):
    for col, (c, title, groups) in enumerate(CAT):
        M = np.array([rstats[n]["by_cat"][c][g] for g in groups]).T
        ax[row, col].imshow(M[12:], aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=.5,
                            extent=[-.5, len(groups) - .5, 11.5, L - .5])
        ax[row, col].set_xticks(range(len(groups)), groups, rotation=25, ha="right", fontsize=8)
        ax[row, col].set_title(f"{title}\nI = {rstats[n]['mutual_info_bits'][c]:.3f} бит", fontsize=9)
    ax[row, 0].set_ylabel(f"{LAB[n]}\nслой target")
fig.suptitle("Пословные роутеры в конце обучения: доля выбора слоя по категориям токенов (240 вопросов); "
             "I — взаимная информация выбора и категории", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "r4_token_by_category.png", dpi=150)

# ------------------------------------------------------------------ figure 5: final tau with CIs + acceptance by depth
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
show = ORDER + [f"token_top3{SUF}@modal", f"token_top2{SUF}@modal"]
y = np.arange(len(show))[::-1]
for yi, n in zip(y, show):
    d, ci = final[n]["d_raw"], final[n]["ci_raw"]
    c = COL.get(n.split("@")[0])
    ax[0].barh(yi, d, color=c, alpha=.45 if "@" in n else 1, hatch="//" if "@" in n else None)
    ax[0].errorbar(d, yi, xerr=[[d - ci[0]], [ci[1] - d]], color="black", capsize=3, lw=1)
    ax[0].text(max(d, 0) + 0.02, yi, f"τ={final[n]['tau']:.2f}", va="center", fontsize=8)
lbl = [LAB.get(n, LAB.get(n.split("@")[0], n) + ": всем токенам один набор") for n in show]
ax[0].set_yticks(y, lbl, fontsize=8)
ax[0].axvline(0, color="black", lw=.8)
ax[0].set(xlabel="Δτ к оригиналу, 95% CI (240 вопросов)", title="Итог после 3 эпох")
for n in ORDER:
    ax[1].plot(range(1, 7), final[n]["accept_by_depth"], marker="o", ms=3, color=COL[n], label=LAB[n])
ax[1].set(xlabel="шаг драфта", ylabel="доля позиций, где приняты все шаги до этого",
          title="Принятие по глубине цепочки")
ax[1].legend(frameon=False, fontsize=8)
ax[1].grid(alpha=.2)
fig.tight_layout()
fig.savefig(FIG / "r5_final.png", dpi=150)

(R / "report_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
for n in show:
    f = final[n]
    print(f"{n:20s} tau={f['tau']:.3f} cyc={f['tau_cycles']:.3f} d_raw={f['d_raw']:+.3f} {np.round(f['ci_raw'], 3)} "
          f"d_alt={f['d_alt']:+.3f} {np.round(f['ci_alt'], 3)} " + " ".join(f"{b}={v:.2f}" for b, v in f["by_bench"].items())
          + (f" d_self={f['d_self']:+.3f} {np.round(f['ci_self'], 3)}" if "d_self" in f else ""))
for n, r in rstats.items():
    print(n, "sets", r["n_sets"], "H", round(r["entropy_bits"], 2), "MI", {k: round(v, 3) for k, v in r["mutual_info_bits"].items()})
    print("   top", [(s, round(c, 3)) for s, c in r["top_sets"]])
    print("   acc", r["acc_by_set"])
