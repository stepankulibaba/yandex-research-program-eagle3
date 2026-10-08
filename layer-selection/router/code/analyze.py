"""Figures and a text summary for one training run.

    python analyze.py results/qwen3-1.7b/history.json [figures/qwen3-1.7b]

    tau_curves.png                  tau of every variant over training, and its gain over the original EAGLE-3
    layer_choice_over_training.png  for each routed variant: how often each target layer is chosen, over time
    by_category_<variant>.png       per-token routers at the end: chosen layers by task / token type /
                                    target confidence / position in the response
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"fixed_raw": "#7A7A7A", "fixed_norm": "#2F6DB5", "fixed_alt": "#5FA8D3", "static_top3": "#2E7D4F",
          "static_top2": "#8BC79A", "token_top3": "#B23A48", "token_top2": "#E89A52"}
LABELS = {"fixed_raw": "оригинал (2, N/2, N−3)", "fixed_norm": "оригинал + RMSNorm",
          "fixed_alt": "(N/2, 0.7N, N−3) + RMSNorm", "static_top3": "статический top-3",
          "static_top2": "статический top-2", "token_top3": "роутер top-3", "token_top2": "роутер top-2"}
CATEGORY_TITLES = {"bench": "задача", "token": "тип токена", "conf": "уверенность target", "pos": "позиция"}


def plot_tau(evals, variants, out):
    epochs = [e["epoch"] for e in evals]
    base = [e["res"]["fixed_raw"]["tau_chain"] for e in evals]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for n in variants:
        tau = [e["res"][n]["tau_chain"] for e in evals]
        ax[0].plot(epochs, tau, marker="o", ms=3, color=COLORS.get(n), label=LABELS.get(n, n))
        ax[1].plot(epochs[1:], np.subtract(tau, base)[1:], marker="o", ms=3, color=COLORS.get(n))
    ax[0].set(xlabel="эпоха", ylabel="цепочечная τ (до 6 шагов)", title="Качество драфта по ходу обучения")
    ax[1].axhline(0, color="black", lw=.8)
    ax[1].set(xlabel="эпоха", ylabel="Δτ к оригиналу", title="Разница с оригинальной тройкой слоёв")
    ax[0].legend(frameon=False, fontsize=8)
    for a in ax:
        a.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "tau_curves.png", dpi=150)


def plot_choice_over_time(evals, routed, out):
    epochs = [e["epoch"] for e in evals]
    fig, axes = plt.subplots(len(routed), 1, figsize=(12, 2.2 * len(routed) + 0.6), sharex=True)
    for a, n in zip(np.atleast_1d(axes), routed):
        share = np.array([e["res"][n]["layer_share"] for e in evals[1:]]).T     # [layers, evals]
        im = a.imshow(share, aspect="auto", origin="lower", cmap="magma", vmin=0,
                      extent=[epochs[1] - 0.05, epochs[-1] + 0.05, -0.5, share.shape[0] - 0.5])
        a.set(ylabel="слой target", title=f"{LABELS.get(n, n)}: доля выборов слоя")
        fig.colorbar(im, ax=a, fraction=0.02)
    np.atleast_1d(axes)[-1].set_xlabel("эпоха")
    fig.tight_layout()
    fig.savefig(out / "layer_choice_over_training.png", dpi=150)


def plot_by_category(result, name, out):
    by = result["by"]
    fig, axes = plt.subplots(1, len(CATEGORY_TITLES), figsize=(16, 3.8), sharey=True)
    for a, key in zip(axes, CATEGORY_TITLES):
        groups = sorted(by[key])
        a.imshow(np.array([by[key][g] for g in groups]).T, aspect="auto", origin="lower", cmap="magma", vmin=0)
        a.set_xticks(range(len(groups)), groups, rotation=30, ha="right", fontsize=8)
        a.set_title(CATEGORY_TITLES[key])
    axes[0].set_ylabel("слой target")
    fig.suptitle(f"{LABELS.get(name, name)}: какие слои выбираются (конец обучения)")
    fig.tight_layout()
    fig.savefig(out / f"by_category_{name}.png", dpi=150)


def top_layers(share, k=4):
    share = np.array(share)
    return ", ".join(f"{int(i)}:{share[i]:.2f}" for i in np.argsort(share)[::-1][:k] if share[i] > 0)


def main(history_path, out):
    evals = json.loads(Path(history_path).read_text())["evals"]
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    final = evals[-1]["res"]
    variants = list(final)
    routed = [n for n in variants if "layer_share" in final[n]]

    plot_tau(evals, variants, out)
    if routed:
        plot_choice_over_time(evals, routed, out)
    for n in routed:
        if n.startswith("token"):
            plot_by_category(final[n], n, out)

    print(f"step {evals[-1]['step']}:")
    for n in variants:
        r = final[n]
        line = (f"  {n:12s} tau_chain={r['tau_chain']:.3f} tau_cycles={r['tau_cycles']:.3f} "
                f"step1={r['accept_by_depth'][0]:.3f} step6={r['accept_by_depth'][-1]:.3f}")
        if n in routed:
            line += f"   layers {top_layers(r['layer_share'])}"
        print(line)
    for n in routed:
        print(f"\n{n}: chosen layers over training")
        for e in evals[1:]:
            print(f"  step {e['step']:6d}: {top_layers(e['res'][n]['layer_share'])}")
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    path = Path(sys.argv[1])
    main(path, Path(sys.argv[2]) if len(sys.argv) > 2 else path.parent / "figures")
