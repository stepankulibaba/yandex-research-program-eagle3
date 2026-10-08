"""Phase A: analysis of the released fusion and inference-only ablations (no training)."""
import json
import time
from pathlib import Path

import numpy as np
import torch

import layer_lab as lab


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(cfg):
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    lab.patch_eagle(cfg["eagle_dir"])
    lab.seed_all(cfg["seed"])
    model = lab.load_pair(cfg["base"], cfg["draft"], cfg["total_token"], cfg["depth"], cfg["top_k"])
    tok = model.get_tokenizer()
    N = len(model.base_model.model.layers)
    H = model.base_model.config.hidden_size
    tri = lab.default_triplet(N)
    W = model.ea_layer.fc.weight.detach().clone()
    fc_orig = model.ea_layer.fc
    assert W.shape == (H, 3 * H), W.shape
    result = {"config": {k: v for k, v in cfg.items()}, "n_layers": N, "hidden": H, "triplet": tri}
    log(f"loaded: N={N} H={H} triplet={tri}")

    prompts = lab.eval_prompts(cfg["eagle_dir"], tok, cfg["prompt_counts"])
    base_rows, base_out = lab.run_tau(model, prompts, cfg["max_new_tokens"])
    result["baseline"] = lab.tau_summary(base_rows)
    log(f"baseline {result['baseline']}")

    # The patched capture with an explicit triplet must reproduce upstream exactly.
    lab.set_capture(model, tri)
    check_rows, _ = lab.run_tau(model, prompts[:3], cfg["max_new_tokens"], reference=base_out)
    lab.set_capture(model, None)
    base_sub = {r["id"]: r for r in base_rows}
    result["patch_check"] = [{"id": r["id"], "mismatch": r["mismatch"],
                              "same_cycles": r["cycles"] == base_sub[r["id"]]["cycles"]} for r in check_rows]
    log(f"patch check {result['patch_check']}")

    # ---------------- hidden states ----------------
    eval_seqs = []
    for p in prompts:
        gen = base_out[p["id"]]
        ids = torch.cat([p["ids"][0], gen])
        mask = torch.cat([torch.zeros(p["ids"].shape[1], dtype=torch.long), torch.ones(gen.numel(), dtype=torch.long)])
        eval_seqs.append({"ids": ids, "mask": mask})
    train_seqs = lab.ultrachat_sequences(tok, cfg["n_train_seqs"], max_len=cfg["max_len"])
    log(f"ultrachat sequences: {len(train_seqs)}")
    train = lab.collect_positions(model, train_seqs, cfg["train_positions"])
    test = lab.collect_positions(model, eval_seqs, cfg["test_positions"])
    log(f"positions train={train['states'].shape} test={test['states'].shape}")
    torch.cuda.empty_cache()

    result["norms_train"] = lab.layer_norm_stats(train["states"])
    result["norms_test"] = lab.layer_norm_stats(test["states"])
    result["fc_decomposition"] = lab.fc_decomposition(W, test["states"], tri)
    log(f"fc shares {result['fc_decomposition']['contribution_energy_share']}")
    result["cka"] = lab.linear_cka(train["states"], n=cfg["cka_tokens"])
    log("cka done")

    g_tr = lab.official_fused(W, train["states"], tri)
    g_te = lab.official_fused(W, test["states"], tri)
    r2 = {}
    for layer in range(N + 1):
        r2[str(layer)] = lab.ridge_r2(train["states"][:, layer].float(), g_tr, test["states"][:, layer].float(), g_te)
    r2["triplet"] = lab.ridge_r2(train["states"][:, list(tri)].float().flatten(1), g_tr,
                                 test["states"][:, list(tri)].float().flatten(1), g_te)
    result["ridge_r2_to_fused"] = r2
    log(f"ridge r2 {r2}")
    del g_tr, g_te

    # ---------------- future-token probes ----------------
    t2d = model.ea_layer.t2d.bool().cpu()
    draft_index = torch.cumsum(t2d.long(), 0) - 1

    def to_draft(y):
        return torch.where(t2d[y], draft_index[y], torch.full_like(y, -100))

    head_w = model.base_model.lm_head.weight[t2d.cuda()].float()
    norm = model.base_model.model.norm
    probes = {}
    layers = cfg.get("probe_layers") or list(range(N + 1))
    for layer in layers:
        for k in range(1, cfg["horizons"] + 1):
            per_epoch, lens = lab.train_future_probe(
                train["states"][:, layer], to_draft(train["labels"][:, k - 1]),
                test["states"][:, layer], to_draft(test["labels"][:, k - 1]),
                head_w, norm, epochs=cfg["probe_epochs"])
            probes[f"{layer}:{k}"] = {"probe_acc": per_epoch[-1], "probe_acc_per_epoch": per_epoch,
                                      "lens_acc": lens}
        log(f"probe layer {layer}: " + ", ".join(f"k{k}={probes[f'{layer}:{k}']['probe_acc']:.3f}"
                                                 for k in range(1, cfg["horizons"] + 1)))
    result["future_probes"] = probes
    lab.dump(result, out / "phase_a.json")

    # ---------------- inference ablations ----------------
    means = train["states"].float().mean(0)
    rms_mean = result["norms_train"]["rms_mean"]
    del train, test
    torch.cuda.empty_cache()

    conditions = [("mean_all", None, [(s, "mean") for s in range(3)])]
    conditions += [(f"mean_{name}", None, [(s, "mean")]) for s, name in enumerate(["low", "mid", "high"])]
    for slot, candidates in enumerate(cfg["shifts"]):
        for new in candidates:
            if new == tri[slot]:
                continue
            capture = list(tri)
            capture[slot] = new
            if capture != sorted(set(capture)):
                continue
            scale = rms_mean[tri[slot]] / rms_mean[new]
            conditions.append((f"shift{slot}_{tri[slot]}to{new}", tuple(capture), [(slot, ("scale", scale))]))
            if new in cfg.get("unscaled_shifts", {}).get(str(slot), []):
                conditions.append((f"shift{slot}_{tri[slot]}to{new}_noscale", tuple(capture), []))
    for rank in cfg["fc_ranks"]:
        conditions.append((f"fc_rank{rank}", None, [("rank", rank)]))

    tau_rows = [dict(r, condition="baseline") for r in base_rows]
    summaries = {"baseline": result["baseline"]}
    for name, capture, edits in conditions:
        mean_vec = None
        slot = scale = rank = None
        if edits and edits[0][0] == "rank":
            rank = edits[0][1]
        elif len(edits) == 3:  # all slots replaced by their means
            mean_vec = torch.cat([means[tri[s]] for s in range(3)])
        elif edits:
            slot, what = edits[0]
            if what == "mean":
                mean_vec = means[tri[slot]]
            else:
                scale = what[1]
        if len(edits) == 3:
            module = _AllMean(W, mean_vec)
        else:
            module = lab.SlotEdit(W, mean=mean_vec, slot=slot, scale=scale, rank=rank)
        model.ea_layer.fc = module
        lab.set_capture(model, capture)
        try:
            rows, _ = lab.run_tau(model, prompts, cfg["max_new_tokens"], reference=base_out)
        finally:
            model.ea_layer.fc = fc_orig
            lab.set_capture(model, None)
        summaries[name] = lab.tau_summary(rows)
        summaries[name]["delta_ci95"] = lab.paired_bootstrap(base_rows, rows)
        summaries[name]["capture"] = list(capture) if capture else list(tri)
        tau_rows += [dict(r, condition=name) for r in rows]
        log(f"{name}: tau={summaries[name]['tau']:.3f} ci={summaries[name]['delta_ci95']}")
        result["ablations"] = summaries
        lab.dump(result, out / "phase_a.json")
        lab.dump(tau_rows, out / "tau_rows.json")
    log("phase A finished")
    return result


class _AllMean(torch.nn.Module):
    def __init__(self, W, mean):
        super().__init__()
        self.register_buffer("g", (W.float().cuda() @ mean.float().cuda()))

    def forward(self, x):
        return self.g.expand(*x.shape[:-1], -1).to(x.dtype)
