"""Train new draft inputs while the draft stays frozen, then measure tau with each of them.

    python train_fusions.py qwen3-1.7b      -> results/h200/qwen3-1.7b/fusion/phase_b.json, tau_rows.json, fusion_*.pt
    needs train_data.pt (gen_data.py)

Compared inputs (draft_inputs.build_variants and the extra layer sets below): the upstream layers with a fresh
matrix, with RMSNorm per layer, the official matrix fine-tuned, other layer sets, a learned mix of all layers,
a soft per-token router, low-rank and low-rank MoE inputs. Each is trained with the step-0 objective on the same
sequences in the same order (one target pass per sequence, shared by all variants).

Controls: the official matrix measured through the same adapter must give the official tau
("official_via_adapter"); each variant's tau is compared with the official one on the same prompts.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

from acceptance import measure_acceptance, paired_bootstrap, prefix_tau_pair, tau_summary
from config import BENCHES, EAGLE_DIR, PAIRS, PROTOCOL, generation_kwargs, log, results_dir, save_json
from draft_inputs import AllLayersAdapter, FixedFusion, LowRankMoEFusion, MixFusion, build_variants, count_params
from eagle_model import benchmark_prompts, default_triplet, load_eagle, prompt_plus_answer, set_capture
from target_states import draft_step0_logits, evaluate_step0, step0_batch, step0_loss


def make_config(pair, n_train=8000):
    """All settings of the run (saved into the result file)."""
    N = json.loads((PAIRS[pair]["target"] / "config.json").read_text())["num_hidden_layers"]
    q = round(0.7 * N)
    tri = default_triplet(N)
    return {"eagle_dir": str(EAGLE_DIR), "base": str(PAIRS[pair]["target"]), "draft": str(PAIRS[pair]["draft"]),
            "seed": 0, "total_token": PROTOCOL["total_token"], "depth": PROTOCOL["depth"], "top_k": PROTOCOL["top_k"],
            "max_new_tokens": PROTOCOL["max_new_tokens"], "gen_kwargs": generation_kwargs(pair),
            "prompt_counts": BENCHES, "max_len": 2048, "out": str(results_dir(pair) / "fusion"), "amp_dtype": "bfloat16",
            "train_data": str(results_dir(pair) / "train_data.pt"), "n_train_seqs": n_train, "skip": 0, "epochs": 1,
            "accum": 8,                      # sequences per optimizer step
            "lr": 1e-3, "fast_lr_mult": 10.0,  # mixing weights / router learn 10x faster
            "lr_mult": {"fc3_official_ft": 0.1}, "warmup_frac": 0.03, "clip": 1.0,
            "aux_coef": 0.01, "balanced_moe": 0.1,   # MoE load-balance coefficients
            "log_every": 25, "max_train_seconds": 3 * 3600, "repeat_official": False, "variants": None,
            "custom_variants": [[f"tri_{tri[1]}_{q}_{tri[2]}", [tri[1], q, tri[2]]],
                                [f"tri_{tri[0]}_{q}_{tri[2]}", [tri[0], q, tri[2]]],
                                [f"pair_{q}_{tri[2]}", [q, tri[2]]],
                                [f"single_{tri[2]}", [tri[2]]],
                                [f"single_{N}", [N]]]}


def all_variants(cfg, H, N, W):
    """build_variants + the extra layer sets + a load-balanced MoE. Construction order fixes the random init."""
    variants = build_variants(H, N, W)
    if cfg.get("variants"):
        variants = {k: v for k, v in variants.items() if k in cfg["variants"]}
    for name, layers in cfg.get("custom_variants", []):
        variants[name] = (FixedFusion(H, layers, norm=True), f"layers {tuple(layers)} + per-source RMSNorm; fc {len(layers)}H->H")
    if cfg.get("balanced_moe"):
        moe = variants["moe_lowrank"][0]
        bal = LowRankMoEFusion(H, moe.layers, moe.down.out_features, moe.E, moe.e_down.shape[-1])
        bal.aux_coef = cfg["balanced_moe"]
        variants["moe_lowrank_bal"] = (bal, f"as moe_lowrank, load-balance coefficient {cfg['balanced_moe']}")
    return variants


def make_trainer(module, note, name, cfg, lr_lambda, amp):
    module.cuda().train()
    mult = cfg.get("lr_mult", {}).get(name, 1.0)
    fast = [p for n, p in module.named_parameters() if n.startswith("logits") or n.startswith("router.")]
    slow = [p for n, p in module.named_parameters() if not (n.startswith("logits") or n.startswith("router."))]
    groups = [{"params": slow, "lr": cfg["lr"] * mult}]
    if fast:
        groups.append({"params": fast, "lr": cfg["lr"] * mult * cfg["fast_lr_mult"]})
    opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
    return {"module": module, "note": note, "opt": opt, "sched": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
            "scaler": torch.amp.GradScaler("cuda", enabled=amp == torch.float16), "params": count_params(module),
            "curve": [], "run_loss": 0.0, "run_acc": 0.0, "run_n": 0, "seconds": 0.0}


def train(model, trainers, train_seqs, cfg, amp):
    """One pass over the data; every variant sees the same sequences in the same order."""
    ea = model.ea_layer
    accum = cfg["accum"]
    total_steps = math.ceil(len(train_seqs) * cfg["epochs"] / accum)
    step = seq_count = 0
    target_seconds = 0.0
    started = time.time()
    stopped_early = False
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(train_seqs), generator=torch.Generator().manual_seed(cfg["seed"] + epoch)).tolist()
        for i in order:
            if time.time() - started > cfg["max_train_seconds"] and seq_count % accum == 0:
                stopped_early = True
                log(f"time budget reached after {step} optimizer steps")
                break
            t0 = time.perf_counter()
            states, target_p, pos_mask, next_ids = step0_batch(model, train_seqs[i])
            torch.cuda.synchronize()
            target_seconds += time.perf_counter() - t0
            seq_count += 1
            if pos_mask.sum() == 0:
                continue
            feats = states.float()
            for s in trainers.values():
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=amp):
                    logits = draft_step0_logits(ea, s["module"](feats), next_ids)
                loss, acc = step0_loss(logits, target_p, pos_mask)
                total = loss
                if isinstance(s["module"], LowRankMoEFusion):
                    total = total + (getattr(s["module"], "aux_coef", None) or cfg["aux_coef"]) * s["module"].aux_loss
                s["scaler"].scale(total / accum).backward()
                s["run_loss"] += loss.item()
                s["run_acc"] += acc.item()
                s["run_n"] += 1
                if seq_count % accum == 0:
                    s["scaler"].unscale_(s["opt"])
                    torch.nn.utils.clip_grad_norm_(s["module"].parameters(), cfg["clip"])
                    s["scaler"].step(s["opt"])
                    s["scaler"].update()
                    s["opt"].zero_grad(set_to_none=True)
                    s["sched"].step()
                torch.cuda.synchronize()
                s["seconds"] += time.perf_counter() - t0
            if seq_count % accum == 0:
                step += 1
                if step % cfg["log_every"] == 0 or step == total_steps:
                    line = []
                    for name, s in trainers.items():
                        n = max(s["run_n"], 1)
                        s["curve"].append({"step": step, "loss": s["run_loss"] / n, "acc": s["run_acc"] / n})
                        line.append(f"{name}={s['run_acc'] / n:.3f}")
                        s["run_loss"] = s["run_acc"] = 0.0
                        s["run_n"] = 0
                    log(f"step {step}/{total_steps} ({time.time() - started:.0f}s) acc: " + " ".join(line))
        if stopped_early:
            break
    return {"optimizer_steps": step, "planned_steps": total_steps, "stopped_early": stopped_early,
            "train_seconds": time.time() - started, "target_forward_seconds": target_seconds}


@torch.no_grad()
def diagnostics(model, m, eval_seqs):
    """Mixing weights / router weights of MixFusion, expert load of the MoE."""
    info = {}
    if isinstance(m, MixFusion):
        info["mix_weights_static"] = torch.softmax(m.logits.float(), -1).cpu().tolist()
        if m.router is not None:
            avg = torch.zeros(m.S, m.L, device="cuda")
            cnt = 0
            for seq in eval_seqs:
                st, _, pm, _ = step0_batch(model, seq)
                w = m.weights(st.float())[0]
                keep = pm[0] > 0
                avg += w[keep].sum(0)
                cnt += int(keep.sum())
                ent = -(w[keep] * w[keep].clamp_min(1e-9).log()).sum(-1).mean(0)
            info["router_mean_weights"] = (avg / max(cnt, 1)).cpu().tolist()
            info["router_last_entropy"] = ent.cpu().tolist()
    if isinstance(m, LowRankMoEFusion):
        loads = torch.zeros(m.E, device="cuda")
        for seq in eval_seqs:
            st, _, _, _ = step0_batch(model, seq)
            m(st.float())
            loads += m.last_load
        info["expert_load"] = (loads / len(eval_seqs)).cpu().tolist()
    return info


def tau_with(model, module, prompts, cfg, base_out, n_states, hidden):
    """tau of EAGLE generation with `module` as the draft input (all layers captured)."""
    ea = model.ea_layer
    fc_orig = ea.fc
    ea.fc = AllLayersAdapter(module, n_states, hidden)
    set_capture(model, tuple(range(n_states)))
    try:
        rows, _ = measure_acceptance(model, prompts, cfg["max_new_tokens"], reference=base_out,
                                     gen_kwargs=cfg.get("gen_kwargs"))
    finally:
        ea.fc = fc_orig
        set_capture(model, None)
    return rows


def run(cfg, pair):
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    amp = getattr(torch, cfg.get("amp_dtype", "float16"))
    model = load_eagle(pair, dtype=torch.float16)       # the original run loaded this stage in float16
    tok = model.get_tokenizer()
    ea = model.ea_layer
    N = len(model.base_model.model.layers)
    H = model.base_model.config.hidden_size
    tri = default_triplet(N)
    W = ea.fc.weight.detach().clone()
    result = {"config": dict(cfg), "n_layers": N, "hidden": H, "triplet": tri, "official_fc_params": int(W.numel())}

    # the official draft: tau, step-0 metrics, and the same weights through the adapter (control)
    prompts = benchmark_prompts(tok, cfg["prompt_counts"])
    base_rows, base_out = measure_acceptance(model, prompts, cfg["max_new_tokens"], gen_kwargs=cfg.get("gen_kwargs"))
    result["tau"] = {"official": tau_summary(base_rows)}
    log(f"official tau {result['tau']['official']}")
    eval_seqs = [prompt_plus_answer(p, base_out[p["id"]]) for p in prompts]
    official = FixedFusion(H, tri, init_weight=W).cuda().eval()
    set_capture(model, tuple(range(N + 1)))
    result["step0_eval"] = {"official": evaluate_step0(model, official, eval_seqs, amp)}
    log(f"official step0 {result['step0_eval']['official']}")
    result["tau"]["official_via_adapter"] = tau_summary(tau_with(model, official, prompts, cfg, base_out, N + 1, H))
    log(f"official via adapter tau {result['tau']['official_via_adapter']}")
    set_capture(model, tuple(range(N + 1)))
    del official

    # training
    variants = all_variants(cfg, H, N, W)
    train_seqs = torch.load(cfg["train_data"])[: cfg["n_train_seqs"]]
    log(f"train sequences: {len(train_seqs)}, tokens={sum(s['ids'].numel() for s in train_seqs)}")
    total_steps = math.ceil(len(train_seqs) * cfg["epochs"] / cfg["accum"])
    warmup = max(1, int(cfg["warmup_frac"] * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min((step - warmup) / max(1, total_steps - warmup), 1.0)))

    trainers = {}
    for name, (module, note) in variants.items():
        trainers[name] = make_trainer(module, note, name, cfg, lr_lambda, amp)
        log(f"{name}: params={trainers[name]['params']} ({note})")
    result["variants"] = {k: {"note": v["note"], "params": v["params"]} for k, v in trainers.items()}
    result.update(train(model, trainers, train_seqs, cfg, amp))
    for name, s in trainers.items():
        result["variants"][name].update(curve=s["curve"], seconds=s["seconds"])

    # held-out step-0 metrics
    for name, s in trainers.items():
        m = s["module"].eval()
        result["step0_eval"][name] = evaluate_step0(model, m, eval_seqs, amp)
        result["variants"][name].update(diagnostics(model, m, eval_seqs))
        log(f"{name} step0 {result['step0_eval'][name]}")
        torch.save({k: v.half() for k, v in m.state_dict().items()}, out / f"fusion_{name}.pt")
    save_json(result, out / "phase_b.json")

    # tau of every variant, compared with the official draft on the same prompts
    tau_rows = [dict(r, variant="official") for r in base_rows]
    for name, s in trainers.items():
        rows = tau_with(model, s["module"].eval(), prompts, cfg, base_out, N + 1, H)
        result["tau"][name] = tau_summary(rows)
        result["tau"][name]["delta_ci95"] = paired_bootstrap(base_rows, rows)
        result["tau"][name]["prefix_pair_official_vs_variant"] = prefix_tau_pair(base_rows, rows)
        tau_rows += [dict(r, variant=name) for r in rows]
        log(f"{name} tau {result['tau'][name]['tau']:.3f} ci={result['tau'][name]['delta_ci95']}")
        save_json(result, out / "phase_b.json")
        save_json(tau_rows, out / "tau_rows.json")
    log("finished")
    return result


if __name__ == "__main__":
    run(make_config(sys.argv[1]), sys.argv[1])
