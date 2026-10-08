"""Phase B: train alternative fusion modules against the frozen official draft.

All variants see the same sequences in the same order; the frozen target is run
once per sequence and its features are shared by every variant. Only the
fusion module is trained, with the EAGLE-3 step-0 objective (soft cross-entropy
to the target distribution over the draft vocabulary).
"""
import math
import time
from pathlib import Path

import torch

import layer_lab as lab


AMP = torch.float16


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


@torch.no_grad()
def evaluate_step0(model, fusion, seqs, amp_dtype=torch.float16):
    loss_sum = acc_sum = weight = 0.0
    for seq in seqs:
        states, tp, pm, nxt = lab.target_batch(model, seq)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = lab.draft_step0_logits(model.ea_layer, fusion(states.float()), nxt)
        loss, acc = lab.step0_loss(logits, tp, pm)
        w = pm.sum().item()
        loss_sum += loss.item() * w
        acc_sum += acc.item() * w
        weight += w
    return {"loss": loss_sum / max(weight, 1), "acc": acc_sum / max(weight, 1), "positions": weight}


def run(cfg):
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    lab.patch_eagle(cfg["eagle_dir"])
    lab.seed_all(cfg["seed"])
    model = lab.load_pair(cfg["base"], cfg["draft"], cfg["total_token"], cfg["depth"], cfg["top_k"])
    tok = model.get_tokenizer()
    ea = model.ea_layer
    N = len(model.base_model.model.layers)
    H = model.base_model.config.hidden_size
    tri = lab.default_triplet(N)
    W = ea.fc.weight.detach().clone()
    fc_orig = ea.fc
    global AMP
    AMP = getattr(torch, cfg.get("amp_dtype", "float16"))
    result = {"config": dict(cfg), "n_layers": N, "hidden": H, "triplet": tri,
              "official_fc_params": int(W.numel())}

    prompts = lab.eval_prompts(cfg["eagle_dir"], tok, cfg["prompt_counts"])
    base_rows, base_out = lab.run_tau(model, prompts, cfg["max_new_tokens"], gen_kwargs=cfg.get("gen_kwargs"))
    result["tau"] = {"official": lab.tau_summary(base_rows)}
    log(f"official tau {result['tau']['official']}")
    if cfg.get("repeat_official"):
        rows, _ = lab.run_tau(model, prompts, cfg["max_new_tokens"], reference=base_out, gen_kwargs=cfg.get("gen_kwargs"))
        result["tau"]["official_repeat"] = lab.tau_summary(rows)
        log(f"official repeat (run-to-run determinism) {result['tau']['official_repeat']}")
    eval_seqs = []
    for p in prompts:
        gen = base_out[p["id"]]
        eval_seqs.append({"ids": torch.cat([p["ids"][0], gen]),
                          "mask": torch.cat([torch.zeros(p["ids"].shape[1], dtype=torch.long),
                                             torch.ones(gen.numel(), dtype=torch.long)])})

    official = lab.FixedFusion(H, tri, init_weight=W).cuda().eval()
    lab.set_capture(model, tuple(range(N + 1)))
    result["step0_eval"] = {"official": evaluate_step0(model, official, eval_seqs, AMP)}
    log(f"official step0 {result['step0_eval']['official']}")

    # Control: the official weights routed through the all-layer adapter must give the official tau.
    ea.fc = lab.StackAdapter(official, N + 1, H)
    try:
        rows, _ = lab.run_tau(model, prompts, cfg["max_new_tokens"], reference=base_out, gen_kwargs=cfg.get("gen_kwargs"))
    finally:
        ea.fc = fc_orig
        lab.set_capture(model, None)
    result["tau"]["official_via_adapter"] = lab.tau_summary(rows)
    log(f"official via adapter tau {result['tau']['official_via_adapter']}")
    lab.set_capture(model, tuple(range(N + 1)))
    del official

    variants = lab.build_variants(H, N, W)
    if cfg.get("variants"):
        variants = {k: v for k, v in variants.items() if k in cfg["variants"]}
    for name, layers in cfg.get("custom_variants", []):
        variants[name] = (lab.FixedFusion(H, layers, norm=True),
                          f"layers {tuple(layers)} + per-source RMSNorm; fc {len(layers)}H->H")
    if cfg.get("balanced_moe"):
        moe = variants["moe_lowrank"][0]
        bal = lab.LowRankMoEFusion(H, moe.layers, moe.down.out_features, moe.E, moe.e_down.shape[-1])
        bal.aux_coef = cfg["balanced_moe"]
        variants["moe_lowrank_bal"] = (bal, f"as moe_lowrank, load-balance coefficient {cfg['balanced_moe']}")
    if cfg.get("train_data"):
        train_seqs = torch.load(cfg["train_data"])[: cfg["n_train_seqs"]]
    else:
        train_seqs = lab.ultrachat_sequences(tok, cfg["n_train_seqs"], max_len=cfg["max_len"], skip=cfg["skip"])
    log(f"train sequences: {len(train_seqs)}, tokens={sum(s['ids'].numel() for s in train_seqs)}")

    accum = cfg["accum"]
    total_steps = math.ceil(len(train_seqs) * cfg["epochs"] / accum)
    warmup = max(1, int(cfg["warmup_frac"] * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    state = {}
    for name, (module, note) in variants.items():
        module.cuda().train()
        mult = cfg.get("lr_mult", {}).get(name, 1.0)
        fast = [p for n, p in module.named_parameters() if n.startswith("logits") or n.startswith("router.")]
        slow = [p for n, p in module.named_parameters() if not (n.startswith("logits") or n.startswith("router."))]
        groups = [{"params": slow, "lr": cfg["lr"] * mult}]
        if fast:
            groups.append({"params": fast, "lr": cfg["lr"] * mult * cfg["fast_lr_mult"]})
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
        state[name] = {"module": module, "note": note, "opt": opt,
                       "sched": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
                       "scaler": torch.amp.GradScaler("cuda", enabled=AMP == torch.float16), "params": lab.count_params(module),
                       "curve": [], "run_loss": 0.0, "run_acc": 0.0, "run_n": 0, "seconds": 0.0}
        log(f"{name}: params={state[name]['params']} ({note})")
    result["variants"] = {k: {"note": v["note"], "params": v["params"]} for k, v in state.items()}

    step = 0
    target_seconds = 0.0
    seq_count = 0
    started = time.time()
    order = list(range(len(train_seqs)))
    stopped_early = False
    for epoch in range(cfg["epochs"]):
        g = torch.Generator().manual_seed(cfg["seed"] + epoch)
        order = torch.randperm(len(train_seqs), generator=g).tolist()
        for i in order:
            if time.time() - started > cfg["max_train_seconds"] and seq_count % accum == 0:
                stopped_early = True
                log(f"time budget reached after {step} optimizer steps")
                break
            t0 = time.perf_counter()
            states, tp, pm, nxt = lab.target_batch(model, train_seqs[i])
            torch.cuda.synchronize()
            target_seconds += time.perf_counter() - t0
            seq_count += 1
            if pm.sum() == 0:
                continue
            feats = states.float()
            for name, s in state.items():
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=AMP):
                    fused = s["module"](feats)
                    logits = lab.draft_step0_logits(ea, fused, nxt)
                loss, acc = lab.step0_loss(logits, tp, pm)
                total = loss
                if isinstance(s["module"], lab.LowRankMoEFusion):
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
                    for name, s in state.items():
                        n = max(s["run_n"], 1)
                        s["curve"].append({"step": step, "loss": s["run_loss"] / n, "acc": s["run_acc"] / n})
                        line.append(f"{name}={s['run_acc'] / n:.3f}")
                        s["run_loss"] = s["run_acc"] = 0.0
                        s["run_n"] = 0
                    log(f"step {step}/{total_steps} ({time.time() - started:.0f}s) acc: " + " ".join(line))
        if stopped_early:
            break
    result["optimizer_steps"] = step
    result["planned_steps"] = total_steps
    result["stopped_early"] = stopped_early
    result["train_seconds"] = time.time() - started
    result["target_forward_seconds"] = target_seconds
    for name, s in state.items():
        result["variants"][name].update(curve=s["curve"], seconds=s["seconds"])

    # ---------------- evaluation ----------------
    for name, s in state.items():
        m = s["module"].eval()
        result["step0_eval"][name] = evaluate_step0(model, m, eval_seqs, AMP)
        info = result["variants"][name]
        if isinstance(m, lab.MixFusion):
            with torch.no_grad():
                info["mix_weights_static"] = torch.softmax(m.logits.float(), -1).cpu().tolist()
                if m.router is not None:
                    avg = torch.zeros(m.S, m.L, device="cuda")
                    cnt = 0
                    for seq in eval_seqs:
                        st, _, pm, _ = lab.target_batch(model, seq)
                        w = m.weights(st.float())[0]
                        keep = pm[0] > 0
                        avg += w[keep].sum(0)
                        cnt += int(keep.sum())
                        ent = -(w[keep] * w[keep].clamp_min(1e-9).log()).sum(-1).mean(0)
                    info["router_mean_weights"] = (avg / max(cnt, 1)).cpu().tolist()
                    info["router_last_entropy"] = ent.cpu().tolist()
        if isinstance(m, lab.LowRankMoEFusion):
            loads = torch.zeros(m.E, device="cuda")
            with torch.no_grad():
                for seq in eval_seqs:
                    st, _, pm, _ = lab.target_batch(model, seq)
                    m(st.float())
                    loads += m.last_load
            info["expert_load"] = (loads / len(eval_seqs)).cpu().tolist()
        log(f"{name} step0 {result['step0_eval'][name]}")
        torch.save({k: v.half() for k, v in m.state_dict().items()}, out / f"fusion_{name}.pt")
    lab.dump(result, out / "phase_b.json")

    tau_rows = [dict(r, variant="official") for r in base_rows]
    for name, s in state.items():
        ea.fc = lab.StackAdapter(s["module"].eval(), N + 1, H)
        lab.set_capture(model, tuple(range(N + 1)))
        try:
            rows, _ = lab.run_tau(model, prompts, cfg["max_new_tokens"], reference=base_out, gen_kwargs=cfg.get("gen_kwargs"))
        finally:
            ea.fc = fc_orig
            lab.set_capture(model, None)
        result["tau"][name] = lab.tau_summary(rows)
        result["tau"][name]["delta_ci95"] = lab.paired_bootstrap(base_rows, rows)
        result["tau"][name]["prefix_pair_official_vs_variant"] = lab.prefix_tau_pair(base_rows, rows)
        tau_rows += [dict(r, variant=name) for r in rows]
        log(f"{name} tau {result['tau'][name]['tau']:.3f} ci={result['tau'][name]['delta_ci95']}")
        lab.dump(result, out / "phase_b.json")
        lab.dump(tau_rows, out / "tau_rows.json")
    log("phase B finished")
    return result
