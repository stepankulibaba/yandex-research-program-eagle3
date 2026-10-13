"""Train the draft variants side by side from the same official draft (Qwen3-1.7B, AngelSlim EAGLE-3).

    python train.py en                     # data/train_en.pt -> results/en/
    python train.py ml                     # data/train_ml.pt (with multilingual prompts) -> results/ml/

All variants (config.VARIANTS) see the same texts in the same order; the target forward pass is shared. The whole
draft is trained (fusion fc, the decoder layer, norms, head); the target is frozen. Loss: soft cross-entropy to the
target's next-token distribution at each of 6 unrolled draft steps, weighted 0.8^(s-1) (EAGLE-3's training-time
test). The static head learns the target distribution restricted to its 32k tokens (as EAGLE-3); the full and
SpecVocab heads learn the full distribution, SpecVocab's ranker with weight 0.1.

Writes results/<run>/history.json after every evaluation and results/<run>/<variant>.pt at the end. Every
--ckpt-every steps the training state goes to results/<run>/resume.pt; a new run with the same command continues
from it.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from config import DATA, DEPTH, DRAFT, RESULTS, STEP_DECAY, TARGET, VARIANTS
from draft import DraftBody, load_checkpoint
from evaluation import Target, score_texts
from heads import build_head, extend_to_full


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def official_head(state):
    """The official draft's head and the target ids of its vocabulary (draft id i -> target id i + d2t[i])."""
    weight = state["lm_head.weight"]
    ids = torch.arange(weight.shape[0]) + state["d2t"].long()
    return weight, ids


def build_variants(target, names, device="cuda"):
    state, cfg = load_checkpoint(DRAFT)
    off_w, off_ids = official_head(state)
    full = None
    if any(VARIANTS[n][0] != "static" for n in names):
        full = extend_to_full(off_w.to(device), off_ids.to(device), target.out_weight, log).cpu()
    variants = {}
    for n in names:
        kind, extra = VARIANTS[n]
        body = DraftBody(state, cfg, target.embedding).to(device)
        head = build_head(kind, off_w, off_ids, full_weight=full, **extra).to(device)
        variants[n] = (body, head)
        log(f"{n}: head {kind}, {sum(p.numel() for p in head.parameters()):,} parameters, "
            f"{head.cost():,} multiply-adds per drafted token")
    return variants, off_ids


def ttt_loss(body, head, states, ids, mask, teacher_logp):
    """Weighted loss over the unrolled draft steps. Step s at position t guesses x_{t+s+1}; the target's
    distribution for that token is its output at position t+s. Only guesses inside the answer are scored."""
    T = ids.shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hs = body.unroll(states, ids, DEPTH)
    loss = 0.0
    for step, h in enumerate(hs, start=1):
        n = T - step
        scored = torch.zeros(n, dtype=torch.bool, device=ids.device)
        scored[: n - 1] = mask[1:n].bool()                       # first guess x_{t+2} lies in the answer
        if scored.any():
            ce = head.loss(h[:n][scored], teacher_logp[step:][scored])
            loss = loss + STEP_DECAY ** (step - 1) * ce.mean()
    return loss


def lr_multiplier(step, total, warmup=50):
    return min(1.0, (step + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * step / total))


def watch_set(n_en=64, n_per_lang=8):
    """Small fixed evaluation set for progress during training: English benchmarks + a few texts per language."""
    seqs = random.Random(1).sample(torch.load(DATA / "eval_en.pt"), n_en)
    lang = torch.load(DATA / "eval_lang.pt")
    for code in sorted({s["lang"] for s in lang}):
        pool = [s for s in lang if s["lang"] == code]
        seqs += random.Random(2).sample(pool, min(n_per_lang, len(pool)))
    return seqs


def main(a):
    out = RESULTS / a.run
    out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)
    target = Target(TARGET)
    variants, off_ids = build_variants(target, a.variants)
    optimizers, params = {}, {}
    for n, (body, head) in variants.items():
        ranker = [p for name, p in head.named_parameters() if name in ("vocab", "down")]
        main_params = list(body.parameters()) + [p for name, p in head.named_parameters() if name not in ("vocab", "down")]
        groups = [{"params": main_params, "lr": a.lr}]
        if ranker:
            groups.append({"params": ranker, "lr": a.lr_ranker})
        optimizers[n] = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
        for g in optimizers[n].param_groups:
            g["initial_lr"] = g["lr"]
        params[n] = main_params + ranker

    train = torch.load(DATA / f"train_{a.run}.pt")
    watch = watch_set()
    a.steps_per_epoch = a.steps_per_epoch or len(train)      # default: one pass over the texts per epoch
    total = a.steps_per_epoch * a.epochs
    history = {"config": {**vars(a), "depth": DEPTH, "watch_texts": len(watch), "train_texts": len(train)},
               "evals": []}
    losses = {n: [] for n in variants}

    def run_eval(step):
        res, oov = score_texts(target, variants, watch, ks=(2048,), official_ids=off_ids)
        entry = {"step": step, "epoch": step / a.steps_per_epoch, "res": res, "oov": oov}
        if step:
            entry["train_loss"] = {n: float(np.mean(v[-a.eval_every:])) for n, v in losses.items()}
        history["evals"].append(entry)
        line = []
        for key, groups in res.items():
            if "tau_chain" in groups.get("all", {}):
                line.append(f"{key}: en={groups['en']['tau_chain']:.2f} ru={groups.get('ru', {}).get('tau_chain', 0):.2f}")
        log(f"eval step {step}: " + " | ".join(line))
        (out / "history.json").write_text(json.dumps(history))

    ckpt_path = out / "resume.pt"

    def save_ckpt(step):
        try:
            tmp = out / "resume.pt.tmp"
            torch.save({"step": step, "history": history, "losses": losses,
                        "bodies": {n: b.state_dict() for n, (b, h) in variants.items()},
                        "heads": {n: h.state_dict() for n, (b, h) in variants.items()},
                        "opt": {n: o.state_dict() for n, o in optimizers.items()},
                        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}, tmp)
            os.replace(tmp, ckpt_path)
            log(f"checkpoint saved at step {step}")
        except Exception as e:  # noqa: BLE001 - a failed checkpoint must not stop training
            log(f"checkpoint at step {step} failed: {e}")

    start = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        for n, (body, head) in variants.items():
            body.load_state_dict(ck["bodies"][n])
            head.load_state_dict(ck["heads"][n])
            optimizers[n].load_state_dict(ck["opt"][n])
        history, losses, start = ck["history"], ck["losses"], ck["step"]
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
        del ck
        log(f"resumed from step {start}")
    else:
        run_eval(0)

    step, t0 = start, time.time()
    order = list(range(len(train)))
    for epoch in range(a.epochs):
        random.Random(epoch).shuffle(order)
        for j, i in enumerate(order[: a.steps_per_epoch]):
            if epoch * a.steps_per_epoch + j < start:
                continue
            ids, mask = train[i]["ids"].cuda(), train[i]["mask"].cuda()
            states, target_logits = target(ids)
            teacher_logp = torch.log_softmax(target_logits.float(), -1)
            lr_mult = lr_multiplier(step, total)
            for n, (body, head) in variants.items():
                opt = optimizers[n]
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * lr_mult
                loss = ttt_loss(body, head, states, ids, mask, teacher_logp)
                if torch.is_tensor(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params[n], 0.5)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    losses[n].append(float(loss))
            del teacher_logp, target_logits
            step += 1
            if step % 100 == 0:
                log(f"step {step}/{total} ({time.time() - t0:.0f}s, {(time.time() - t0) / (step - start):.2f} s/step, "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB) " +
                    " ".join(f"{n}={np.mean(v[-100:]):.3f}" for n, v in losses.items()))
            if step % a.eval_every == 0 or step == total:
                run_eval(step)
            if step % a.ckpt_every == 0 and step < total:
                save_ckpt(step)
            if a.max_steps and step >= a.max_steps:
                break
        if a.max_steps and step >= a.max_steps:
            break

    for n, (body, head) in variants.items():
        torch.save({"kind": head.kind, "extra": VARIANTS[n][1],
                    "body": {k: v.detach().to(torch.bfloat16).cpu() for k, v in body.state_dict().items()},
                    "head": {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() else v.detach()).cpu()
                             for k, v in head.state_dict().items()}}, out / f"{n}.pt")
    if ckpt_path.exists():
        ckpt_path.unlink()
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=["en", "ml"], help="training data: en (as before) or ml (+ multilingual)")
    ap.add_argument("--variants", type=lambda s: s.split(","), default=list(VARIANTS))
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="default: all training texts")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-ranker", type=float, default=1e-4)
    main(ap.parse_args())
