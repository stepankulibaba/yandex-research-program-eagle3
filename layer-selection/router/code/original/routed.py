"""Unfrozen EAGLE-3 draft with a hard top-k layer router in place of the fixed fusion layers.

Every variant starts from the same official draft (decoder, norms and head are unfrozen and trained) with a
fresh fusion projection, and is trained with the multi-step (training-time-test style) objective on the same
target-generated sequences. All variants share one target forward per sequence.

Variants
  fixed_raw    : original EAGLE-3: layers (2, N//2, N-3), no normalisation
  fixed_norm   : same layers, per-layer RMSNorm + learnable per-layer gain
  fixed_alt    : layers (N//2, round(0.7N), N-3) with RMSNorm (best set from the frozen study)
  static_topK  : one learned top-K choice of layers shared by all tokens
  token_topK   : per-token router (reads the last target layer) picks K layers out of N+1

Selected states are sorted by depth, normalised, multiplied by their gate (softmax over the selected logits,
scaled to mean 1, which carries the router gradient), concatenated and projected by W (K*H -> H).

python routed.py run <pair> [--steps-per-epoch N] [--epochs E] [--variants a,b,...]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import zoo  # noqa: E402

PAIRS = {
    "qwen3-1.7b": {"base": ROOT / "models/qwen3-1.7b/base", "draft": ROOT / "models/qwen3-1.7b/draft",
                   "data": ROOT / "results/qwen3-1.7b"},
    "dsl-8b": {"base": ROOT / "models/dsl-8b/base", "draft": ROOT / "models/dsl-8b/draft",
               "data": ROOT / "results/dsl-8b"},
}
DEPTH = 6
DECAY = 0.8


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rms(x, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


class Fusion(nn.Module):
    def __init__(self, H, L, mode, k, layers=None, norm=True):
        super().__init__()
        self.mode, self.k, self.L, self.norm = mode, k, L, norm
        self.register_buffer("layers", torch.tensor(layers if layers is not None else [0] * k))
        self.W = nn.Linear(k * H, H, bias=False)
        self.gain = nn.Parameter(torch.ones(L, H)) if norm else None
        if mode == "static":
            self.logits = nn.Parameter(torch.zeros(L))
        if mode == "token":
            self.router = nn.Linear(H, L)
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
        self.noise = 0.0
        self.last = {}

    def route(self, feats):
        T = feats.shape[0]
        if self.mode == "fixed":
            idx = self.layers[None].expand(T, -1)
            return idx, torch.ones(T, self.k, device=feats.device), None
        if self.mode == "static":
            logits = self.logits[None].expand(T, -1)
        else:
            logits = self.router(rms(feats[:, -1]).float())
        probs = torch.softmax(logits.float(), -1)
        noisy = logits + self.noise * torch.randn_like(logits) if (self.training and self.noise > 0) else logits
        top = noisy.topk(self.k, -1).indices
        idx, order = top.sort(-1)
        gate = torch.softmax(logits.gather(-1, idx).float(), -1) * self.k
        return idx, gate, probs

    def forward(self, feats):
        T = feats.shape[0]
        idx, gate, probs = self.route(feats)
        x = feats[torch.arange(T, device=feats.device)[:, None], idx]          # [T, k, H]
        if self.norm:
            x = rms(x) * self.gain[idx].to(x.dtype)
        x = x * gate[..., None].to(x.dtype)
        self.last = {"idx": idx.detach(), "probs": None if probs is None else probs.detach()}
        return self.W(x.flatten(1).to(self.W.weight.dtype)).to(torch.bfloat16)


class Target:
    def __init__(self, path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                          attn_implementation="sdpa").eval()
        self.model.requires_grad_(False)
        self.N = self.model.config.num_hidden_layers
        self._pre = None
        self.model.model.norm.register_forward_pre_hook(lambda m, a: setattr(self, "_pre", a[0]))

    @torch.no_grad()
    def __call__(self, ids):
        out = self.model(input_ids=ids[None].cuda(), output_hidden_states=True)
        hs = list(out.hidden_states[: self.N]) + [self._pre]
        return torch.stack([h[0] for h in hs], 1), out.logits[0]


def build_variants(N, H, which):
    L = N + 1
    tri = [2, N // 2, N - 3]
    alt = [N // 2, round(0.7 * N), N - 3]
    spec = {
        "fixed_raw": ("fixed", 3, tri, False),
        "fixed_norm": ("fixed", 3, tri, True),
        "fixed_alt": ("fixed", 3, alt, True),
        "static_top3": ("static", 3, None, True),
        "token_top3": ("token", 3, None, True),
        "token_top2": ("token", 2, None, True),
        "static_top2": ("static", 2, None, True),
    }
    return {n: Fusion(H, L, *spec[n]).cuda() for n in which}


def token_class(tok, ids):
    """Coarse token categories for routing analysis."""
    out = []
    for t in tok.convert_ids_to_tokens(ids.tolist()):
        s = (t or "").replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
        core = s.strip()
        if not core:
            out.append("space")
        elif core.isdigit():
            out.append("digit")
        elif all(not c.isalnum() for c in core):
            out.append("punct")
        elif s.startswith(" ") or s.startswith("\n"):
            out.append("word_start")
        else:
            out.append("word_cont")
    return out


@torch.no_grad()
def evaluate(target, drafts, fusions, eval_seqs, tok, N):
    """Chain tau per variant and routing statistics on held-out texts."""
    res = {}
    for name, d in drafts.items():
        f = fusions[name]
        f.eval()
        d.fuse = f
        leads, cyc = [], []
        counts = torch.zeros(N + 1)
        by = {}
        for s in eval_seqs:
            ids, mask = s["ids"].cuda(), s["mask"].cuda()
            feats, logits = target(ids)
            greedy = logits.argmax(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = torch.stack([d.d2t_ids[l.argmax(-1)] for l in d.unroll(feats, ids, DEPTH)])
            lead, _ = zoo.chain_stats(pred, greedy, mask, DEPTH)
            leads.append(lead.cpu())
            lf = zoo.chain_lead_full(pred, greedy, DEPTH)
            valid = torch.zeros_like(mask, dtype=torch.bool)
            start = int(mask.nonzero()[0]) - 1
            valid[start: ids.shape[0] - DEPTH - 1] = True
            cyc += zoo.renewal_tau(lf, valid, DEPTH)
            if f.mode != "fixed":
                idx = f.last["idx"][mask.bool()].cpu()
                counts += torch.bincount(idx.flatten(), minlength=N + 1).float()
                conf = torch.softmax(logits.float(), -1).max(-1).values[mask.bool()].cpu()
                cls = token_class(tok, ids[mask.bool()].cpu())
                pos = torch.arange(ids.shape[0])[mask.bool().cpu()] - start
                for key, groups in (("bench", [s["bench"]] * len(cls)), ("token", cls),
                                    ("conf", ["hi" if c > 0.9 else ("mid" if c > 0.5 else "lo") for c in conf.tolist()]),
                                    ("pos", ["<64" if p < 64 else ("<256" if p < 256 else ">=256") for p in pos.tolist()])):
                    for g in set(groups):
                        sel = torch.tensor([x == g for x in groups])
                        b = by.setdefault(key, {}).setdefault(g, torch.zeros(N + 1))
                        b += torch.bincount(idx[sel].flatten(), minlength=N + 1).float()
        lead = torch.cat(leads).float()
        r = {"tau_chain": 1 + float(lead.mean()), "tau_cycles": float(np.mean(cyc)),
             "accept_by_depth": [float((lead >= s).float().mean()) for s in range(1, DEPTH + 1)]}
        if f.mode != "fixed":
            r["layer_share"] = (counts / counts.sum()).tolist()
            r["by"] = {k: {g: (v / v.sum()).tolist() for g, v in groups.items()} for k, groups in by.items()}
        if f.mode == "static":
            r["static_probs"] = torch.softmax(f.logits.detach().float(), -1).cpu().tolist()
        res[name] = r
        f.train()
    return res


def run(pair, which, steps_per_epoch, epochs, eval_every, lr_draft, lr_fusion, lr_router, n_eval, noise0, noise_frac):
    p = PAIRS[pair]
    out = p["data"] / "routed"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)
    target = Target(p["base"])
    tok, N = target.tok, target.N
    H = target.model.config.hidden_size
    embed = target.model.model.embed_tokens.weight
    fusions = build_variants(N, H, which)
    drafts, opts, params_all = {}, {}, {}
    for name in which:
        d = zoo.Draft(p["draft"], N, target_embed=embed, name=name)
        dparams = d.trainable(["attn", "mlp", "norms", "lm_head"])
        f = fusions[name]
        fparams = [q for n_, q in f.named_parameters() if not (n_.startswith("router") or n_ == "logits")]
        rparams = [q for n_, q in f.named_parameters() if n_.startswith("router") or n_ == "logits"]
        groups = [{"params": dparams, "lr": lr_draft}, {"params": fparams, "lr": lr_fusion}]
        if rparams:
            groups.append({"params": rparams, "lr": lr_router})
        opts[name] = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
        d.fuse = f
        drafts[name] = d
        params_all[name] = dparams + fparams + rparams
        log(f"{name}: fusion params {sum(q.numel() for q in fparams + rparams):,}")
    total = steps_per_epoch * epochs
    for name, o in opts.items():
        for g in o.param_groups:
            g["initial_lr"] = g["lr"]
    train = torch.load(p["data"] / "train_data.pt")
    base_out = torch.load(p["data"] / "baseline_outputs.pt")
    import run as runmod  # prompts of the paper protocol, same as the frozen study
    import layer_lab as lab
    lab.patch_eagle(runmod.EAGLE)
    prompts = lab.eval_prompts(runmod.EAGLE, tok, runmod.BENCHES + [["humaneval", None]])
    rng = random.Random(1)
    prompts = rng.sample(prompts, min(n_eval, len(prompts)))
    eval_seqs = []
    for q in prompts:
        gen = base_out[q["id"]]
        ids = torch.cat([q["ids"][0], gen])
        mask = torch.zeros_like(ids)
        mask[q["ids"].shape[1]:] = 1
        eval_seqs.append({"ids": ids, "mask": mask, "bench": q["bench"]})
    history = {"config": {"pair": pair, "variants": which, "steps_per_epoch": steps_per_epoch, "epochs": epochs,
                          "lr_draft": lr_draft, "lr_fusion": lr_fusion, "lr_router": lr_router, "depth": DEPTH,
                          "noise0": noise0, "noise_frac": noise_frac, "n_eval": len(eval_seqs)},
               "evals": []}
    ev = evaluate(target, drafts, fusions, eval_seqs, tok, N)
    history["evals"].append({"step": 0, "epoch": 0.0, "res": ev})
    log("step 0: " + " ".join(f"{n}={r['tau_chain']:.3f}" for n, r in ev.items()))
    order = list(range(len(train)))
    step = 0
    t0 = time.time()
    run_loss = {n: [] for n in which}
    for ep in range(epochs):
        random.Random(ep).shuffle(order)
        for i in order[:steps_per_epoch]:
            s = train[i]
            ids, mask = s["ids"].cuda(), s["mask"].cuda()
            feats, logits_t = target(ids)
            T = ids.shape[0]
            frac = step / max(total, 1)
            noise = noise0 * max(0.0, 1 - frac / noise_frac) if noise_frac > 0 else 0.0
            lr_mult = min(1.0, (step + 1) / 50) * 0.5 * (1 + np.cos(np.pi * frac))
            for name in which:
                d, f, o = drafts[name], fusions[name], opts[name]
                f.noise = noise
                for g in o.param_groups:
                    g["lr"] = g["initial_lr"] * lr_mult
                tgt_logp = torch.log_softmax(logits_t.float()[:, d.d2t_ids], -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outs = d.unroll(feats, ids, DEPTH)
                loss = 0.0
                for k, lg in enumerate(outs, start=1):
                    valid = torch.zeros(T, dtype=torch.bool, device="cuda")
                    valid[: T - k - 1] = mask[1: T - k].bool()
                    pk = tgt_logp[k:][: T - k].exp()
                    lp = torch.log_softmax(lg[: T - k].float(), -1)
                    ce = -(pk * lp).sum(-1)
                    loss = loss + DECAY ** (k - 1) * ce[valid[: T - k]].mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_all[name], 0.5)
                o.step()
                o.zero_grad(set_to_none=True)
                run_loss[name].append(float(loss))
            step += 1
            if step % 100 == 0:
                log(f"step {step}/{total} ({time.time() - t0:.0f}s) noise={noise:.2f} " +
                    " ".join(f"{n}={np.mean(v[-100:]):.3f}" for n, v in run_loss.items()))
            if step % eval_every == 0 or step == total:
                ev = evaluate(target, drafts, fusions, eval_seqs, tok, N)
                history["evals"].append({"step": step, "epoch": step / steps_per_epoch, "res": ev,
                                         "train_loss": {n: float(np.mean(v[-eval_every:])) for n, v in run_loss.items()}})
                log(f"eval step {step}: " + " ".join(f"{n}={r['tau_chain']:.3f}" for n, r in ev.items()))
                for n, r in ev.items():
                    if "layer_share" in r:
                        top = np.argsort(r["layer_share"])[::-1][:5]
                        log(f"   {n} top layers: " + ", ".join(f"{int(i)}:{r['layer_share'][i]:.2f}" for i in top))
                (out / "history.json").write_text(json.dumps(history))
    for n, f in fusions.items():
        torch.save({"fusion": {k: v.detach().cpu() for k, v in f.state_dict().items()},
                    "draft": drafts[n].state()}, out / f"{n}.pt")
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("pair")
    ap.add_argument("--variants", default="fixed_raw,fixed_norm,fixed_alt,static_top3,token_top3,token_top2")
    ap.add_argument("--steps-per-epoch", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lr-draft", type=float, default=5e-5)
    ap.add_argument("--lr-fusion", type=float, default=5e-4)
    ap.add_argument("--lr-router", type=float, default=5e-3)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--noise0", type=float, default=1.0)
    ap.add_argument("--noise-frac", type=float, default=0.3)
    a = ap.parse_args()
    run(a.pair, a.variants.split(","), a.steps_per_epoch, a.epochs, a.eval_every, a.lr_draft, a.lr_fusion,
        a.lr_router, a.n_eval, a.noise0, a.noise_frac)
