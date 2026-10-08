"""Unfrozen EAGLE-3 draft with a learned choice of target layers.

All variants (see fusion.py) start from the same official draft, get a fresh fusion matrix W, and are trained
side by side on the same sequences in the same order; the target forward pass is shared. The whole draft
(attention, MLP, norms, head) is trained; the target is frozen.

    python train.py qwen3-1.7b --steps-per-epoch 8000 --epochs 3 --eval-every 1000

Writes results/<pair>/history.json (after every evaluation) and results/<pair>/<variant>.pt (at the end).
Every --ckpt-every steps the full training state is saved to results/<pair>/resume.pt; if that file exists,
a new run continues from it (so an interrupted run is restarted with the same command).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PAIRS, DATA, RESULTS
from draft import EagleDraft
from fusion import build_fusions
from metrics import leading_accepted, chain_positions, cycle_lengths

DEPTH = 6            # draft steps per position (training-time-test unroll)
STEP_DECAY = 0.8     # loss weight of step s is 0.8^(s-1), as in EAGLE-3
# token_top3 / token_top2 (raw router input, lr 5e-3) froze their choice in the first steps on Qwen3-1.7B;
# the *_s versions are the corrected per-token routers. The old ones can still be run with --variants.
ALL_VARIANTS = ["fixed_raw", "fixed_norm", "fixed_alt", "static_top3", "static_top2", "token_top3_s", "token_top2_s"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Target:
    """Frozen target model returning all N+1 hidden states: inputs of layers 0..N-1 and the final pre-norm state."""

    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                          attn_implementation="sdpa").eval()
        self.model.requires_grad_(False)
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden = self.model.config.hidden_size
        self.embedding = self.model.model.embed_tokens.weight
        # HF returns the last state after the final norm; grab it before the norm instead
        self._pre_norm = None
        self.model.model.norm.register_forward_pre_hook(lambda module, args: setattr(self, "_pre_norm", args[0]))

    @torch.no_grad()
    def __call__(self, ids):
        out = self.model(input_ids=ids[None].cuda(), output_hidden_states=True)
        states = list(out.hidden_states[: self.n_layers]) + [self._pre_norm]
        return torch.stack([s[0] for s in states], 1), out.logits[0]       # [T, N+1, H], [T, vocab]


# ---------------------------------------------------------------------------------------------- training

def distillation_loss(step_logits, target_logits, draft_to_target, response_mask):
    """Soft cross-entropy between the target's next-token distribution and the draft's guess, for every step.

    Draft step s at position t guesses x_{t+s+1}; the target's distribution for that token is its output at t+s.
    The target distribution is restricted to the draft vocabulary and renormalised.
    """
    T = response_mask.shape[0]
    target_logp = torch.log_softmax(target_logits.float()[:, draft_to_target], -1)
    loss = 0.0
    for step, logits in enumerate(step_logits, start=1):
        n = T - step
        teacher = target_logp[step:].exp()                          # [n, V]
        student = torch.log_softmax(logits[:n].float(), -1)        # [n, V]
        ce = -(teacher * student).sum(-1)
        scored = torch.zeros(n, dtype=torch.bool, device=ce.device)
        scored[: n - 1] = response_mask[1:n].bool()                 # first guess x_{t+2} lies in the response
        loss = loss + STEP_DECAY ** (step - 1) * ce[scored].mean()
    return loss


def lr_multiplier(step, total, warmup=50):
    return min(1.0, (step + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * step / total))


def exploration_noise(step, total, start, until_frac):
    """Router noise: linear decay from `start` to 0 over the first `until_frac` of training."""
    if until_frac <= 0:
        return 0.0
    return start * max(0.0, 1 - (step / total) / until_frac)


# ---------------------------------------------------------------------------------------------- evaluation

def token_kind(tokenizer, ids):
    """Coarse token type for the routing breakdown."""
    kinds = []
    for t in tokenizer.convert_ids_to_tokens(ids.tolist()):
        s = (t or "").replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
        core = s.strip()
        if not core:
            kinds.append("space")
        elif core.isdigit():
            kinds.append("digit")
        elif all(not c.isalnum() for c in core):
            kinds.append("punct")
        elif s.startswith(" ") or s.startswith("\n"):
            kinds.append("word_start")
        else:
            kinds.append("word_cont")
    return kinds


class RoutingStats:
    """Counts how often each target layer is chosen, overall and per category of the token."""

    def __init__(self, n_states):
        self.n = n_states
        self.total = torch.zeros(n_states)
        self.by = {}

    def add(self, choice, categories):
        """choice: [P, k] chosen layers at P response positions; categories: {name: [P labels]}."""
        self.total += torch.bincount(choice.flatten(), minlength=self.n).float()
        for name, labels in categories.items():
            for label in set(labels):
                rows = torch.tensor([x == label for x in labels])
                counts = self.by.setdefault(name, {}).setdefault(label, torch.zeros(self.n))
                counts += torch.bincount(choice[rows].flatten(), minlength=self.n).float()

    def shares(self):
        return (self.total / self.total.sum()).tolist(), \
            {k: {g: (v / v.sum()).tolist() for g, v in groups.items()} for k, groups in self.by.items()}


@torch.no_grad()
def evaluate(target, drafts, fusions, eval_seqs):
    """Chain tau of every variant on held-out texts; for routed variants also which layers they pick."""
    results = {}
    for name, draft in drafts.items():
        fusion = fusions[name]
        fusion.eval()
        leading, cycles = [], []
        routing = RoutingStats(target.n_layers + 1)
        gate_top, score_std = [], []
        for seq in eval_seqs:
            ids, mask = seq["ids"].cuda(), seq["mask"].cuda()
            states, logits = target(ids)
            greedy = logits.argmax(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = draft.predict_tokens(draft.unroll(states, ids, DEPTH))
            lead = leading_accepted(predicted, greedy, DEPTH)
            leading.append(lead[chain_positions(mask, DEPTH)].cpu())
            first = int(mask.nonzero()[0]) - 1                       # the cycle that produces the first response token
            last = ids.shape[0] - DEPTH - 2
            if last >= first:
                cycles += cycle_lengths(lead, first, last, DEPTH)

            if fusion.mode != "fixed":
                resp = mask.bool()
                confidence = torch.softmax(logits.float(), -1).max(-1).values[resp].cpu()
                position = torch.arange(ids.shape[0])[resp.cpu()] - first
                gate_top.append((fusion.last_gates[resp].max(-1).values / fusion.k).cpu())
                score_std.append(fusion.last_score_std.expand(len(resp))[resp].cpu())
                routing.add(fusion.last_choice[resp].cpu(), {
                    "bench": [seq["bench"]] * int(resp.sum()),
                    "token": token_kind(target.tokenizer, ids[resp].cpu()),
                    "conf": ["hi" if c > 0.9 else ("mid" if c > 0.5 else "lo") for c in confidence.tolist()],
                    "pos": ["<64" if p < 64 else ("<256" if p < 256 else ">=256") for p in position.tolist()],
                })

        lead = torch.cat(leading).float()
        r = {"tau_chain": 1 + float(lead.mean()), "tau_cycles": float(np.mean(cycles)),
             "accept_by_depth": [float((lead >= s).float().mean()) for s in range(1, DEPTH + 1)]}
        if fusion.mode != "fixed":
            r["layer_share"], r["by"] = routing.shares()
            # saturation check: share of the gate mass on the strongest chosen layer (1/k = equal, 1 = only one)
            # and the spread of the scores over layers
            r["gate_top_share"] = float(torch.cat(gate_top).mean())
            r["score_std"] = float(torch.cat(score_std).mean())
        if fusion.mode == "static":
            r["static_probs"] = torch.softmax(fusion.logits.detach().float(), -1).cpu().tolist()
        results[name] = r
        fusion.train()
    return results


def load_eval_set(pair, n_eval):
    seqs = torch.load(DATA / pair / "eval.pt")
    return random.Random(1).sample(seqs, min(n_eval, len(seqs)))


# ---------------------------------------------------------------------------------------------- main loop

def main(a):
    out = RESULTS / a.pair
    out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)

    target = Target(PAIRS[a.pair]["target"])
    fusions = build_fusions(target.n_layers, target.hidden, a.variants)
    drafts, optimizers, params = {}, {}, {}
    for name in a.variants:
        draft = EagleDraft(PAIRS[a.pair]["draft"], target.embedding)
        draft.fusion = fusions[name]
        draft_params = draft.make_trainable()
        proj_params = fusions[name].projection_parameters()
        router_params = fusions[name].router_parameters()
        groups = [{"params": draft_params, "lr": a.lr_draft}, {"params": proj_params, "lr": a.lr_fusion}]
        if router_params:
            f = fusions[name]
            groups.append({"params": router_params,
                           "lr": a.lr_token_router if f.mode == "token" and f.unit_input else a.lr_router})
        optimizers[name] = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
        for g in optimizers[name].param_groups:
            g["initial_lr"] = g["lr"]
        drafts[name] = draft
        params[name] = draft_params + proj_params + router_params
        log(f"{name}: fusion parameters {sum(p.numel() for p in proj_params + router_params):,}")

    train = torch.load(DATA / a.pair / "train.pt")
    eval_seqs = load_eval_set(a.pair, a.n_eval)
    total = a.steps_per_epoch * a.epochs
    history = {"config": {**vars(a), "depth": DEPTH, "n_eval": len(eval_seqs)}, "evals": []}
    losses = {n: [] for n in a.variants}

    def run_eval(step):
        res = evaluate(target, drafts, fusions, eval_seqs)
        entry = {"step": step, "epoch": step / a.steps_per_epoch, "res": res}
        if step:
            entry["train_loss"] = {n: float(np.mean(v[-a.eval_every:])) for n, v in losses.items()}
        history["evals"].append(entry)
        log(f"eval step {step}: " + " ".join(f"{n}={r['tau_chain']:.3f}" for n, r in res.items()))
        for n, r in res.items():
            if "layer_share" in r:
                top = np.argsort(r["layer_share"])[::-1][:5]
                log(f"   {n} top layers: " + ", ".join(f"{int(i)}:{r['layer_share'][i]:.2f}" for i in top) +
                    f" | gate on strongest {r['gate_top_share']:.2f}, score spread {r['score_std']:.2f}")
        (out / "history.json").write_text(json.dumps(history))

    ckpt_path = out / "resume.pt"

    def save_ckpt(step):
        try:
            tmp = out / "resume.pt.tmp"
            torch.save({"step": step, "history": history, "losses": losses,
                        "fusions": {n: f.state_dict() for n, f in fusions.items()},
                        "drafts": {n: {k: w.detach() for k, w in d.weights.items()} for n, d in drafts.items()},
                        "opt": {n: o.state_dict() for n, o in optimizers.items()},
                        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}, tmp)
            os.replace(tmp, ckpt_path)
            log(f"checkpoint saved at step {step}")
        except Exception as e:  # a failed checkpoint must not stop training
            log(f"checkpoint at step {step} failed: {e}")

    start = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)   # straight to GPU: ~36 GB, spares RAM
        for n in a.variants:
            fusions[n].load_state_dict(ck["fusions"][n])
            for k, w in ck["drafts"][n].items():
                drafts[n].weights[k].data.copy_(w)
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
            noise = exploration_noise(step, total, a.noise0, a.noise_frac)
            lr_mult = lr_multiplier(step, total)
            for name in a.variants:
                draft, opt = drafts[name], optimizers[name]
                draft.fusion.noise = noise
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * lr_mult
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    step_logits = draft.unroll(states, ids, DEPTH)
                loss = distillation_loss(step_logits, target_logits, draft.draft_to_target, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params[name], 0.5)
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses[name].append(float(loss))
            step += 1
            if step % 100 == 0:
                log(f"step {step}/{total} ({time.time() - t0:.0f}s, {(time.time() - t0) / (step - start):.2f} s/step) noise={noise:.2f} " +
                    " ".join(f"{n}={np.mean(v[-100:]):.3f}" for n, v in losses.items()))
            if step % a.eval_every == 0 or step == total:
                run_eval(step)
            if step % a.ckpt_every == 0 and step < total:
                save_ckpt(step)

    for name in a.variants:
        torch.save({"fusion": {k: v.detach().cpu() for k, v in fusions[name].state_dict().items()},
                    "draft": drafts[name].state()}, out / f"{name}.pt")
    if ckpt_path.exists():
        ckpt_path.unlink()
    log("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(PAIRS))
    ap.add_argument("--variants", type=lambda s: s.split(","), default=ALL_VARIANTS)
    ap.add_argument("--steps-per-epoch", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--lr-draft", type=float, default=5e-5)
    ap.add_argument("--lr-fusion", type=float, default=5e-4)
    ap.add_argument("--lr-router", type=float, default=5e-3)
    ap.add_argument("--lr-token-router", type=float, default=1e-3, help="lr of the unit-input token routers (*_s)")
    ap.add_argument("--noise0", type=float, default=1.0, help="initial router noise")
    ap.add_argument("--noise-frac", type=float, default=0.3, help="share of training with decaying noise")
    main(ap.parse_args())
