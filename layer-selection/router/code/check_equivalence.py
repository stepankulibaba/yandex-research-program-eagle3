"""Checks that the readable code computes the same as the code of the server run (original/).

Run from a directory that has original/ next to this file and the server's models/results layout:
    python check_equivalence.py <server_root>
Compares, on the trained checkpoints of the server run: eval texts, fusion init, draft logits, loss,
gradients, and the full evaluation (tau and layer choice).
"""
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SERVER = Path(sys.argv[1])
sys.path.insert(0, str(HERE))
import config  # noqa: E402

config.PAIRS["qwen3-1.7b"] = {"target": SERVER / "models/qwen3-1.7b/base", "draft": SERVER / "models/qwen3-1.7b/draft"}
config.EAGLE_REPO = SERVER / "EAGLE"
import prepare_data  # noqa: E402
prepare_data.PAIRS, prepare_data.EAGLE_REPO = config.PAIRS, config.EAGLE_REPO
import train as new  # noqa: E402
from draft import EagleDraft  # noqa: E402
from fusion import build_fusions  # noqa: E402

sys.path.insert(0, str(HERE / "original"))
import routed as old  # noqa: E402
import zoo  # noqa: E402

PAIR = "qwen3-1.7b"
CKPT = SERVER / "results/qwen3-1.7b/routed"
VARIANTS = ["fixed_raw", "fixed_alt", "static_top3", "token_top3", "token_top2"]


def report(name, ok, detail=""):
    print(f"[{'OK' if ok else 'DIFF'}] {name} {detail}", flush=True)


target = new.Target(config.PAIRS[PAIR]["target"])
N, H = target.n_layers, target.hidden

# 1. eval texts: readable prepare_data vs the original construction
import run as runmod  # noqa: E402
import layer_lab as lab  # noqa: E402
prompts = lab.eval_prompts(SERVER / "EAGLE", target.tokenizer, runmod.BENCHES + [["humaneval", None]])
prompts = random.Random(1).sample(prompts, min(96, len(prompts)))
base_out = torch.load(SERVER / "results/qwen3-1.7b/baseline_outputs.pt")
old_eval = []
for q in prompts:
    ids = torch.cat([q["ids"][0], base_out[q["id"]]])
    mask = torch.zeros_like(ids)
    mask[q["ids"].shape[1]:] = 1
    old_eval.append({"ids": ids, "mask": mask, "bench": q["bench"], "id": q["id"]})
all_new = prepare_data.make_eval(PAIR, SERVER / "results/qwen3-1.7b/baseline_outputs.pt")
new_eval = random.Random(1).sample(all_new, min(96, len(all_new)))
same = all(a["id"] == b["id"] and torch.equal(a["ids"], b["ids"]) and torch.equal(a["mask"], b["mask"])
           for a, b in zip(old_eval, new_eval)) and len(old_eval) == len(new_eval)
report("eval texts (96 sampled of %d)" % len(all_new), same)

# 2. fusion initialisation with the same seed
torch.manual_seed(0)
f_old = old.build_variants(N, H, VARIANTS)
torch.manual_seed(0)
f_new = build_fusions(N, H, VARIANTS)
report("fusion init", all(torch.equal(f_old[n].W.weight, f_new[n].W.weight) for n in VARIANTS))


# 3. trained checkpoints: logits, loss, gradients
def load_old(name):
    ck = torch.load(CKPT / f"{name}.pt")
    f = old.build_variants(N, H, [name])[name]
    f.load_state_dict(ck["fusion"])
    d = zoo.Draft(config.PAIRS[PAIR]["draft"], N, target_embed=target.embedding, name=name)
    d.trainable(["attn", "mlp", "norms", "lm_head"])
    for k, v in ck["draft"].items():
        if isinstance(d.W.get(k), torch.nn.Parameter):
            d.W[k].data.copy_(v.float())
    d.fuse = f
    return d, f


def load_new(name):
    ck = torch.load(CKPT / f"{name}.pt")
    f = build_fusions(N, H, [name])[name]
    f.load_state_dict(ck["fusion"])
    d = EagleDraft(config.PAIRS[PAIR]["draft"], target.embedding)
    d.fusion = f
    d.make_trainable()
    for k, v in ck["draft"].items():
        k = {"norm": "final_norm"}.get(k, k)
        if k in d.weights:
            d.weights[k].data.copy_(v.float())
    return d, f


seq = torch.load(SERVER / "results/qwen3-1.7b/train_data.pt")[0]
ids, mask = seq["ids"].cuda(), seq["mask"].cuda()
states, tlogits = target(ids)
for name in VARIANTS:
    (d_old, f_old1), (d_new, f_new1) = load_old(name), load_new(name)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = d_old.unroll(states, ids, 6)
        ln = d_new.unroll(states, ids, 6)
    diff = max(float((a.float() - b.float()).abs().max()) for a, b in zip(lo, ln))
    # original loss, copied from original/routed.py
    T = ids.shape[0]
    tgt_logp = torch.log_softmax(tlogits.float()[:, d_old.d2t_ids], -1)
    loss_old = 0.0
    for k, lg in enumerate(lo, start=1):
        valid = torch.zeros(T, dtype=torch.bool, device="cuda")
        valid[: T - k - 1] = mask[1: T - k].bool()
        pk = tgt_logp[k:][: T - k].exp()
        ce = -(pk * torch.log_softmax(lg[: T - k].float(), -1)).sum(-1)
        loss_old = loss_old + 0.8 ** (k - 1) * ce[valid[: T - k]].mean()
    loss_new = new.distillation_loss(ln, tlogits, d_new.draft_to_target, mask)
    loss_old.backward()
    loss_new.backward()
    gdiff = float((f_old1.W.weight.grad - f_new1.W.weight.grad).abs().max() /
                  f_old1.W.weight.grad.abs().max())
    report(f"{name}: logits / loss / grad", diff == 0 and abs(float(loss_old) - float(loss_new)) < 1e-6 and gdiff < 1e-6,
           f"max|dlogits|={diff:.2e} loss {float(loss_old):.6f} vs {float(loss_new):.6f} rel|dgrad|={gdiff:.1e}")

# 4. full evaluation on 24 texts
names = ["fixed_alt", "static_top3", "token_top3"]
olds = {n: load_old(n) for n in names}
news = {n: load_new(n) for n in names}
r_old = old.evaluate(target, {n: olds[n][0] for n in names}, {n: olds[n][1] for n in names}, old_eval[:24],
                     target.tokenizer, N)
r_new = new.evaluate(target, {n: news[n][0] for n in names}, {n: news[n][1] for n in names}, new_eval[:24])
for n in names:
    a, b = r_old[n], r_new[n]
    ok = a["tau_chain"] == b["tau_chain"] and a["tau_cycles"] == b["tau_cycles"] and \
        a.get("layer_share") == b.get("layer_share") and a.get("by") == b.get("by")
    report(f"evaluate {n}", ok, f"tau {a['tau_chain']:.4f} vs {b['tau_chain']:.4f}, "
                                f"cycles {a['tau_cycles']:.4f} vs {b['tau_cycles']:.4f}")
