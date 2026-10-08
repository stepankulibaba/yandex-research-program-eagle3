"""What the draft actually uses: evaluation of the official and the trained drafts with parts of the input removed.

    python probe.py dsl-8b

For every draft (the official one with its own fc, and every trained variant found in results/<pair>/) and three
input conditions, chain tau on all eval questions (and on the 96 used during training):
    full      the normal input [embedding of the next token ; g]
    no_g      g = 0: no target features at all, the draft is a one-layer LM over token embeddings
    no_emb    embedding = 0: target features only, without the shifted token
Output: results/<pair>/probe.json and a table in the log. No training; a few minutes on one GPU.
"""
import json
import sys

import numpy as np
import torch

from config import PAIRS, DATA, RESULTS
from draft import EagleDraft
from fusion import build_fusions
from metrics import leading_accepted, chain_positions
from train import Target, DEPTH, ALL_VARIANTS, load_eval_set

pair = sys.argv[1]
out_dir = RESULTS / pair
target = Target(PAIRS[pair]["target"])
N, H = target.n_layers, target.hidden
eval_seqs = torch.load(DATA / pair / "eval.pt")
train_eval_ids = {s["id"] for s in load_eval_set(pair, 96)}


def official():
    fusion = build_fusions(N, H, ["fixed_raw"])["fixed_raw"].eval()
    fc = torch.load(PAIRS[pair]["draft"] / "pytorch_model.bin", map_location="cpu", weights_only=True)["fc.weight"]
    with torch.no_grad():
        fusion.W.weight.copy_(fc.float())
    d = EagleDraft(PAIRS[pair]["draft"], target.embedding)
    d.fusion = fusion
    return d


def trained(name):
    ck = torch.load(out_dir / f"{name}.pt", weights_only=False)
    fusion = build_fusions(N, H, [name])[name]
    fusion.load_state_dict(ck["fusion"])
    fusion.eval()
    d = EagleDraft(PAIRS[pair]["draft"], target.embedding)
    for k, v in ck["draft"].items():
        k = {"norm": "final_norm"}.get(k, k)
        if k in d.weights:
            d.weights[k] = v.to("cuda", torch.bfloat16)
    d.fusion = fusion
    return d


class ZeroG(torch.nn.Module):
    """Same fusion, output replaced by zeros."""
    def __init__(self, fusion):
        super().__init__()
        self.fusion = fusion

    def forward(self, states):
        return torch.zeros_like(self.fusion(states))


drafts = {"official": official()}
drafts.update({n: trained(n) for n in ALL_VARIANTS if (out_dir / f"{n}.pt").exists()})
zero_emb = torch.zeros_like(target.embedding)
conditions = ("full", "no_g", "no_emb")

rows = {d: {c: [] for c in conditions} for d in drafts}
with torch.no_grad():
    for i, s in enumerate(eval_seqs):
        ids, mask = s["ids"].cuda(), s["mask"].cuda()
        states, logits = target(ids)
        greedy = logits.argmax(-1)
        for name, d in drafts.items():
            fusion, emb = d.fusion, d.embedding
            for c in conditions:
                d.fusion = ZeroG(fusion) if c == "no_g" else fusion
                d.embedding = zero_emb if c == "no_emb" else emb
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = d.predict_tokens(d.unroll(states, ids, DEPTH))
                lead = leading_accepted(pred, greedy, DEPTH)[chain_positions(mask, DEPTH)]
                rows[name][c].append({"id": s["id"], "bench": s["bench"], "lead_sum": int(lead.sum()), "n": int(lead.numel())})
            d.fusion, d.embedding = fusion, emb
        if i % 40 == 0:
            print(f"{i}/{len(eval_seqs)}", flush=True)


def tau(rs):
    return 1 + sum(r["lead_sum"] for r in rs) / max(1, sum(r["n"] for r in rs))


res = {}
for name, by_c in rows.items():
    res[name] = {}
    for c, rs in by_c.items():
        res[name][c] = {"all_240": tau(rs), "train_eval_96": tau([r for r in rs if r["id"] in train_eval_ids]),
                        **{b: tau([r for r in rs if r["bench"] == b]) for b in sorted({r["bench"] for r in rs})}}
(out_dir / "probe.json").write_text(json.dumps({"tau": res, "per_seq": rows}, indent=1))

print(f"\nchain tau (depth {DEPTH}) on {len(eval_seqs)} questions  [on the 96 used during training]")
print(f"{'draft':16s}" + "".join(f"{c:>20s}" for c in conditions))
for name, r in res.items():
    print(f"{name:16s}" + "".join(f"{r[c]['all_240']:12.3f} [{r[c]['train_eval_96']:.3f}]" for c in conditions))
