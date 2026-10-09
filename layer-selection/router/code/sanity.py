"""Check before training: the official draft with its own fc must reach a high tau through this code.

    python sanity.py dsl-8b [n_texts]

Runs the official draft (its fc, layers (2, N/2, N-3), no normalisation) on held-out texts and prints chain tau.
For comparison it also runs the same draft with a wrong RoPE base (1e6 instead of the config/EAGLE value):
a correct implementation must be clearly better. Exits with code 1 if the check fails.
"""
import sys

import torch

from config import PAIRS, DATA
from draft import EagleDraft
from fusion import build_fusions
from metrics import leading_accepted, chain_positions
from train import Target, DEPTH

pair = sys.argv[1]
n_texts = int(sys.argv[2]) if len(sys.argv) > 2 else 40
target = Target(PAIRS[pair]["target"])
N, H = target.n_layers, target.hidden
seqs = torch.load(DATA / pair / "eval.pt")[:: max(1, 240 // n_texts)][:n_texts]

official = torch.load(PAIRS[pair]["draft"] / "pytorch_model.bin", map_location="cpu", weights_only=True)["fc.weight"]
fusion = build_fusions(N, H, ["fixed_raw"])["fixed_raw"].eval()
with torch.no_grad():
    fusion.W.weight.copy_(official.float())
draft = EagleDraft(PAIRS[pair]["draft"], target.embedding)
draft.fusion = fusion


@torch.no_grad()
def chain_tau():
    leads = []
    for s in seqs:
        ids, mask = s["ids"].cuda(), s["mask"].cuda()
        states, logits = target(ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = draft.predict_tokens(draft.unroll(states, ids, DEPTH))
        leads.append(leading_accepted(pred, logits.argmax(-1), DEPTH)[chain_positions(mask, DEPTH)].float().cpu())
    return 1 + float(torch.cat(leads).mean())


theta = draft.rope_theta
tau = chain_tau()
draft.rope_theta = 1e6 if theta != 1e6 else 1e4
tau_wrong = chain_tau()
draft.rope_theta = theta
print(f"N={N} H={H} layers={fusion.layers.tolist()} rope_theta={theta}")
print(f"official draft, chain tau (depth {DEPTH}, {len(seqs)} texts): {tau:.3f}")
print(f"same draft with rope_theta={1e6 if theta != 1e6 else 1e4:g}: {tau_wrong:.3f}")
ok = tau > 3.0 and tau > tau_wrong
print("SANITY OK" if ok else "SANITY FAILED: draft implementation does not reproduce the official draft")
sys.exit(0 if ok else 1)
