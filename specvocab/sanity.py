"""Check before training: the official draft must reach a high tau through this code.

    python sanity.py [n_texts]

Runs the official draft (its fc and 32k head) on held-out English texts and prints chain tau; then the same draft
with a wrong RoPE base. A correct implementation is clearly better than the wrong one. Exits 1 if not.
"""
import sys

import torch

from config import DATA, DEPTH, DRAFT, TARGET
from draft import DraftBody, load_checkpoint
from evaluation import Target, score_texts
from heads import StaticHead
from train import official_head

n_texts = int(sys.argv[1]) if len(sys.argv) > 1 else 40
target = Target(TARGET)
state, cfg = load_checkpoint(DRAFT)
w, ids = official_head(state)
body, head = DraftBody(state, cfg, target.embedding).cuda(), StaticHead(w, ids).cuda()
seqs = torch.load(DATA / "eval_en.pt")[:: max(1, 240 // n_texts)][:n_texts]


def tau():
    res, _ = score_texts(target, {"official": (body, head)}, seqs)
    return res["official|exact"]["all"]["tau_chain"]


theta = body.rope_theta
wrong = 1e4 if theta != 1e4 else 1e6
good = tau()
body.rope_theta = wrong
bad = tau()
body.rope_theta = theta
print(f"official draft, chain tau (depth {DEPTH}, {len(seqs)} texts): {good:.3f}  (rope_theta={theta:g})")
print(f"same draft with rope_theta={wrong:g}: {bad:.3f}")
ok = good > 2.5 and good > bad + 0.2
print("SANITY OK" if ok else "SANITY FAILED: the draft code does not reproduce the official draft")
sys.exit(0 if ok else 1)
