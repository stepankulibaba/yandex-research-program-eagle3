"""Check before training: the official draft must reach its known tau through this code.

    python sanity.py [n_texts]

Runs the official draft (its fc and 32k head) on held-out English texts and prints chain tau (depth 6). The same
code in our layer study gave 2.395 for this draft, against 2.406 in real EAGLE generation, so a working setup is
clearly above 2.0; a broken one (wrong weights, wrong layers, misaligned positions) falls to ~1.0-1.3.
For information it also prints tau with the other RoPE base (EAGLE at the pinned commit uses the config's 1e6).
Exits 1 if the check fails.
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
texts = torch.load(DATA / "eval_en.pt")
seqs = texts[:: max(1, len(texts) // n_texts)][:n_texts]


def tau():
    res, _ = score_texts(target, {"official": (body, head)}, seqs)
    return res["official|exact"]["all"]["tau_chain"]


theta = body.rope_theta
good = tau()
body.rope_theta = 1e4 if theta != 1e4 else 1e6
other = tau()
body.rope_theta = theta
print(f"official draft, chain tau (depth {DEPTH}, {len(seqs)} texts): {good:.3f}  (rope_theta={theta:g}; "
      f"expected ~2.4)")
print(f"for information, rope_theta={1e4 if theta != 1e4 else 1e6:g}: {other:.3f}")
ok = good > 2.0
print("SANITY OK" if ok else "SANITY FAILED: the draft code does not reproduce the official draft")
sys.exit(0 if ok else 1)
