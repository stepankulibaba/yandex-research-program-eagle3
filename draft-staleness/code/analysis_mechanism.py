"""Transfer vs damage: why the draft does not adapt by itself (no GPU).

    python code/analysis_mechanism.py

On the original target's text (geometry1b, 1536 test tokens) every position is put in one of two groups:
  same     the edited target predicts the same next token as the original one
  changed  it predicts something else
and the draft's depth-1 prediction is compared when it reads the OLD (original target) or the NEW (edited target)
states.  transfer = gain on "changed" from reading the new states; damage = loss on "same".
Part 2 (needs the full geometry1b npz with the states, ~34 MB per edit, not in git, and the official draft's fc):
whether the shift is scale or direction, which captured layer the draft sees it through, and what the rank-2
correction does to the fc output.
"""
import collections
import os
from pathlib import Path

import numpy as np

DIR = Path(os.environ.get("STALENESS_RESULTS", Path(__file__).resolve().parent.parent / "results" / "llama31-8b"))


def ids(name):
    """Compact id file from git, or the full geometry1b npz if it is there."""
    p = DIR / f"geo1b_{name}_ids.npz"
    return np.load(p if p.exists() else DIR / f"geo1b_{name}.npz")
EDITS = ["limo_lora", "toolace", "supernova", "vikhr", "ultramedical", "huatuo_o1", "abliterated", "r1_distill",
         "nemotron_nano"]
H = 4096

top0 = ids("base")["top0"]
print("1. same / changed tokens (depth-1 agreement with the edited target's argmax)")
print(f"{'edit':14s} {'changed':>8s} | same: old / new / damage | changed: new follows NEW / OLD target (old features: NEW / OLD)")
for e in EDITS:
    z = ids(e)
    t1, dn, do = z["top1"], z["draft_new"], z["draft_old"]
    valid = (t1 >= 0) & (top0 >= 0)
    U, C = valid & (t1 == top0), valid & (t1 != top0)
    ao, an = (do[U] == t1[U]).mean(), (dn[U] == t1[U]).mean()
    print(f"{e:14s} {C.sum() / valid.sum():8.3f} | {ao:.3f} / {an:.3f} / {ao - an:+.3f} | "
          f"{(dn[C] == t1[C]).mean():.3f} / {(dn[C] == top0[C]).mean():.3f}   "
          f"(old features: {(do[C] == t1[C]).mean():.3f} / {(do[C] == top0[C]).mean():.3f})")

# examples: which tokens the new states break (where the target did not change)
try:
    import torch
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(hf_hub_download("unsloth/Meta-Llama-3.1-8B-Instruct", "tokenizer.json"))
    sd = torch.load(hf_hub_download("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "pytorch_model.bin"), map_location="cpu",
                    weights_only=True)
    d2t = sd["d2t"].long().numpy()
    dec = lambda i: repr(tok.decode([int(i + d2t[i])])) if i >= 0 else "<OOV>"
    for e in ("nemotron_nano", "r1_distill"):
        z = ids(e)
        t1, dn, do = z["top1"], z["draft_new"], z["draft_old"]
        bad = (t1 >= 0) & (t1 == top0) & (dn != t1) & (do == t1)
        print(f"\n{e}: {bad.sum()} tokens where the target did NOT change, old states right, new states wrong")
        print("  wrong draft predictions:", collections.Counter(dec(dn[i]) for i in np.where(bad)[0]).most_common(12))
        print("  true tokens there:     ", collections.Counter(dec(t1[i]) for i in np.where(bad)[0]).most_common(12))
except Exception as error:     # noqa: BLE001
    sd = None
    print("token examples skipped:", error)

print("\n2. scale or direction (needs the full geometry1b npz)")
if not (DIR / "geo1b_base.npz").exists() or sd is None:
    print("  skipped; numbers: results/llama31-8b/analysis/analysis_mechanism_v1.txt")
    raise SystemExit
W = sd["fc.weight"].float().numpy()
f0 = np.load(DIR / "geo1b_base.npz")["f0"].astype(np.float64)
for e in EDITS:
    f1 = np.load(DIR / f"geo1b_{e}.npz")["f1"].astype(np.float64)
    cells = []
    for j in range(3):
        a, c = f0[:, j * H:(j + 1) * H], f1[:, j * H:(j + 1) * H]
        na, nc = np.linalg.norm(a, axis=1), np.linalg.norm(c, axis=1)
        s = (a * c).sum(1) / na ** 2
        resc = 1 - np.sum(np.linalg.norm(c - s[:, None] * a, axis=1) ** 2) / np.sum(np.linalg.norm(c - a, axis=1) ** 2)
        cells.append(f"|f1|/|f0| {np.median(nc / na):.2f} cos {np.median((a * c).sum(1) / (na * nc)):.3f} rescale {resc:.2f}")
    u0, u1 = f0 @ W.T, f1 @ W.T
    parts = np.array([np.sum(((f1[:, j * H:(j + 1) * H] - f0[:, j * H:(j + 1) * H]) @ W[:, j * H:(j + 1) * H].T) ** 2)
                      for j in range(3)])
    print(f"  {e:14s} " + " || ".join(cells) +
          f" || fc output: cos {np.median((u0 * u1).sum(1) / (np.linalg.norm(u0, axis=1) * np.linalg.norm(u1, axis=1))):.3f}"
          f" |u1|/|u0| {np.median(np.linalg.norm(u1, axis=1) / np.linalg.norm(u0, axis=1)):.2f}"
          f" layer shares {np.round(parts / parts.sum(), 2).tolist()}")
