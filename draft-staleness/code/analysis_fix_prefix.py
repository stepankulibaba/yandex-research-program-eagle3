"""The fc correction compared on IDENTICAL text: the shared prefix of the answers with and without the correction.

    python code/analysis_fix_prefix.py      (needs the `tokenizers` package; downloads the Llama-3.1 tokenizer)

EAGLE is lossless in exact arithmetic, but in bf16 the tree verification can resolve near-ties differently from the
run without the correction, so the two answers to a prompt may diverge after some point (36% of the Nemotron answers and
11% of the R1-Distill answers are identical to the letter). Here every prompt contributes only the verification cycles
that end before the first differing token (minus 2 for safety); tau change and a paired bootstrap CI over prompts.
"""
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DIR = Path(__file__).resolve().parent.parent / "results" / "llama31-8b"
EN = ["general", "math", "code", "medicine", "tool"]
rng = np.random.default_rng(0)
tok = Tokenizer.from_file(hf_hub_download("unsloth/Meta-Llama-3.1-8B-Instruct", "tokenizer.json"))


def tau_prefix(acc, k):
    """Tokens and cycles of the cycles that end within the first k answer tokens (position 0 is the prefill token)."""
    pos, tokens, cycles = 1, 0, 0
    for a in acc:
        if pos + a + 1 > k:
            break
        pos += a + 1
        tokens += a + 1
        cycles += 1
    return tokens, cycles


for e in ("nemotron_nano", "r1_distill", "toolace", "abliterated"):
    o = json.loads((DIR / f"{e}__inst.json").read_text(encoding="utf-8"))
    f = json.loads((DIR / f"{e}__inst_fix.json").read_text(encoding="utf-8"))
    ro, rf = {r["id"]: r for r in o["rows"]}, {r["id"]: r for r in f["rows"]}
    pairs, shared = [], []
    for i in (i for i in ro if ro[i]["domain"] in EN):
        a = tok.encode(ro[i]["answer"], add_special_tokens=False).ids
        b = tok.encode(rf[i]["answer"], add_special_tokens=False).ids
        k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        k = max(k - 2, 0)
        to, co = tau_prefix(ro[i]["accepted"], k)
        tf, cf = tau_prefix(rf[i]["accepted"], k)
        if co >= 3 and cf >= 3:
            pairs.append((to, co, tf, cf))
            shared.append(k / max(len(a), 1))
    P = np.array(pairs, float)

    def ratio(M):
        return (M[:, 2].sum() / M[:, 3].sum()) / (M[:, 0].sum() / M[:, 1].sum()) - 1

    lo, hi = np.percentile([ratio(P[rng.integers(0, len(P), len(P))]) for _ in range(4000)], [2.5, 97.5])
    print(f"{e:14s} prompts {len(P)} | shared prefix covers {np.mean(shared):.2f} of the answer | tau on identical text "
          f"{P[:, 0].sum() / P[:, 1].sum():.3f} -> {P[:, 2].sum() / P[:, 3].sum():.3f} "
          f"({ratio(P) * 100:+.1f}%, 95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}])")
