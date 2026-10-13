"""Final evaluation of the trained variants (and the untouched official draft) on all evaluation texts.

    python evaluate.py en        -> results/en/final.json

Texts: MT-bench, GSM8K, HumanEval (English, 240) and m-ArenaHard + MGSM in six languages (660), all answered
greedily by the target. For every variant: tau_chain and tau_cycles per benchmark and per language; for SpecVocab
also with k candidates for every k in config.K_VALUES and the recall of the candidates per draft step; per
language the share of answer tokens outside the official 32k vocabulary.
"""
import json
import sys
import time

import torch

from config import DATA, DRAFT, K_VALUES, RESULTS, TARGET, VARIANTS
from draft import DraftBody, load_checkpoint
from evaluation import Target, score_texts
from heads import FullHead, SpecVocabHead, StaticHead
from train import official_head


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_variant(path, state, cfg, target):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    body = DraftBody(state, cfg, target.embedding)
    body.load_state_dict({k: v.float() for k, v in saved["body"].items()})
    h = saved["head"]
    if saved["kind"] == "static":
        head = StaticHead(h["weight"], h["vocab_ids"])
    elif saved["kind"] == "full":
        head = FullHead(h["weight"])
    else:
        head = SpecVocabHead.__new__(SpecVocabHead)
        FullHead.__init__(head, h["weight"])
        head.aux_weight = saved["extra"].get("aux_weight", 0.1)
        head.vocab = torch.nn.Parameter(h["vocab"].float())
        head.down = torch.nn.Parameter(h["down"].float())
    return body.cuda(), head.cuda()


def main(run):
    out = RESULTS / run
    target = Target(TARGET)
    state, cfg = load_checkpoint(DRAFT)
    off_w, off_ids = official_head(state)
    variants = {"official": (DraftBody(state, cfg, target.embedding).cuda(), StaticHead(off_w, off_ids).cuda())}
    for name in VARIANTS:
        if (out / f"{name}.pt").exists():
            variants[name] = load_variant(out / f"{name}.pt", state, cfg, target)
    log(f"variants: {list(variants)}")
    seqs = torch.load(DATA / "eval_en.pt") + torch.load(DATA / "eval_lang.pt")
    res, oov = score_texts(target, variants, seqs, ks=K_VALUES, official_ids=off_ids, log=log)
    costs = {n: (h.cost() if h.kind != "specvocab" else {k: h.cost(k) for k in K_VALUES}) for n, (b, h) in variants.items()}
    result = {"texts": len(seqs), "k_values": K_VALUES, "oov_share": oov, "head_cost": costs, "res": res}
    (out / "final.json").write_text(json.dumps(result, indent=1))
    for key, groups in res.items():
        if "tau_chain" in groups.get("all", {}):
            log(f"{key:24s} " + " ".join(f"{g}={groups[g]['tau_chain']:.2f}" for g in ["en", "ru", "uk", "de", "zh", "ja"]
                                         if g in groups))
    log(f"wrote {out / 'final.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "en")
