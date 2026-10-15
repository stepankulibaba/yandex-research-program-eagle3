"""Step 1c: is there a shared low-dimensional "response" of the draft to target edits, beyond its generic error?

    python geometry1c.py [TARGET ...]

On the common text (same tokens for every edit), in clamped+centered log-prob space on the draft vocabulary:
  R0      = lp(T0) - lp(D(f0))                      generic error of the draft (no edit)
  R1      = lp(T1) - lp(D(f1))                      what the draft gets wrong on the edited target
  Rspec   = R1 - R0                                 the edit-specific part
          = [lp(T1) - lp(T0)]  -  [lp(D(f1)) - lp(D(f0))]
              target change         free reaction of the draft
Reported (results/geo1c.json):
  energy shares |R0|^2, |Rspec|^2, cosine(R1, R0) per edit; how much of the target change the free reaction covers
  spectrum of Rspec (top-k energy share), and LEAVE-ONE-EDIT-OUT: the share of edit e's Rspec energy (test tokens) inside
  the top-k output subspace built from the OTHER edits' Rspec (fit tokens), vs a random subspace and vs the generic R0
  subspace.
"""
import gc
import sys

import numpy as np
import torch

from config import DRAFTS, MODELS, RESULTS, TARGETS, log, save_json
from geometry1 import (answer_rows, cached_common_texts, center, draft_pass, load_target, prompts, stack, target_pass)

KS = (1, 4, 16, 64)


def main():
    names = sys.argv[1:] or [t for t in TARGETS if t != "base"]
    from staleness import fetch, target_dir
    from native_draft import NativeDraft
    from transformers import AutoTokenizer
    import shutil

    rows = prompts()
    base_path = target_dir("base")
    dpath = fetch(DRAFTS["inst"], MODELS / "draft_inst")
    t0 = load_target(base_path)
    embed = t0.get_input_embeddings().weight.detach().cpu()
    draft = NativeDraft(dpath, embed)
    vocab_ids = draft.d2t_ids
    to_draft = torch.full((embed.shape[0],), -1, dtype=torch.long)
    to_draft[vocab_ids.cpu()] = torch.arange(vocab_ids.numel())
    seqs = cached_common_texts(t0, AutoTokenizer.from_pretrained(str(base_path)), rows)
    f0, lp0, _ = target_pass(t0, seqs, vocab_ids, to_draft)
    del t0
    gc.collect(); torch.cuda.empty_cache()
    _, dl0 = draft_pass(draft, seqs, f0)
    fit, test = answer_rows(seqs, rows, "fit"), answer_rows(seqs, rows, "test")

    def residual(lp, dl, split):
        return (center(stack(lp, split, "t").float()) - center(stack(dl, split, "d").float())).half()

    R0 = {"fit": residual(lp0, dl0, fit), "test": residual(lp0, dl0, test)}
    T0lp = {s: center(stack(lp0, r, "t").float()).half() for s, r in (("fit", fit), ("test", test))}
    D0lp = {s: center(stack(dl0, r, "d").float()).half() for s, r in (("fit", fit), ("test", test))}
    spec, out = {}, {"edits": {}}
    for name in names:
        try:
            t1 = load_target(target_dir(name))
            f1, lp1, _ = target_pass(t1, seqs, vocab_ids, to_draft)
            del t1
            gc.collect(); torch.cuda.empty_cache()
            _, dl1 = draft_pass(draft, seqs, f1)
            spec[name] = {}
            info = {}
            for s, r in (("fit", fit), ("test", test)):
                T1 = center(stack(lp1, r, "t").float())
                D1 = center(stack(dl1, r, "d").float())
                tchange = T1 - T0lp[s].float()                 # what the target changed
                react = D1 - D0lp[s].float()                   # what the draft changed by itself
                rs = tchange - react
                spec[name][s] = rs.half()
                if s == "test":
                    r0 = R0["test"].float()
                    info = {"energy_R0": float(r0.pow(2).sum(1).mean()),
                            "energy_Rspec": float(rs.pow(2).sum(1).mean()),
                            "energy_target_change": float(tchange.pow(2).sum(1).mean()),
                            "free_reaction_covers": float(1 - rs.pow(2).sum() / tchange.pow(2).sum()),
                            "cos_R1_R0": float(((rs + r0) * r0).sum() / ((rs + r0).norm() * r0.norm()))}
            out["edits"][name] = info
            log(f"{name}: |R0|^2 {info['energy_R0']:.1f} |Rspec|^2 {info['energy_Rspec']:.1f} "
                f"|target change|^2 {info['energy_target_change']:.1f} free reaction covers {info['free_reaction_covers']:.3f}")
            del f1, lp1, dl1
        except Exception as error:      # noqa: BLE001
            log(f"FAILED {name}: {error!r}")
        for p in (MODELS / name, MODELS / f"{name}_adapter"):
            shutil.rmtree(p, ignore_errors=True)
        gc.collect(); torch.cuda.empty_cache()

    # output subspaces (right singular vectors over the 32000 draft tokens), from fit tokens
    def basis(M, k):
        _, _, vh = torch.linalg.svd(M.float().cuda(), full_matrices=False)
        return vh[:k]

    def share(M, Q):
        M = M.float().cuda()
        return float((M @ Q.T).pow(2).sum() / M.pow(2).sum())

    kmax = max(KS)
    own = {e: basis(spec[e]["fit"][::2], kmax) for e in spec}
    q0 = basis(R0["fit"][::2], kmax)
    for e in spec:
        st = out["edits"][e]
        st["own_spectrum"] = {str(k): share(spec[e]["test"], own[e][:k]) for k in KS}
        others = torch.cat([own[o][:16] for o in spec if o != e])           # 16 directions from each other edit
        qo, _ = torch.linalg.qr(others.T)
        q_others = qo.T
        st["loo_others_subspace"] = {"dims": int(q_others.shape[0]), "share": share(spec[e]["test"], q_others)}
        rnd, _ = torch.linalg.qr(torch.randn(q_others.shape[1], q_others.shape[0], device="cuda"))
        st["random_subspace_same_dims"] = share(spec[e]["test"], rnd.T)
        st["generic_R0_subspace"] = {str(k): share(spec[e]["test"], q0[:k]) for k in KS}
        log(f"{e}: own top-16 {st['own_spectrum']['16']:.3f} | other edits' {q_others.shape[0]} dirs "
            f"{st['loo_others_subspace']['share']:.3f} | random {st['random_subspace_same_dims']:.3f} | "
            f"generic R0 top-64 {st['generic_R0_subspace']['64']:.3f}")
    save_json(out, RESULTS / "geo1c.json")


if __name__ == "__main__":
    main()
