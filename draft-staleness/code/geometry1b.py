"""Step 1b: can the draft's inputs be restored, and how do the target's PARAMETERS change compared to its STATES.

    python geometry1b.py [TARGET ...]          (default: all edited targets)

Same common text and fit/test split as geometry1.py (results/common_texts.json). Per edit T1 (results/geo1b_<t>.json):

  restoration   the draft reads T1 features mapped back towards T0's, fitted on the fit half only (forward passes of
                T0 and T1 on any text: no generation by T1, no draft training); depth-1 agreement with T1's argmax on
                the test half for: new features, old features (upper bound), mean shift removed, per-dimension affine,
                per-layer ridge (lambda chosen on a held-out quarter of the fit prompts), and R^2 of the restored states
  parameters    every matrix of every layer: relative change ||W1-W0||/||W0|| and how concentrated the change is
                (energy of the top 1/8/32 singular directions, randomized SVD); for the matrices that write into the
                residual stream (o_proj, down_proj) the top-8 left singular vectors are saved, to compare the
                directions the weights move the residual stream in with the directions the states actually move
  data          for offline analysis (geo1b_<t>.npz): states of the 3 captured layers for 1536 test tokens of T1,
                T0's states for the same tokens (geo1b_base.npz), target top-1 ids, draft top-1 ids old/new
"""
import gc
import json
import shutil
import sys

import numpy as np
import torch

from config import DRAFTS, MODELS, RESULTS, TARGETS, log, save_json
from geometry1 import (agree, answer_rows, cached_common_texts, draft_pass, load_target, prompts, stack,
                       target_pass)

N_SAVE = 1536
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
WRITE = ("self_attn.o_proj", "mlp.down_proj")
MODULES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


@torch.no_grad()
def weight_geometry(t0, t1):
    p0 = dict(t0.named_parameters())
    rel, top, left = {}, {}, {}
    for name, w1 in t1.named_parameters():
        w0 = p0[name]
        d = (w1.float() - w0.float())
        n0, nd = float(w0.float().norm()), float(d.norm())
        rel[name] = nd / max(n0, 1e-12)
        if d.dim() == 2 and nd > 0 and any(m in name for m in MODULES):
            u, s, _ = torch.svd_lowrank(d, q=40, niter=2)
            e = (s ** 2) / nd ** 2
            top[name] = [float(e[:1].sum()), float(e[:8].sum()), float(e[:32].sum())]
            if any(m in name for m in WRITE):
                left[name] = u[:, :8].half().cpu().numpy()
        del d
    return rel, top, left


def ridge_map(X, Y, lam_rel):
    """Y ~ X + [X, 1] B: ridge on the CORRECTION B ((H+1) x H), so strong regularisation means the identity map
    (a plain ridge Y ~ [X,1]B shrinks to zero and overfits: 4097 x 4096 parameters per layer from ~6k tokens)."""
    Xa = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1)
    G = Xa.T @ Xa
    G += lam_rel * G.diagonal().mean() * torch.eye(G.shape[0], device=X.device)
    return torch.linalg.solve(G, Xa.T @ (Y - X))


def apply_map(f, maps, H):
    out = []
    for j, B in enumerate(maps):
        x = f[:, j * H:(j + 1) * H].float().cuda()
        out.append(x + torch.cat([x, torch.ones(x.shape[0], 1, device="cuda")], 1) @ B)
    return torch.cat(out, 1).half().cpu()


def r2(pred, target):
    return float(1 - (pred - target).pow(2).sum() / (target - target.mean(0)).pow(2).sum())


def main():
    names = sys.argv[1:] or [t for t in TARGETS if t != "base"]
    from staleness import fetch, target_dir
    from native_draft import NativeDraft

    rows = prompts()
    base_path = target_dir("base")
    dpath = fetch(DRAFTS["inst"], MODELS / "draft_inst")
    from transformers import AutoTokenizer
    tok0 = AutoTokenizer.from_pretrained(str(base_path))
    t0 = load_target(base_path)
    embed = t0.get_input_embeddings().weight.detach().cpu()
    draft = NativeDraft(dpath, embed)
    vocab_ids = draft.d2t_ids
    to_draft = torch.full((embed.shape[0],), -1, dtype=torch.long)
    to_draft[vocab_ids.cpu()] = torch.arange(vocab_ids.numel())

    seqs = cached_common_texts(t0, tok0, rows)
    f0, lp0, top0 = target_pass(t0, seqs, vocab_ids, to_draft)
    del lp0
    fit, test = answer_rows(seqs, rows, "fit"), answer_rows(seqs, rows, "test")
    # lambda selection: the last fit prompt of every domain is the validation quarter
    fit_ids = [k for k, r in enumerate(rows) if r["split"] == "fit"]
    by_domain = {}
    for k in fit_ids:
        by_domain.setdefault(rows[k]["domain"], []).append(k)
    val_ids = {v[-1] for v in by_domain.values()}
    fit_a = [x for x in fit if x[0] not in val_ids]
    fit_v = [x for x in fit if x[0] in val_ids]

    # the tokens saved for offline analysis: an even subsample of test answer positions
    test_t = stack([torch.arange(len(i)) for i, _ in seqs], test, "d")
    pick = torch.linspace(0, test_t.numel() - 1, N_SAVE).long()
    F0_test = stack(f0, test, "d")
    np.savez_compressed(RESULTS / "geo1b_base.npz", f0=F0_test[pick].numpy(), top0=stack(top0, test, "t")[pick].numpy())
    x0, dl0 = draft_pass(draft, seqs, f0)
    H = F0_test.shape[1] // 3

    for name in names:
        out = RESULTS / f"geo1b_{name}.json"
        if out.exists():
            log(f"skip {name}")
            continue
        try:
            t1 = load_target(target_dir(name))
            rel, top, left = weight_geometry(t0, t1)
            f1, _, top1 = target_pass(t1, seqs, vocab_ids, to_draft)
            del t1
            gc.collect(); torch.cuda.empty_cache()
            tgt1 = stack(top1, test, "t")
            res = {"target": name}

            # restoration maps, fitted on the fit half (lambda chosen on its validation quarter)
            Fa0, Fa1 = stack(f0, fit_a, "d"), stack(f1, fit_a, "d")
            Fv0, Fv1 = stack(f0, fit_v, "d"), stack(f1, fit_v, "d")
            best = None
            for lam in LAMBDAS:
                maps = [ridge_map(Fa1[:, j * H:(j + 1) * H].float().cuda(), Fa0[:, j * H:(j + 1) * H].float().cuda(), lam)
                        for j in range(3)]
                score = r2(apply_map(Fv1, maps, H).float(), Fv0.float())
                if best is None or score > best[1]:
                    best = (lam, score)
            Ff0, Ff1 = stack(f0, fit, "d"), stack(f1, fit, "d")
            maps = [ridge_map(Ff1[:, j * H:(j + 1) * H].float().cuda(), Ff0[:, j * H:(j + 1) * H].float().cuda(), best[0])
                    for j in range(3)]
            mu = (Ff1.float() - Ff0.float()).mean(0)
            a = ((Ff1.float() - Ff1.float().mean(0)) * (Ff0.float() - Ff0.float().mean(0))).mean(0) / \
                Ff1.float().var(0, unbiased=False).clamp(min=1e-6)
            b = Ff0.float().mean(0) - a * Ff1.float().mean(0)
            del Fa0, Fa1, Fv0, Fv1, Ff0, Ff1

            variants = {"new": f1, "old": f0,
                        "meanfix": [f - mu.half() for f in f1],
                        "diag": [(f.float() * a + b).half() for f in f1],
                        "ridge": [apply_map(f, maps, H) for f in f1]}
            F0t = F0_test.float()
            for v, feats in variants.items():
                if v == "old":
                    dl = dl0
                else:
                    _, dl = draft_pass(draft, seqs, feats)
                res[f"acc_{v}"] = agree(stack(dl, test, "d"), tgt1)
                res[f"r2_{v}"] = r2(stack(feats, test, "d").float(), F0t) if v != "old" else 1.0
                if v == "new":
                    dl_new = dl
            gap = res["acc_old"] - res["acc_new"]
            res["gap_old_new"] = gap
            res["recovered_ridge"] = (res["acc_ridge"] - res["acc_new"]) / gap if gap > 0.01 else None
            res["recovered_diag"] = (res["acc_diag"] - res["acc_new"]) / gap if gap > 0.01 else None
            res["ridge_lambda"], res["ridge_val_r2"] = best
            res["weights_rel"] = rel
            res["weights_top_energy"] = top
            save_json(res, out)
            np.savez_compressed(RESULTS / f"geo1b_{name}.npz",
                                f1=stack(f1, test, "d")[pick].numpy(), top1=tgt1[pick].numpy(),
                                draft_old=stack(dl0, test, "d")[pick].argmax(-1).numpy(),
                                draft_new=stack(dl_new, test, "d")[pick].argmax(-1).numpy(),
                                **{f"left::{k}": v for k, v in left.items()})
            log(f"{name}: acc new {res['acc_new']:.3f} old {res['acc_old']:.3f} meanfix {res['acc_meanfix']:.3f} "
                f"diag {res['acc_diag']:.3f} ridge {res['acc_ridge']:.3f} (lambda {best[0]}, val R2 {best[1]:.3f}) | "
                f"R2 new {res['r2_new']:.3f} diag {res['r2_diag']:.3f} ridge {res['r2_ridge']:.3f}")
            del f1, variants, dl, dl_new
        except Exception as error:      # noqa: BLE001
            log(f"FAILED {name}: {error!r}")
            import traceback
            traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
        for p in (MODELS / name, MODELS / f"{name}_adapter"):
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
