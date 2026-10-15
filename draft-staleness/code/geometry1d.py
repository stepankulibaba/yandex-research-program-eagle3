"""Step 1d: SELECTIVE restoration of the draft's inputs. Remove the global distortion of the edited target's states but
keep the information about what the target now does.

    python geometry1d.py [TARGET ...]          (default: nemotron_nano r1_distill abliterated)

Common text and fit/test split as in geometry1.py. Every restoration is a function of the NEW states only, fitted on
the fit half from forward passes of T0 and T1 (no generation by T1, no draft training). Depth-1 agreement with T1's
argmax on the test half, reported overall and separately on
  same     tokens where T1 predicts what T0 predicted  (here the new states can only hurt: "damage")
  changed  tokens where T1 now predicts something else (here the new states can help: "transfer")
Variants:
  new / old                       draft reads T1 / T0 states (old = upper bound for "same", lower for "changed")
  mean[L]                         subtract the mean shift, layers L
  diag[L]                         per-dimension affine map new -> old, layers L
  pc{k}[L]                        only along the top-k principal directions v of the shift (fit half): the old
                                  coordinate v.h0 is predicted from v.h1 by a 1-d affine fit; other directions untouched
  pcr{k}[L]                       same directions, old coordinates predicted from the whole h1 by ridge (shrunk to identity)
  L in {29}, {16,29}, {2,16,29}
Writes results/geo1d_<target>.json.
"""
import gc
import shutil
import sys

import torch

from config import DRAFTS, MODELS, RESULTS, log, save_json
from geometry1 import (answer_rows, cached_common_texts, draft_pass, load_target, prompts, stack, target_pass)

LAYER_SETS = {"29": [2], "16+29": [1, 2], "all": [0, 1, 2]}     # indices into the captured layers (2, 16, 29)
KS = (1, 4, 16, 64)
LAMBDAS = (1e-1, 1.0, 10.0, 100.0)


def main():
    names = sys.argv[1:] or ["nemotron_nano", "r1_distill", "abliterated"]
    from staleness import fetch, target_dir
    from native_draft import NativeDraft
    from transformers import AutoTokenizer

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
    f0, _, top0 = target_pass(t0, seqs, vocab_ids, to_draft)
    del t0
    gc.collect(); torch.cuda.empty_cache()
    fit, test = answer_rows(seqs, rows, "fit"), answer_rows(seqs, rows, "test")
    fit_ids = [k for k, r in enumerate(rows) if r["split"] == "fit"]
    last = {}
    for k in fit_ids:
        last[rows[k]["domain"]] = k
    val_ids = set(last.values())
    fit_a = [x for x in fit if x[0] not in val_ids]
    fit_v = [x for x in fit if x[0] in val_ids]
    _, dl0 = draft_pass(draft, seqs, f0)
    H = f0[0].shape[1] // 3
    t0_test = stack(top0, test, "t")

    for name in names:
        out = RESULTS / f"geo1d_{name}.json"
        if out.exists():
            log(f"skip {name}")
            continue
        t1 = load_target(target_dir(name))
        f1, _, top1 = target_pass(t1, seqs, vocab_ids, to_draft)
        del t1
        gc.collect(); torch.cuda.empty_cache()
        tgt = stack(top1, test, "t")
        valid = (tgt >= 0) & (t0_test >= 0)
        same, changed = valid & (tgt == t0_test), valid & (tgt != t0_test)

        def score(dl):
            pred = stack(dl, test, "d").argmax(-1)
            hit = pred == tgt
            return {"all": float(hit.float().mean()), "same": float(hit[same].float().mean()),
                    "changed": float(hit[changed].float().mean())}

        # statistics of the shift on the fit half, per captured layer
        Ff0, Ff1 = stack(f0, fit, "d").float().cuda(), stack(f1, fit, "d").float().cuda()
        Fa0, Fa1 = stack(f0, fit_a, "d").float().cuda(), stack(f1, fit_a, "d").float().cuda()
        Fv0, Fv1 = stack(f0, fit_v, "d").float().cuda(), stack(f1, fit_v, "d").float().cuda()
        stats = []
        for j in range(3):
            sl = slice(j * H, (j + 1) * H)
            x0, x1 = Ff0[:, sl], Ff1[:, sl]
            d = x1 - x0
            mu = d.mean(0)
            a = ((x1 - x1.mean(0)) * (x0 - x0.mean(0))).mean(0) / x1.var(0, unbiased=False).clamp(min=1e-6)
            b = x0.mean(0) - a * x1.mean(0)
            _, _, V = torch.linalg.svd(d - mu, full_matrices=False)
            V = V[:max(KS)]
            # 1-d affine fit of the old coordinate from the new one, per direction (+ the mean direction first)
            P = torch.cat([(mu / mu.norm())[None], V])                      # (1+K) x H, mean direction first
            P, _ = torch.linalg.qr(P.T)
            P = P.T
            c0, c1 = x0 @ P.T, x1 @ P.T
            alpha = ((c1 - c1.mean(0)) * (c0 - c0.mean(0))).mean(0) / c1.var(0, unbiased=False).clamp(min=1e-6)
            beta = c0.mean(0) - alpha * c1.mean(0)
            # ridge from the whole new state to the old coordinates (shrunk to "coordinate unchanged")
            ridge = {}
            Xa = torch.cat([Fa1[:, sl], torch.ones(Fa1.shape[0], 1, device="cuda")], 1)
            Xv = torch.cat([Fv1[:, sl], torch.ones(Fv1.shape[0], 1, device="cuda")], 1)
            G0 = Xa.T @ Xa
            best = None
            for lam in LAMBDAS:
                G = G0 + lam * G0.diagonal().mean() * torch.eye(G0.shape[0], device="cuda")
                Bm = torch.linalg.solve(G, Xa.T @ ((Fa0[:, sl] - Fa1[:, sl]) @ P.T))
                pred = Fv1[:, sl] @ P.T + Xv @ Bm
                err = float((pred - Fv0[:, sl] @ P.T).pow(2).sum())
                if best is None or err < best[1]:
                    best = (lam, err)
            Xf = torch.cat([x1, torch.ones(x1.shape[0], 1, device="cuda")], 1)
            G = Xf.T @ Xf
            G += best[0] * G.diagonal().mean() * torch.eye(G.shape[0], device="cuda")
            ridge = torch.linalg.solve(G, Xf.T @ ((x0 - x1) @ P.T))
            stats.append({"mu": mu, "a": a, "b": b, "P": P, "alpha": alpha, "beta": beta, "ridge": ridge, "lam": best[0]})
        del Ff0, Ff1, Fa0, Fa1, Fv0, Fv1
        torch.cuda.empty_cache()

        def transform(kind, layers, k=None):
            outs = []
            for f in f1:
                g = f.float().cuda()
                for j in layers:
                    s, sl = stats[j], slice(j * H, (j + 1) * H)
                    x = g[:, sl]
                    if kind == "mean":
                        x = x - s["mu"]
                    elif kind == "diag":
                        x = x * s["a"] + s["b"]
                    elif kind in ("pc", "pcr"):
                        P = s["P"][:k + 1]                                  # mean direction + k shift PCs
                        c = x @ P.T
                        if kind == "pc":
                            new_c = c * s["alpha"][:k + 1] + s["beta"][:k + 1]
                        else:
                            xa = torch.cat([x, torch.ones(x.shape[0], 1, device="cuda")], 1)
                            new_c = c + xa @ s["ridge"][:, :k + 1]
                        x = x + (new_c - c) @ P
                    g[:, sl] = x
                outs.append(g.half().cpu())
            return outs

        res = {"target": name, "n_same": int(same.sum()), "n_changed": int(changed.sum()),
               "ridge_lambda": [s["lam"] for s in stats]}
        _, dl1 = draft_pass(draft, seqs, f1)
        res["new"], res["old"] = score(dl1), score(dl0)
        variants = []
        for ls, layers in LAYER_SETS.items():
            variants += [(f"mean[{ls}]", "mean", layers, None), (f"diag[{ls}]", "diag", layers, None)]
            variants += [(f"pc{k}[{ls}]", "pc", layers, k) for k in KS]
            variants += [(f"pcr{k}[{ls}]", "pcr", layers, k) for k in KS]
        for label, kind, layers, k in variants:
            _, dl = draft_pass(draft, seqs, transform(kind, layers, k))
            res[label] = score(dl)
            del dl
        gap = res["old"]["all"] - res["new"]["all"]
        res["recovered_all"] = {lab: (res[lab]["all"] - res["new"]["all"]) / gap for lab, *_ in variants} if gap > 0.005 else None
        save_json(res, out)
        best_lab = max((lab for lab, *_ in variants), key=lambda lab: res[lab]["all"])
        log(f"{name}: new {res['new']} | old {res['old']} | best {best_lab} {res[best_lab]} "
            f"(recovers {res['recovered_all'][best_lab] if res['recovered_all'] else float('nan'):.2f} of the gap)")
        del f1, dl1
        gc.collect(); torch.cuda.empty_cache()
        for p in (MODELS / name, MODELS / f"{name}_adapter"):
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
