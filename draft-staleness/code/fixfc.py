"""Tree-tau test of the selective input restoration, folded into the draft's fc.

    python fixfc.py TARGET [TARGET ...]

Pre-registered variant (chosen before this test, from geometry1d): per captured layer j, the directions P_j = [mean shift,
top-1 principal direction of the shift] (fit half of the common text), and a ridge map from the WHOLE new state to the
change of these two coordinates (lambda on the validation quarter, shrunk to "no change"):
    x' = x + ([x, 1] R_j) P_j  =  x (I + A_j) + c_j,   A_j = R_j[:H] P_j (rank 2),  c_j = R_j[H] P_j
Folded into the draft: W_fc' = W_fc blockdiag(I + A_j^T), bias = W_fc [c_2; c_16; c_29]. Fitted from forward passes of
T0 and T1 on T0's text only: no generation by T1, no draft training.
Then the staleness protocol (staleness.run: authors' EAGLE generation, tree 60/7/10, 96 prompts, T1's own answers)
with the patched draft -> results/<target>__inst_fix.json, to compare with results/<target>__inst.json.
"""
import gc
import sys
import time

import torch
from transformers import AutoModelForCausalLM

from config import DRAFTS, MODELS, RESULTS, log, save_json
from geometry1 import answer_rows, cached_common_texts, prompts, stack

LAYERS = (2, 16, 29)
LAMBDAS = (1e-1, 1.0, 10.0, 100.0)


@torch.no_grad()
def features(path, seqs):
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    out = []
    for ids, _ in seqs:
        hs = model(torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states
        out.append(torch.cat([hs[i][0] for i in LAYERS], -1).half().cpu())
    del model
    gc.collect(); torch.cuda.empty_cache()
    return out


def fit_correction(f0, f1, seqs, rows):
    fit = answer_rows(seqs, rows, "fit")
    last = {}
    for k, r in enumerate(rows):
        if r["split"] == "fit":
            last[r["domain"]] = k
    val = set(last.values())
    fit_a, fit_v = [x for x in fit if x[0] not in val], [x for x in fit if x[0] in val]
    X0, X1 = stack(f0, fit, "d").float().cuda(), stack(f1, fit, "d").float().cuda()
    A0, A1 = stack(f0, fit_a, "d").float().cuda(), stack(f1, fit_a, "d").float().cuda()
    V0, V1 = stack(f0, fit_v, "d").float().cuda(), stack(f1, fit_v, "d").float().cuda()
    H = X0.shape[1] // 3
    blocks, biases, info = [], [], []
    for j in range(3):
        sl = slice(j * H, (j + 1) * H)
        d = X1[:, sl] - X0[:, sl]
        mu = d.mean(0)
        _, _, Vt = torch.linalg.svd(d - mu, full_matrices=False)
        P, _ = torch.linalg.qr(torch.stack([mu / mu.norm(), Vt[0]]).T)
        P = P.T                                                            # 2 x H, orthonormal

        def solve(Xn, Xo, lam):
            Xa = torch.cat([Xn, torch.ones(Xn.shape[0], 1, device="cuda")], 1)
            G = Xa.T @ Xa
            G += lam * G.diagonal().mean() * torch.eye(G.shape[0], device="cuda")
            return torch.linalg.solve(G, Xa.T @ ((Xo - Xn) @ P.T))         # (H+1) x 2

        best = None
        for lam in LAMBDAS:
            R = solve(A1[:, sl], A0[:, sl], lam)
            Xv = torch.cat([V1[:, sl], torch.ones(V1.shape[0], 1, device="cuda")], 1)
            err = float(((V1[:, sl] @ P.T + Xv @ R) - V0[:, sl] @ P.T).pow(2).sum())
            if best is None or err < best[1]:
                best = (lam, err)
        R = solve(X1[:, sl], X0[:, sl], best[0])
        blocks.append(torch.eye(H, device="cuda") + R[:H] @ P)            # I + A_j  (row-vector convention)
        biases.append(R[H] @ P)                                            # c_j
        info.append(best[0])
    return blocks, torch.cat(biases), info


def patch_fc(ea_model, blocks, bias_in):
    fc = ea_model.ea_layer.fc
    W = fc.weight.data.float()                                             # H x 3H
    H = W.shape[0]
    Wn = torch.cat([W[:, j * H:(j + 1) * H] @ blocks[j].T for j in range(3)], 1)
    new = torch.nn.Linear(W.shape[1], H, bias=True).to(fc.weight.device, fc.weight.dtype)
    new.weight.data.copy_(Wn.to(fc.weight.dtype))
    new.bias.data.copy_((W @ bias_in).to(fc.weight.dtype))
    ea_model.ea_layer.fc = new


def main():
    import staleness
    from staleness import fetch, target_dir
    rows = prompts()
    base_path = target_dir("base")
    fetch(DRAFTS["inst"], MODELS / "draft_inst")
    seqs = cached_common_texts(None, None, rows)
    f0 = features(base_path, seqs)
    for name in sys.argv[1:]:
        out = RESULTS / f"{name}__inst_fix.json"
        if out.exists():
            log(f"skip {name}")
            continue
        t_start = time.time()
        tpath = target_dir(name)
        f1 = features(tpath, seqs)
        blocks, bias, lams = fit_correction(f0, f1, seqs, rows)
        log(f"{name}: correction fitted (lambda {lams}) in {time.time() - t_start:.0f} s")
        original_load = staleness.load

        def load_patched(target_path, draft_path):
            model = original_load(target_path, draft_path)
            patch_fc(model, blocks, bias)
            return model

        staleness.load = load_patched
        try:
            staleness.run(name, "inst", tag="_fix")
        finally:
            staleness.load = original_load
        del f1, blocks, bias
        gc.collect(); torch.cuda.empty_cache()
        import shutil
        for p in (MODELS / name, MODELS / f"{name}_adapter"):
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
