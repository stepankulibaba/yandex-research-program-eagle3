"""Stage 1 geometry: how do public EAGLE-3 drafts for the same target differ, in quantities that are invariant
to the internal symmetries of independently trained drafts (neuron / head permutations, residual rotations).

Invariant views used
  * fc read-geometry: W_fc acts on TARGET feature space, which is shared by all drafts. Its right singular
    subspaces and per-layer block energies are comparable across drafts without any alignment.
  * fc output on data: linear CKA between g = W_fc x of two drafts on the same target features
    (invariant to orthogonal maps of the draft residual space).
  * lm_head token geometry: linear CKA of lm_head rows over tokens shared by both draft vocabularies.
  * provenance: raw cosine of weights; high values reveal a shared initialisation.
"""
import itertools
import json
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent          # target/, zoo/, EAGLE/, results/ sit next to code/
sys.path.insert(0, str(CODE))
import zoo  # noqa: E402
import zoo_eval  # noqa: E402

OUT = ROOT / "results/zoo"


def cka(x, y):
    x = x.double() - x.double().mean(0)
    y = y.double() - y.double().mean(0)
    hxy = (x.T @ y).pow(2).sum()
    return float(hxy / torch.sqrt((x.T @ x).pow(2).sum() * (y.T @ y).pow(2).sum()))


_VH = {}


def right_basis(key, w):
    if key not in _VH:
        _VH[key] = torch.linalg.svd(w.float(), full_matrices=False).Vh
    return _VH[key]


def subspace_overlap(va, vb, r):
    """Mean cos^2 of principal angles between the top-r right singular subspaces."""
    return float(torch.linalg.svdvals(va[:r] @ vb[:r].T).pow(2).mean())


def main(n_texts=24):
    tgt = zoo_eval.Target()
    N = tgt.N
    drafts = zoo_eval.load_drafts(N, tgt.model.model.embed_tokens.weight)
    drafts.pop("deepseek_ttt7_plus1", None)
    H = next(iter(drafts.values())).H
    res = {"blocks": {}, "pairs": {}}

    # Target features on a few texts of each mode.
    feats = {}
    for mode in ("nothink", "think"):
        p = OUT / f"texts_{mode}.pt"
        if p.exists():
            allseq = torch.load(p)
            seqs = allseq[:: max(1, len(allseq) // n_texts)][:n_texts]
            fs = []
            for s in seqs:
                f, _ = tgt.feats(s["ids"].cuda())
                fs.append(f[s["mask"].cuda().bool()].to(torch.bfloat16))
            feats[mode] = torch.cat(fs)
    all_feats = torch.cat(list(feats.values()))
    sub = all_feats[torch.randperm(all_feats.shape[0])[:6000]]

    # Per-draft: block energies of fc (weight-only and data-weighted), per mode.
    gs = {}
    for n, d in drafts.items():
        Wfc = d.W["fc"].float()
        blocks = Wfc.split(H, dim=1)
        info = {"layer_ids": d.layer_ids,
                "weight_energy_share": [float(b.pow(2).sum() / Wfc.pow(2).sum()) for b in blocks]}
        for mode, f in feats.items():
            x = f[:, d.layer_ids].float()
            c = [(x[:, i] @ blocks[i].T).pow(2).sum(-1).mean() for i in range(len(blocks))]
            tot = sum(c)
            info[f"data_energy_share_{mode}"] = [float(ci / tot) for ci in c]
        res["blocks"][n] = info
        gs[n] = d.fuse(sub).float()

    # Pairwise invariant similarities.
    names = list(drafts)
    for a, b in itertools.combinations(names, 2):
        da, db = drafts[a], drafts[b]
        pr = {}
        # provenance: raw cosine of same-shaped tensors
        for k in ("q", "k", "v", "o", "gate", "up", "down", "fc"):
            wa, wb = da.W.get(k), db.W.get(k)
            if wa is not None and wb is not None and wa.shape == wb.shape:
                pr[f"cos_{k}"] = float(torch.nn.functional.cosine_similarity(wa.float().flatten(), wb.float().flatten(), dim=0))
        # fc read subspaces per shared target layer
        shared = sorted(set(da.layer_ids) & set(db.layer_ids))
        for layer in shared:
            ba = da.W["fc"][:, da.layer_ids.index(layer) * da.H:(da.layer_ids.index(layer) + 1) * da.H]
            bb = db.W["fc"][:, db.layer_ids.index(layer) * db.H:(db.layer_ids.index(layer) + 1) * db.H]
            va, vb = right_basis((a, layer), ba), right_basis((b, layer), bb)
            for r in (64, 512):
                pr[f"fc_subspace_L{layer}_r{r}"] = subspace_overlap(va, vb, r)
        pr["cka_fc_output"] = cka(gs[a], gs[b])
        # lm_head geometry over shared tokens
        common = (da.in_vocab & db.in_vocab).nonzero().squeeze(1)
        if len(common) > 1000:
            ia = torch.full((da.vocab_target,), -1, device=common.device, dtype=torch.long)
            ia[da.d2t_ids] = torch.arange(da.draft_vocab, device=common.device)
            ib = torch.full((db.vocab_target,), -1, device=common.device, dtype=torch.long)
            ib[db.d2t_ids] = torch.arange(db.draft_vocab, device=common.device)
            sel = common[torch.randperm(len(common), device=common.device)[:8000]]
            pr["cka_lm_head_rows"] = cka(da.W["lm_head"][ia[sel]].float(), db.W["lm_head"][ib[sel]].float())
            pr["shared_vocab"] = int(len(common))
        res["pairs"][f"{a}|{b}"] = pr
        print(a, b, {k: round(v, 3) for k, v in pr.items() if isinstance(v, float)}, flush=True)
    (OUT / "zoo_geometry.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
