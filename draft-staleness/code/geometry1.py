"""Step 1 of the geometry search: how the edited targets move the draft's inputs, and what the draft would have to learn.

    python geometry1.py [TARGET ...]          (default: all edited targets of config.TARGETS)

Common text: the ORIGINAL target T0 answers 48 prompts (8 per domain, up to 384 greedy tokens); every edited target T1
reads exactly these token ids, so all edits are compared on identical inputs. Prompts are split by prompt into a fit
half and a test half (4 + 4 per domain). Everything is measured at draft depth 1 (the draft step that reads target
features), aligned so that draft position t and target logits row t+1 both predict token t+2.

Per edit (results/geo1_<target>.json, factors in geo1_<target>.npz):
  feature geometry   relative shift per captured layer; shift seen through the draft's input W_fc; share of the
                     shift that is a constant offset; share of shift energy in the top-k input directions of W_fc
  behaviour          target argmax change T0 -> T1; KL(T0 || T1) on the draft vocabulary
  draft, depth 1     agreement with T1's argmax when the draft reads T0 features (acc_old), T1 features (acc_new),
                     T1 features minus their mean shift (acc_meanfix)
  what to learn      logit residual R = logsoftmax(T1) - logsoftmax(draft(T1 features)) on the draft vocabulary,
                     regressed on the draft's pre-head states X with ridge + reduced rank k: test agreement for
                     k = 1..256 and full rank; the same with the generic correction fitted on T0 (no edit info).
                     The rank-32 output basis is saved to compare edits with each other later.
"""
import gc
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CODE, DRAFTS, MODELS, RESULTS, TARGETS, log, save_json

PER_DOMAIN, MAX_NEW = 8, 384
LAYERS = (2, 16, 29)
STOP = {128001, 128008, 128009}
RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TOPK_FC = (16, 64, 256)


def prompts():
    rows = [json.loads(x) for x in (CODE / "prompts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for d in sorted({r["domain"] for r in rows}):
        sel = [r for r in rows if r["domain"] == d][:PER_DOMAIN]
        for i, r in enumerate(sel):
            out.append(dict(r, split="fit" if i < PER_DOMAIN // 2 else "test"))
    return out


def load_target(path):
    return AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16, device_map="cuda:0").eval()


@torch.no_grad()
def common_texts(t0, tok, rows):
    seqs = []
    for r in rows:
        text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        pids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        gen = t0.generate(pids, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None,
                          eos_token_id=sorted(STOP), pad_token_id=128009)[0].tolist()
        n = pids.shape[1]
        for i in range(n, len(gen)):
            if gen[i] in STOP:
                gen = gen[:i + 1]
                break
        seqs.append((gen, n))
    return seqs


def cached_common_texts(t0, tok, rows):
    """T0's greedy answers to the 48 prompts, written once to results/common_texts.json and reused."""
    path = RESULTS / "common_texts.json"
    if path.exists():
        return [tuple(x) for x in json.loads(path.read_text())]
    log("T0 writes the common texts")
    seqs = common_texts(t0, tok, rows)
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(seqs))
    return seqs


@torch.no_grad()
def target_pass(model, seqs, vocab_ids, to_draft):
    """Per sequence: captured features [T, 3*H] (fp16, cpu), log-probs on the draft vocabulary [T, V] (fp16, cpu) and
    the target's full-vocabulary argmax as a draft index (-1 when that token is outside the draft vocabulary)."""
    feats, logps, tops = [], [], []
    for ids, _ in seqs:
        out = model(torch.tensor([ids], device="cuda"), output_hidden_states=True)
        feats.append(torch.cat([out.hidden_states[i][0] for i in LAYERS], -1).half().cpu())
        logits = out.logits[0].float()
        logps.append(F.log_softmax(logits, -1)[:, vocab_ids].half().cpu())
        tops.append(to_draft[logits.argmax(-1).cpu()])
        del out, logits
    return feats, logps, tops


@torch.no_grad()
def draft_pass(draft, seqs, feats):
    """Depth-1 pre-head states X [T-1, H] and draft log-probs [T-1, V]; row t predicts token t+2."""
    xs, logps = [], []
    m = draft.model
    for (ids, _), f in zip(seqs, feats):
        m.reset()
        h, _ = m(f[:-1].to("cuda", torch.bfloat16)[None], input_ids=torch.tensor([ids[1:]], device="cuda"), use_cache=True)
        x = m.norm(h)[0]
        xs.append(x.float().cpu())
        logps.append(F.log_softmax(m.lm_head(x).float(), -1).cpu())
    return xs, logps


def answer_rows(seqs, split_rows, split):
    """(sequence index, draft rows t, target rows t+1) for answer positions of one split."""
    out = []
    for k, ((ids, n), r) in enumerate(zip(seqs, split_rows)):
        if r["split"] != split:
            continue
        t = torch.arange(n - 1, len(ids) - 2)
        out.append((k, t, t + 1))
    return out


def stack(parts, rows, which):
    """Concatenate per-sequence tensors at the selected rows. which: 'd' draft rows, 't' target rows."""
    return torch.cat([parts[k][(td if which == "d" else tt)] for k, td, tt in rows])


FLOOR = -15.0      # log-probs below this are noise for argmax agreement; without the floor the regression chases the tail


def center(z):
    z = z.clamp(min=FLOOR)
    return z - z.mean(-1, keepdim=True)


def ridge_rrr(X, Y, lam_rel=1e-1):
    """Ridge B (H x V) and the SVD of L^T B (L: Cholesky of X^T X + lam I) for reduced-rank truncation."""
    X, Y = X.cuda(), Y.cuda()
    G = X.T @ X
    lam = lam_rel * G.diagonal().mean()
    G += lam * torch.eye(G.shape[0], device="cuda")
    B = torch.linalg.solve(G, X.T @ Y)
    L = torch.linalg.cholesky(G)
    _, S, Vh = torch.linalg.svd(L.T @ B, full_matrices=False)
    return B, S, Vh


def agree(logits, target_ids):
    return float((logits.argmax(-1) == target_ids).float().mean())


def main():
    names = sys.argv[1:] or [t for t in TARGETS if t != "base"]
    from staleness import fetch, target_dir
    from native_draft import NativeDraft

    rows = prompts()
    base_path = target_dir("base")
    dpath = fetch(DRAFTS["inst"], MODELS / "draft_inst")
    tok0 = AutoTokenizer.from_pretrained(str(base_path))
    t0 = load_target(base_path)
    embed = t0.get_input_embeddings().weight.detach().cpu()
    draft = NativeDraft(dpath, embed)
    vocab_ids = draft.d2t_ids                                   # target ids of the 32k draft tokens
    to_draft = torch.full((embed.shape[0],), -1, dtype=torch.long)
    to_draft[vocab_ids.cpu()] = torch.arange(vocab_ids.numel())

    seqs = cached_common_texts(t0, tok0, rows)
    f0, lp0, top0 = target_pass(t0, seqs, vocab_ids, to_draft)
    del t0
    gc.collect(); torch.cuda.empty_cache()

    # the draft's input sensitivity: right singular vectors of W_fc (H x 3H)
    Wfc = draft.model.fc.weight.detach().float()
    _, s_fc, vh_fc = torch.linalg.svd(Wfc, full_matrices=False)
    vh_fc = vh_fc[:max(TOPK_FC)].cpu()

    x0, dl0 = draft_pass(draft, seqs, f0)
    fit, test = answer_rows(seqs, rows, "fit"), answer_rows(seqs, rows, "test")
    # generic head correction: fitted on T0 itself (no information about any edit)
    X0 = stack(x0, fit, "d")
    Y0 = center(stack(lp0, fit, "t").float()) - center(stack(dl0, fit, "d"))
    B0, _, _ = ridge_rrr(X0, Y0)
    tgt0_test = stack(top0, test, "t")
    base = {"acc_base_test": agree(stack(dl0, test, "d"), tgt0_test),
            "target_top_outside_draft_vocab": float((tgt0_test < 0).float().mean())}
    log(f"base: depth-1 agreement of the draft with T0 on the test half {base['acc_base_test']:.3f}")
    save_json({"base": base, "n_fit_tokens": int(X0.shape[0]), "n_test_tokens": int(tgt0_test.numel())},
              RESULTS / "geo1_base.json")
    del X0, Y0

    for name in names:
        out_json = RESULTS / f"geo1_{name}.json"
        if out_json.exists():
            log(f"skip {name}")
            continue
        try:
            t1 = load_target(target_dir(name))
            f1, lp1, top1 = target_pass(t1, seqs, vocab_ids, to_draft)
            del t1
            gc.collect(); torch.cuda.empty_cache()
            res = {"target": name}

            # --- feature geometry on answer positions (all prompts) ---
            allr = fit + test
            F0 = stack(f0, allr, "d").float().cuda()
            D = stack(f1, allr, "d").float().cuda() - F0
            H = F0.shape[1] // 3
            res["shift_rel"] = [float((D[:, j * H:(j + 1) * H].norm(dim=1) / F0[:, j * H:(j + 1) * H].norm(dim=1)).mean())
                                for j in range(3)]
            mu = D.mean(0)
            res["offset_share"] = [float(mu[j * H:(j + 1) * H].pow(2).sum() / D[:, j * H:(j + 1) * H].pow(2).sum(1).mean())
                                   for j in range(3)]
            Wc, vh = Wfc.cuda(), vh_fc.cuda()
            fcD, fc0 = D @ Wc.T, F0 @ Wc.T
            res["shift_fc_rel"] = float((fcD.norm(dim=1) / fc0.norm(dim=1)).mean())
            res["offset_share_fc"] = float((Wc @ mu).pow(2).sum() / fcD.pow(2).sum(1).mean())
            res["energy_in_fc_top"] = {str(k): float((D @ vh[:k].T).pow(2).sum() / D.pow(2).sum()) for k in TOPK_FC}
            res["energy_in_fc_top_base_h"] = {str(k): float((F0 @ vh[:k].T).pow(2).sum() / F0.pow(2).sum()) for k in TOPK_FC}
            # principal directions of the shift (for cross-edit comparison)
            Dc = D - mu
            _, sD, vD = torch.linalg.svd(Dc[::2], full_matrices=False)
            mu = mu.cpu()
            eD = (sD ** 2 / (sD ** 2).sum()).cpu()
            res["shift_pca_share"] = {str(k): float(eD[:k].sum()) for k in (1, 4, 16, 64)}
            pca = vD[:32].cpu().half().numpy()
            del F0, D, Dc, fcD, fc0, Wc, vh

            # --- target behaviour and draft agreement, test half ---
            lp0t, lp1t = stack(lp0, test, "t").float(), stack(lp1, test, "t").float()
            tgt1 = stack(top1, test, "t")
            res["target_argmax_change"] = float((stack(top0, test, "t") != tgt1).float().mean())
            res["target_top_outside_draft_vocab"] = float((tgt1 < 0).float().mean())
            res["target_kl"] = float((lp0t.exp() * (lp0t - lp1t)).sum(-1).mean())
            del lp0t
            x1, dl1 = draft_pass(draft, seqs, f1)
            f1m = [f - mu.half() for f in f1]
            _, dlm = draft_pass(draft, seqs, f1m)
            dl0t, dl1t = stack(dl0, test, "d"), stack(dl1, test, "d")
            res["acc_old"] = agree(dl0t, tgt1)
            res["acc_new"] = agree(dl1t, tgt1)
            res["acc_meanfix"] = agree(stack(dlm, test, "d"), tgt1)
            res["draft_kl_old_new"] = float((dl0t.exp() * (dl0t - dl1t)).sum(-1).mean())
            res["draft_top1_change"] = float((dl0t.argmax(-1) != dl1t.argmax(-1)).float().mean())
            del dlm, f1m

            # --- what the draft would have to learn: residual in log-probs, reduced-rank on X ---
            X1f, X1t = stack(x1, fit, "d"), stack(x1, test, "d")
            R = center(stack(lp1, fit, "t").float()) - center(stack(dl1, fit, "d"))
            B, S, Vh = ridge_rrr(X1f, R)
            Xt = X1t.cuda()
            base_logits = dl1t.cuda()
            res["acc_generic_head"] = agree((base_logits + Xt @ B0), tgt1.cuda())
            res["acc_full_rank"] = agree((base_logits + Xt @ B), tgt1.cuda())
            acc_k = {}
            for k in RANKS:
                Bk = (B @ Vh[:k].T) @ Vh[:k]
                acc_k[str(k)] = agree(base_logits + Xt @ Bk, tgt1.cuda())
            res["acc_rank"] = acc_k
            res["residual_spectrum_share"] = {str(k): float((S[:k] ** 2).sum() / (S ** 2).sum()) for k in RANKS}
            np.savez_compressed(RESULTS / f"geo1_{name}.npz", mu=mu.half().numpy(), shift_pca=pca,
                                head_out_basis=Vh[:32].half().cpu().numpy(), head_sv=S[:256].cpu().numpy())
            res["n_fit_tokens"], res["n_test_tokens"] = int(X1f.shape[0]), int(tgt1.numel())
            save_json(res, out_json)
            log(f"{name}: shift {[round(v, 3) for v in res['shift_rel']]} fc {res['shift_fc_rel']:.3f} offset "
                f"{[round(v, 2) for v in res['offset_share']]} | acc old {res['acc_old']:.3f} new {res['acc_new']:.3f} "
                f"meanfix {res['acc_meanfix']:.3f} generic {res['acc_generic_head']:.3f} rank16 {acc_k['16']:.3f} "
                f"rank64 {acc_k['64']:.3f} full {res['acc_full_rank']:.3f}")
            del x1, dl1, f1, lp1, B, Xt, base_logits
        except Exception as error:      # noqa: BLE001
            log(f"FAILED {name}: {error!r}")
            import traceback
            traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
        if name != "base":
            import shutil
            for p in (MODELS / name, MODELS / f"{name}_adapter"):
                shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
