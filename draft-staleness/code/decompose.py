"""Where the draft's loss on a fine-tuned target comes from: vocabulary, free transfer through the new features, rest.

    python decompose.py TARGET [TARGET ...]        (each needs models/<TARGET>, plus models/base and models/draft_inst)

For 8 prompts per domain the edited target T1 writes its own greedy answer (up to 384 tokens). On exactly these
token ids we evaluate the frozen official draft D0 (authors' CNets, real incremental KV, chain of up to 6
proposals, see native_draft.py) twice:
  tau_new_features  D0 reads T1's hidden states       = what one gets in deployment
  tau_old_features  D0 reads T0's hidden states        = as if the draft could not see the edit at all
so  free_transfer = tau_new_features - tau_old_features  on the same text.
For reference, T0 answers the same prompts and D0 is scored on T0's own text with T0's features (tau_base).
Also: relative shift of the three captured hidden states, share of first rejections at out-of-vocabulary tokens.
Writes results/decomp_<TARGET>.json.
"""
import gc
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CODE, MODELS, RESULTS, log, save_json

PER_DOMAIN, MAX_NEW, DEPTH = 8, 384, 6
LAYERS = (2, 16, 29)          # EAGLE-3 captures the inputs of layers 2, N/2, N-3 of the 32-layer target
STOP = {128001, 128008, 128009}


def prompts():
    rows = [json.loads(x) for x in (CODE / "prompts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for d in sorted({r["domain"] for r in rows}):
        out += [r for r in rows if r["domain"] == d][:PER_DOMAIN]
    return out


def load(path):
    return AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16, device_map="cuda:0").eval()


@torch.no_grad()
def greedy(model, tok, rows):
    """Own greedy answers, one prompt at a time (no padding effects). Returns [(ids incl. prompt, prompt_len)]."""
    out = []
    for r in rows:
        text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        pids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        gen = model.generate(pids, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None,
                             eos_token_id=sorted(STOP), pad_token_id=128009)
        ids = gen[0].tolist()
        n = pids.shape[1]
        for i in range(n, len(ids)):           # keep the first stop token, drop anything after it
            if ids[i] in STOP:
                ids = ids[:i + 1]
                break
        out.append((ids, n))
    return out


@torch.no_grad()
def features(model, ids):
    hs = model(torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states
    return torch.stack([hs[i][0] for i in LAYERS], 1)          # [T, 3, H]


def oov_first_rejections(pred, ids, prompt_len, in_draft):
    """Share of cycles that ended at a target token the draft cannot propose at all."""
    t, stops, oov = prompt_len - 1, 0, 0
    while t + 1 < len(ids):
        length = 1
        for k in range(pred.shape[0]):
            pos = t + k + 2
            if pos >= len(ids):
                break
            if pred[k, t] != ids[pos]:
                stops += 1
                oov += int(not in_draft[ids[pos]])
                break
            length += 1
        t += length
    return stops, oov


def pooled(scores):
    return sum(s["tokens"] for s in scores) / max(sum(s["cycles"] for s in scores), 1)


def decompose(target, base_cache):
    from native_draft import NativeDraft
    rows = prompts()
    tok = AutoTokenizer.from_pretrained(str(MODELS / target))
    t0 = load(MODELS / "base")
    embed = t0.get_input_embeddings().weight.detach().cpu()
    draft = NativeDraft(MODELS / "draft_inst", embed)
    in_draft = np.zeros(embed.shape[0], dtype=bool)
    in_draft[draft.d2t_ids.cpu().numpy()] = True

    if "rows" not in base_cache:            # T0 on its own text: computed once per process
        tok0 = AutoTokenizer.from_pretrained(str(MODELS / "base"))
        seqs0 = greedy(t0, tok0, rows)
        base_rows = []
        for r, (ids, n) in zip(rows, seqs0):
            _, s = draft.eval_chain(features(t0, ids), ids, n, DEPTH)
            base_rows.append({"domain": r["domain"], "tokens": s["tokens"], "cycles": s["cycles"]})
        base_cache["rows"] = base_rows

    t1 = load(MODELS / target)
    seqs = greedy(t1, tok, rows)
    per = []
    for r, (ids, n) in zip(rows, seqs):
        f1, f0 = features(t1, ids), features(t0, ids)
        shift = [float(((f1[n - 1:, j] - f0[n - 1:, j]).float().norm(dim=-1) / f0[n - 1:, j].float().norm(dim=-1)).mean())
                 for j in range(3)]
        p1, s1 = draft.eval_chain(f1, ids, n, DEPTH)
        p0, s0 = draft.eval_chain(f0, ids, n, DEPTH)
        stops, oov = oov_first_rejections(p1, np.array(ids), n, in_draft)
        answer = np.array(ids[n:])
        per.append({"id": r["id"], "domain": r["domain"], "new_tokens": len(answer),
                    "tokens": s1["tokens"], "cycles_new": s1["cycles"], "cycles_old": s0["cycles"],
                    "tau_new_features": s1["tau"], "tau_old_features": s0["tau"],
                    "oov_answer_share": float((~in_draft[answer]).mean()) if len(answer) else 0.0,
                    "stops": stops, "oov_stops": oov, "feature_shift": shift})
    del t0, t1
    gc.collect()
    torch.cuda.empty_cache()

    def summary(sel, base_sel):
        tok_ = sum(x["tokens"] for x in sel)
        new = tok_ / max(sum(x["cycles_new"] for x in sel), 1)
        old = tok_ / max(sum(x["cycles_old"] for x in sel), 1)
        return {"n": len(sel), "tau_base": pooled(base_sel), "tau_new_features": new, "tau_old_features": old,
                "free_transfer": new - old,
                "oov_stop_share": sum(x["oov_stops"] for x in sel) / max(sum(x["stops"] for x in sel), 1),
                "oov_answer_share": float(np.mean([x["oov_answer_share"] for x in sel])),
                "feature_shift": [float(np.mean([x["feature_shift"][j] for x in sel])) for j in range(3)]}

    base_rows = base_cache["rows"]
    result = {"target": target, "per_domain": PER_DOMAIN, "max_new_tokens": MAX_NEW, "depth": DEPTH, "layers": LAYERS,
              "all": summary(per, base_rows),
              "domains": {d: summary([x for x in per if x["domain"] == d], [x for x in base_rows if x["domain"] == d])
                          for d in sorted({x["domain"] for x in per})},
              "rows": per}
    save_json(result, RESULTS / f"decomp_{target}.json")
    a = result["all"]
    log(f"decomp {target}: base {a['tau_base']:.2f} | new feats {a['tau_new_features']:.2f} | old feats "
        f"{a['tau_old_features']:.2f} | free {a['free_transfer']:+.2f} | oov stops {a['oov_stop_share']:.2f} | "
        f"shift {[round(s, 3) for s in a['feature_shift']]}")


if __name__ == "__main__":
    cache = {}
    for name in sys.argv[1:]:
        decompose(name, cache)
