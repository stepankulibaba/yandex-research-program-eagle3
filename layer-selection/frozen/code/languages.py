"""The official draft on the same questions in six languages: tau, real speed-up, and why the chains stop.

    python languages.py dsl-8b       -> results/h200/dsl-8b/lang/lang.json, rows_<lang>.json

Prompts are parallel across languages: the same m-ArenaHard question ids and the same MGSM rows
(MGSM has no Ukrainian). For every answer we count
  oov_share       answer tokens outside the draft's 32k vocabulary (the draft can never propose them)
  oov_stop_share  cycles that stopped exactly at such a token
plus a small wall-clock comparison: plain greedy generation vs EAGLE on the same prompts.
"""
import sys
import time

import numpy as np
import torch

from acceptance import measure_acceptance, tau_summary
from config import PROTOCOL, generation_kwargs, log, results_dir, save_json
from eagle_model import chat_prompt, draft_vocab_mask, load_eagle

LANGS = ["en", "ru", "uk", "de", "zh", "ja"]
SCRIPTS = {"cyr": ("Ѐ", "ӿ"), "latin": ("a", "z"), "cjk": ("一", "鿿"), "kana": ("぀", "ヿ")}   # Unicode ranges


def script_shares(text):
    """Share of Cyrillic / Latin / CJK / kana letters in a text (checks the model answers in the asked language)."""
    counts = {k: 0 for k in SCRIPTS}
    for ch in text.lower():
        for k, (lo, hi) in SCRIPTS.items():
            if lo <= ch <= hi:
                counts[k] += 1
    total = max(sum(counts.values()), 1)
    return {k: v / total for k, v in counts.items()}


def lang_prompts(tok, lang, n_arena=60, n_mgsm=60):
    """60 m-ArenaHard + 60 MGSM questions; the same items in every language."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def parquet(repo, lang_):
        return pd.read_parquet(hf_hub_download(repo, f"{lang_}/test-00000-of-00001.parquet", repo_type="dataset"))

    rows = []
    ids = sorted(parquet("CohereLabs/m-ArenaHard", "en")["question_id"])[:n_arena]
    for qid, r in parquet("CohereLabs/m-ArenaHard", lang).set_index("question_id").loc[ids].iterrows():
        rows.append({"id": f"arena/{qid}", "bench": "arena", "text": r["prompt"]})
    if lang != "uk":
        mgsm = parquet("juletxara/mgsm", lang)
        for i in range(n_mgsm):
            rows.append({"id": f"mgsm/{i}", "bench": "mgsm", "text": mgsm.iloc[i]["question"]})
    for r in rows:
        r["ids"] = tok(chat_prompt(tok, r["text"]), return_tensors="pt", add_special_tokens=False).input_ids
    return rows


def stop_reasons(accepted, ids, in_draft):
    """Each cycle emits the accepted draft tokens plus one target token: the token the draft failed to propose.
    Count the cycles that stopped, and how many of them stopped at a token outside the draft vocabulary."""
    start, oov_stops, stops = 1, 0, 0                 # position 0 comes from the prefill
    for a in accepted:
        end = start + a
        if end < len(ids):
            stops += 1
            oov_stops += int(not in_draft[ids[end]])
        start = end + 1
    return stops, oov_stops


def speed_check(model, prompts, gen, n=8):
    """Wall clock: plain greedy generation vs EAGLE on the same prompts."""
    sub = prompts[:n // 2] + [p for p in prompts if p["bench"] == "mgsm"][:n // 2]
    v_tok = v_sec = e_tok = e_sec = 0
    for p in sub:
        ids = p["ids"].cuda()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o = model.naivegenerate(ids.clone(), temperature=0.0, max_new_tokens=PROTOCOL["max_new_tokens"], **gen)
        torch.cuda.synchronize(); v_sec += time.perf_counter() - t0
        v_tok += o.shape[1] - ids.shape[1]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o = model.eagenerate(ids.clone(), temperature=0.0, max_new_tokens=PROTOCOL["max_new_tokens"], **gen)
        torch.cuda.synchronize(); e_sec += time.perf_counter() - t0
        e_tok += o.shape[1] - ids.shape[1]
    return {"prompts": len(sub), "vanilla_tok_s": v_tok / v_sec, "eagle_tok_s": e_tok / e_sec,
            "speedup": (e_tok / e_sec) / (v_tok / v_sec)}


def main(pair):
    out = results_dir(pair) / "lang"
    out.mkdir(exist_ok=True)
    model = load_eagle(pair)
    tok = model.get_tokenizer()
    gen = generation_kwargs(pair)
    in_draft = draft_vocab_mask(model).cpu() if hasattr(model.ea_layer, "t2d") else torch.ones(len(tok), dtype=torch.bool)
    result = {"protocol": PROTOCOL, "draft_vocab": int(in_draft.sum()), "target_vocab": int(in_draft.numel()), "langs": {}}
    for lang in LANGS:
        prompts = lang_prompts(tok, lang)
        rows, outs = measure_acceptance(model, prompts, PROTOCOL["max_new_tokens"], gen_kwargs=gen)
        per = []
        for p, r in zip(prompts, rows):
            answer = outs[p["id"]]
            ids = answer.tolist()
            text = tok.decode(answer, skip_special_tokens=True)
            stops, oov_stops = stop_reasons(r["accepted"], ids, in_draft)
            per.append({**{k: v for k, v in r.items() if k != "accepted"}, "accepted": r["accepted"],
                        "oov_share": float((~in_draft[answer]).float().mean()) if len(ids) else 0.0,
                        "oov_stop_share": oov_stops / max(stops, 1), "stops": stops,
                        "chars": len(text), "scripts": script_shares(text),
                        "prompt_tokens": int(p["ids"].shape[1]), "prompt_chars": len(p["text"])})
        summary = {"all": tau_summary(rows)}
        for bench in ("arena", "mgsm"):
            sub = [x for x in per if x["bench"] == bench]
            if not sub:
                continue
            c = sum(x["cycles"] for x in sub)
            new_tok = sum(x["new_tokens"] for x in sub)
            summary[bench] = {
                "tau": (sum(x["accepted_sum"] for x in sub) + c) / c,
                "oov_share": float(np.average([x["oov_share"] for x in sub], weights=[x["new_tokens"] for x in sub])),
                "oov_stop_share": sum(x["oov_stop_share"] * x["stops"] for x in sub) / max(sum(x["stops"] for x in sub), 1),
                "chars_per_token": sum(x["chars"] for x in sub) / max(new_tok, 1),
                "prompt_chars_per_token": sum(x["prompt_chars"] for x in sub) / sum(x["prompt_tokens"] for x in sub),
                "scripts": {k: float(np.mean([x["scripts"][k] for x in sub])) for k in SCRIPTS},
                "eagle_tokens_per_s": new_tok / sum(x["seconds"] for x in sub),
            }
        summary["speed_subset"] = speed_check(model, prompts, gen)
        result["langs"][lang] = summary
        save_json(per, out / f"rows_{lang}.json")
        save_json(result, out / "lang.json")
        log(f"{pair} {lang}: tau={summary['all']['tau']:.3f} " + " ".join(
            f"{b}: tau={summary[b]['tau']:.2f} oov={summary[b]['oov_share']:.3f} oovstop={summary[b]['oov_stop_share']:.3f}"
            for b in ("arena", "mgsm") if b in summary) + f" speedup={summary['speed_subset']['speedup']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
