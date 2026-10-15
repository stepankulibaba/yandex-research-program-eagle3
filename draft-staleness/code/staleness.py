"""Drafter staleness map: the official EAGLE-3 draft of Llama-3.1-8B-Instruct on fine-tuned versions of its target.

    python staleness.py [--runs base:inst toolace:inst ...] [--keep-models]

For every (target, draft) run in config.RUNS: real EAGLE generation (authors' code, tree 60/7/10, greedy, up to
512 new tokens) on the 96 prompts of prompts.jsonl (6 domains x 16), and per answer
  tau             (accepted draft tokens + cycles) / cycles
  oov_share       answer tokens outside the draft's 32k vocabulary (the draft can never propose them)
  oov_stop_share  cycles that stopped exactly at such a token
plus wall-clock speed-up (plain greedy vs EAGLE) on 12 prompts (2 per domain).
Writes results/<target>__<draft>.json; a run that is already there is skipped (restartable).
"""
import argparse
import gc
import json
import shutil
import time

import numpy as np
import torch

from acceptance import measure_acceptance
from config import CODE, DRAFTS, MODELS, PROTOCOL, RESULTS, RUNS, TARGETS, log, save_json
from eagle_model import draft_vocab_mask, patch_eagle, seed_all

GEN = {"is_llama3": True, "max_length": PROTOCOL["max_length"]}
PATTERNS = ["*.json", "*.safetensors", "*.bin", "tokenizer*", "*.model", "*.txt", "*.jinja"]


def fetch(repo, path):
    from huggingface_hub import snapshot_download
    if not (path / "config.json").exists() and not (path / "adapter_config.json").exists():
        log(f"downloading {repo}")
        snapshot_download(repo, local_dir=str(path), allow_patterns=PATTERNS, ignore_patterns=["original/*", "training_args.bin"])
    return path


def target_dir(name):
    spec = TARGETS[name]
    path = MODELS / name
    if isinstance(spec, str):
        return fetch(spec, path)
    if (path / "config.json").exists():
        return path
    # LoRA adapter: merge into the base once and save a plain checkpoint
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = fetch(spec["base"], MODELS / "base")
    adapter = fetch(spec["lora"], MODELS / f"{name}_adapter")
    log(f"merging {spec['lora']} into {spec['base']}")
    model = AutoModelForCausalLM.from_pretrained(str(base), torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(path), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(str(path))
    del model
    gc.collect()
    return path


def load(target_path, draft_path):
    patch_eagle()
    seed_all(0)
    from eagle.model.ea_model import EaModel
    model = EaModel.from_pretrained(
        base_model_path=str(target_path), ea_model_path=str(draft_path),
        total_token=PROTOCOL["total_token"], depth=PROTOCOL["depth"], top_k=PROTOCOL["top_k"],
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda:0", use_eagle3=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def prompts_for(tok):
    rows = [json.loads(x) for x in (CODE / "prompts.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        r["ids"] = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        r["bench"] = r["domain"]
    return rows


def stop_reasons(accepted, ids, in_draft):
    """Cycles that stopped before the end of the answer, and how many stopped at a token outside the draft vocab."""
    start, oov_stops, stops = 1, 0, 0                 # position 0 comes from the prefill
    for a in accepted:
        end = start + a
        if end < len(ids):
            stops += 1
            oov_stops += int(not in_draft[ids[end]])
        start = end + 1
    return stops, oov_stops


@torch.no_grad()
def speed_check(model, prompts, per_domain=2):
    domains = sorted({p["domain"] for p in prompts})
    sub = [p for d in domains for p in [q for q in prompts if q["domain"] == d][:per_domain]]
    v_tok = v_sec = e_tok = e_sec = 0
    for p in sub:
        ids = p["ids"].cuda()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o = model.naivegenerate(ids.clone(), temperature=0.0, max_new_tokens=PROTOCOL["max_new_tokens"], **GEN)
        torch.cuda.synchronize(); v_sec += time.perf_counter() - t0
        v_tok += o.shape[1] - ids.shape[1]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o = model.eagenerate(ids.clone(), temperature=0.0, max_new_tokens=PROTOCOL["max_new_tokens"], **GEN)
        torch.cuda.synchronize(); e_sec += time.perf_counter() - t0
        e_tok += o.shape[1] - ids.shape[1]
    return {"prompts": len(sub), "vanilla_tok_s": v_tok / v_sec, "eagle_tok_s": e_tok / e_sec,
            "speedup": (e_tok / e_sec) / (v_tok / v_sec)}


def summarize(rows):
    c = sum(x["cycles"] for x in rows)
    stops = sum(x["stops"] for x in rows)
    new = sum(x["new_tokens"] for x in rows)
    return {"n": len(rows), "tau": (sum(x["accepted_sum"] for x in rows) + c) / max(c, 1),
            "oov_share": float(np.average([x["oov_share"] for x in rows], weights=[max(x["new_tokens"], 1) for x in rows])),
            "oov_stop_share": sum(x["oov_stop_share"] * x["stops"] for x in rows) / max(stops, 1),
            "mean_new_tokens": new / max(len(rows), 1),
            "hit_limit": float(np.mean([x["new_tokens"] >= PROTOCOL["max_new_tokens"] for x in rows]))}


def run(target, draft, tag=""):
    out = RESULTS / f"{target}__{draft}{tag}.json"
    if out.exists():
        log(f"skip {out.name} (done)")
        return
    t0 = time.time()
    tpath = target_dir(target)
    dpath = fetch(DRAFTS[draft], MODELS / f"draft_{draft}")
    model = load(tpath, dpath)
    # EaModel loads the slow tokenizer, which Llama-3 checkpoints often cannot build: use the target's fast one
    from transformers import AutoTokenizer
    tok = model.tokenizer = AutoTokenizer.from_pretrained(str(tpath))
    in_draft = draft_vocab_mask(model).cpu()
    prompts = prompts_for(tok)
    log(f"{target} + draft {draft}: {len(prompts)} prompts, max prompt {max(p['ids'].shape[1] for p in prompts)} tokens")
    rows, outs = measure_acceptance(model, prompts, PROTOCOL["max_new_tokens"], gen_kwargs=GEN)
    per = []
    for p, r in zip(prompts, rows):
        ans = outs[p["id"]]
        ids = ans.tolist()
        stops, oov_stops = stop_reasons(r["accepted"], ids, in_draft)
        per.append({**r, "domain": p["domain"], "oov_share": float((~in_draft[ans]).float().mean()) if ids else 0.0,
                    "oov_stop_share": oov_stops / max(stops, 1), "stops": stops,
                    "prompt_tokens": int(p["ids"].shape[1]), "answer": tok.decode(ans, skip_special_tokens=True)})
    result = {"target": target, "target_repo": TARGETS[target], "draft": draft, "draft_repo": DRAFTS[draft],
              "protocol": PROTOCOL, "gpu": torch.cuda.get_device_name(0),
              "all": summarize(per), "domains": {d: summarize([x for x in per if x["domain"] == d])
                                                 for d in sorted({x["domain"] for x in per})}}
    result["speed"] = speed_check(model, prompts)
    result["seconds"] = time.time() - t0
    result["rows"] = per
    RESULTS.mkdir(exist_ok=True)
    save_json(result, out)
    log(f"{target} + {draft}: tau {result['all']['tau']:.3f} | " +
        " ".join(f"{d} {v['tau']:.2f}" for d, v in result["domains"].items()) +
        f" | speedup {result['speed']['speedup']:.2f} | {result['seconds'] / 60:.1f} min")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="target:draft pairs; default config.RUNS")
    ap.add_argument("--keep-models", action="store_true")
    args = ap.parse_args()
    runs = [tuple(r.split(":")) for r in args.runs] if args.runs else RUNS
    base_cache = {}
    for i, (target, draft) in enumerate(runs):
        try:
            run(target, draft)
            if draft == "inst" and target != "base" and not (RESULTS / f"decomp_{target}.json").exists():
                from decompose import decompose
                target_dir("base")              # decomposition needs the original target too
                fetch(DRAFTS["inst"], MODELS / "draft_inst")
                target_dir(target)
                decompose(target, base_cache)
        except Exception as error:      # noqa: BLE001 - one broken model must not stop the map
            log(f"FAILED {target} + {draft}: {error!r}")
            import traceback
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
        later = {t for t, _ in runs[i + 1:]} | {"base"}
        if not args.keep_models and target not in later:
            for p in (MODELS / target, MODELS / f"{target}_adapter"):
                shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
