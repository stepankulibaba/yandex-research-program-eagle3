"""Overnight layer study on the H200: paper-protocol evaluation, target-generated data, analysis, fusion training.

Usage: python run.py <stage> <pair>
  stages: eval_official | gen_data | analysis | fusion
  pairs:  dsl-8b | qwen3-1.7b
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent.parent   # frozen/: models/, EAGLE/, results/ (this file lives in code/original/)
sys.path.insert(0, str(CODE))
import layer_lab as lab  # noqa: E402

EAGLE = ROOT / "EAGLE"
PAIRS = {
    "dsl-8b": {"base": ROOT / "models/dsl-8b/base", "draft": ROOT / "models/dsl-8b/draft",
               "gen": {"is_llama3": True}},
    "qwen3-1.7b": {"base": ROOT / "models/qwen3-1.7b/base", "draft": ROOT / "models/qwen3-1.7b/draft", "gen": {}},
}
# EAGLE-3 paper / official DeepSeek eval script: first turn, no system prompt, temperature 0,
# total_token 60, depth 5, top_k 10; eagenerate defaults max_new_tokens=512, max_length=2048.
PROTOCOL = {"total_token": 60, "depth": 5, "top_k": 10, "max_new_tokens": 512, "max_length": 2048}
BENCHES = [["mt_bench", None], ["gsm8k", None]]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def out_dir(pair):
    d = ROOT / "results/h200" / pair
    d.mkdir(parents=True, exist_ok=True)
    return d


def gen_kwargs(pair):
    return dict(PAIRS[pair]["gen"], max_length=PROTOCOL["max_length"])


def load(pair):
    lab.patch_eagle(EAGLE)
    lab.seed_all(0)
    p = PAIRS[pair]
    return lab.load_pair(str(p["base"]), str(p["draft"]), PROTOCOL["total_token"], PROTOCOL["depth"],
                         PROTOCOL["top_k"], dtype=torch.bfloat16)


def save_rows(path, rows):
    lab.dump([{k: v for k, v in r.items()} for r in rows], path)


# --------------------------------------------------------------------------

def stage_eval_official(pair):
    """Official draft, paper protocol: baseline + inference ablations of the three captured layers."""
    out = out_dir(pair)
    model = load(pair)
    tok = model.get_tokenizer()
    N = len(model.base_model.model.layers)
    tri = lab.default_triplet(N)
    W = model.ea_layer.fc.weight.detach().clone()
    fc_orig = model.ea_layer.fc
    H = W.shape[0]
    prompts = lab.eval_prompts(EAGLE, tok, BENCHES + [["humaneval", None]])
    log(f"{pair}: N={N} H={H} triplet={tri} prompts={len(prompts)}")
    rows, base_out = lab.run_tau(model, prompts, PROTOCOL["max_new_tokens"], gen_kwargs=gen_kwargs(pair))
    torch.save({k: v for k, v in base_out.items()}, out / "baseline_outputs.pt")
    save_rows(out / "rows_baseline.json", rows)
    result = {"protocol": PROTOCOL, "n_layers": N, "hidden": H, "triplet": tri,
              "baseline": lab.tau_summary(rows)}
    log(f"baseline {result['baseline']}")
    base_rows = rows
    rep_rows, _ = lab.run_tau(model, prompts[:20], PROTOCOL["max_new_tokens"], reference=base_out,
                              gen_kwargs=gen_kwargs(pair))
    result["determinism_20"] = lab.tau_summary(rep_rows)

    seqs = lab.ultrachat_sequences(tok, 300, max_len=1024)
    stats = lab.collect_positions(model, seqs, 30000)["states"].float()
    means = stats.mean(0)
    rms = stats.pow(2).mean(-1).sqrt().mean(0)
    del stats
    q = round(0.7 * N)
    ablate = [p for p in prompts if p["bench"] in ("mt_bench", "gsm8k")]
    base_sub = [r for r in base_rows if r["bench"] in ("mt_bench", "gsm8k")]

    class Edit(torch.nn.Module):
        def __init__(self, keep=(0, 1, 2), scale=None, rank=None):
            super().__init__()
            Wf = W.float()
            if rank:
                U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
                Wf = (U[:, :rank] * S[:rank]) @ Vh[:rank]
            self.register_buffer("W", Wf.cuda())
            self.register_buffer("fill", torch.cat([means[t] for t in tri]).cuda())
            self.keep, self.scale = keep, scale or {}

        def forward(self, x):
            dtype = x.dtype
            x = x.float().clone()
            for s in range(3):
                sl = slice(s * H, (s + 1) * H)
                if s not in self.keep:
                    x[..., sl] = self.fill[sl]
                if s in self.scale:
                    x[..., sl] = x[..., sl] * self.scale[s]
            return (x @ self.W.T).to(dtype)

    def sc(old, new):
        return float(rms[old] / rms[new])

    conditions = [
        ("only_high", None, Edit(keep=(2,))),
        ("drop_low", None, Edit(keep=(1, 2))),
        ("drop_mid", None, Edit(keep=(0, 2))),
        ("only_low_mid", None, Edit(keep=(0, 1))),
        (f"mid_{tri[1]}to{q}", (tri[0], q, tri[2]), Edit(scale={1: sc(tri[1], q)})),
        (f"high_{tri[2]}to{N}", (tri[0], tri[1], N), Edit(scale={2: sc(tri[2], N)})),
        (f"high_{tri[2]}to{N - 1}", (tri[0], tri[1], N - 1), Edit(scale={2: sc(tri[2], N - 1)})),
        (f"fc_rank{H // 4}", None, Edit(rank=H // 4)),
        (f"fc_rank{H // 2}", None, Edit(rank=H // 2)),
    ]
    result["ablations"] = {}
    for name, capture, module in conditions:
        model.ea_layer.fc = module
        lab.set_capture(model, capture)
        try:
            rows, _ = lab.run_tau(model, ablate, PROTOCOL["max_new_tokens"], reference=base_out,
                                  gen_kwargs=gen_kwargs(pair))
        finally:
            model.ea_layer.fc = fc_orig
            lab.set_capture(model, None)
        summ = lab.tau_summary(rows)
        summ["delta_ci95"] = lab.paired_bootstrap(base_sub, rows)
        summ["prefix_pair"] = lab.prefix_tau_pair(base_sub, rows)
        summ["capture"] = list(capture) if capture else list(tri)
        result["ablations"][name] = summ
        save_rows(out / f"rows_{name}.json", rows)
        log(f"{name}: tau={summ['tau']:.3f} ci={summ['delta_ci95']} prefix={summ['prefix_pair']}")
        lab.dump(result, out / "eval_official.json")
    lab.dump(result, out / "eval_official.json")


# --------------------------------------------------------------------------

def stage_gen_data(pair, n_chat=6000, n_math=2000, max_new=512, batch=96):
    """Target-generated responses (greedy) to UltraChat and GSM8K-train prompts, as the authors did."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = out_dir(pair)
    p = PAIRS[pair]
    tok = AutoTokenizer.from_pretrained(p["base"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(p["base"], torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                 attn_implementation="sdpa").eval()
    prompts = []
    for item in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True):
        m = item["messages"]
        if m and m[0]["role"] == "user" and len(m[0]["content"]) < 4000:
            prompts.append(m[0]["content"])
        if len(prompts) >= n_chat:
            break
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    prompts += [x["question"] for x in gsm.select(range(n_math))]
    texts = [lab.chat_prompt(tok, t) for t in prompts]
    order = np.argsort([len(t) for t in texts])
    stop = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|eot_id|>")] if p["gen"].get("is_llama3") else [])
    seqs = [None] * len(texts)
    started = time.time()
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=stop,
                                 pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            prompt_ids = enc.input_ids[j][enc.attention_mask[j].bool()].cpu()
            resp = gen[j, enc.input_ids.shape[1]:].cpu()
            ends = [k for k, t in enumerate(resp.tolist()) if t in stop]
            resp = resp[: ends[0] + 1] if ends else resp
            ids = torch.cat([prompt_ids, resp])
            mask = torch.zeros_like(ids)
            mask[len(prompt_ids):] = 1
            seqs[i] = {"ids": ids, "mask": mask, "source": "gsm8k" if i >= n_chat else "ultrachat"}
        if (b // batch) % 10 == 0:
            log(f"gen {b + len(idx)}/{len(order)} ({time.time() - started:.0f}s)")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(seqs))
    torch.save([seqs[i] for i in perm], out / "train_data.pt")
    lens = [s["mask"].sum().item() for s in seqs]
    log(f"saved {len(seqs)} sequences, mean response {np.mean(lens):.0f} tokens, "
        f"truncated {np.mean([l >= max_new for l in lens]):.2%}")


# --------------------------------------------------------------------------

def stage_analysis(pair):
    """Layer analysis on target-generated text: norms, fc decomposition, CKA, ridge R^2, future-token probes."""
    out = out_dir(pair)
    model = load(pair)
    tok = model.get_tokenizer()
    N = len(model.base_model.model.layers)
    tri = lab.default_triplet(N)
    W = model.ea_layer.fc.weight.detach().clone()
    data = torch.load(out / "train_data.pt")
    train_seqs = [s for s in data[-1500:]]
    base_out = torch.load(out / "baseline_outputs.pt")
    prompts = lab.eval_prompts(EAGLE, tok, BENCHES + [["humaneval", None]])
    test_seqs = []
    for p in prompts:
        gen = base_out[p["id"]]
        test_seqs.append({"ids": torch.cat([p["ids"][0], gen]),
                          "mask": torch.cat([torch.zeros(p["ids"].shape[1], dtype=torch.long),
                                             torch.ones(gen.numel(), dtype=torch.long)])})
    train = lab.collect_positions(model, train_seqs, 60000)
    test = lab.collect_positions(model, test_seqs, 30000)
    log(f"positions train={tuple(train['states'].shape)} test={tuple(test['states'].shape)}")
    res = {"n_layers": N, "triplet": tri, "norms": lab.layer_norm_stats(train["states"])}
    res["fc_decomposition"] = lab.fc_decomposition(W, test["states"], tri)
    log(f"fc shares {res['fc_decomposition']['contribution_energy_share']}")
    res["cka"] = lab.linear_cka(train["states"], n=4096)
    g_tr = lab.official_fused(W, train["states"], tri)
    g_te = lab.official_fused(W, test["states"], tri)
    r2 = {str(l): lab.ridge_r2(train["states"][:, l].float(), g_tr, test["states"][:, l].float(), g_te)
          for l in range(N + 1)}
    r2["triplet"] = lab.ridge_r2(train["states"][:, list(tri)].float().flatten(1), g_tr,
                                 test["states"][:, list(tri)].float().flatten(1), g_te)
    res["ridge_r2_to_fused"] = r2
    log(f"ridge r2 single max at {max(range(N + 1), key=lambda l: r2[str(l)])}, triplet {r2['triplet']:.3f}")
    del g_tr, g_te
    t2d = model.ea_layer.t2d.bool().cpu() if hasattr(model.ea_layer, "t2d") else torch.ones(
        model.base_model.lm_head.weight.shape[0], dtype=torch.bool)
    draft_index = torch.cumsum(t2d.long(), 0) - 1

    def to_draft(y):
        return torch.where(t2d[y], draft_index[y], torch.full_like(y, -100))

    head_w = model.base_model.lm_head.weight[t2d.cuda()].float()
    norm = model.base_model.model.norm
    probes = {}
    for layer in range(N + 1):
        for k in (1, 2, 3):
            per_epoch, lens = lab.train_future_probe(
                train["states"][:, layer], to_draft(train["labels"][:, k - 1]),
                test["states"][:, layer], to_draft(test["labels"][:, k - 1]), head_w, norm, epochs=3)
            probes[f"{layer}:{k}"] = {"probe_acc": per_epoch[-1], "probe_acc_per_epoch": per_epoch, "lens_acc": lens}
        log(f"probe {layer}: " + " ".join(f"k{k}={probes[f'{layer}:{k}']['probe_acc']:.3f}" for k in (1, 2, 3)))
    res["future_probes"] = probes
    lab.dump(res, out / "analysis.json")


# --------------------------------------------------------------------------

def stage_fusion(pair, n_train=8000):
    import phase_b
    out = out_dir(pair) / "fusion"
    model_cfg = json.loads((PAIRS[pair]["base"] / "config.json").read_text())
    N = model_cfg["num_hidden_layers"]
    q = round(0.7 * N)
    tri = lab.default_triplet(N)
    cfg = {"eagle_dir": str(EAGLE), "base": str(PAIRS[pair]["base"]), "draft": str(PAIRS[pair]["draft"]),
           "seed": 0, "total_token": PROTOCOL["total_token"], "depth": PROTOCOL["depth"], "top_k": PROTOCOL["top_k"],
           "max_new_tokens": PROTOCOL["max_new_tokens"], "gen_kwargs": gen_kwargs(pair),
           "prompt_counts": BENCHES, "max_len": 2048, "out": str(out), "amp_dtype": "bfloat16",
           "train_data": str(out_dir(pair) / "train_data.pt"), "n_train_seqs": n_train, "skip": 0, "epochs": 1,
           "accum": 8, "lr": 1e-3, "fast_lr_mult": 10.0, "lr_mult": {"fc3_official_ft": 0.1}, "warmup_frac": 0.03,
           "clip": 1.0, "aux_coef": 0.01, "balanced_moe": 0.1, "log_every": 25, "max_train_seconds": 3 * 3600,
           "repeat_official": False, "variants": None,
           "custom_variants": [[f"tri_{tri[1]}_{q}_{tri[2]}", [tri[1], q, tri[2]]],
                               [f"tri_{tri[0]}_{q}_{tri[2]}", [tri[0], q, tri[2]]],
                               [f"pair_{q}_{tri[2]}", [q, tri[2]]],
                               [f"single_{tri[2]}", [tri[2]]],
                               [f"single_{N}", [N]]]}
    phase_b.run(cfg)


def stage_sweep(pair, n_train=3000, accum=8, lr=1e-3):
    """Layer sweep with the draft frozen: for every layer l fit a linear map from {l} and from {l, N-3}
    to the draft input (step-0 EAGLE-3 objective), then measure held-out step-0 accuracy on target outputs."""
    out = out_dir(pair)
    model = load(pair)
    ea = model.ea_layer
    tok = model.get_tokenizer()
    N = len(model.base_model.model.layers)
    H = model.base_model.config.hidden_size
    tri = lab.default_triplet(N)
    W = ea.fc.weight.detach().clone()
    lab.set_capture(model, tuple(range(N + 1)))
    variants = {"official": lab.FixedFusion(H, tri, init_weight=W)}
    for l in range(N + 1):
        variants[f"single_{l}"] = lab.FixedFusion(H, [l], norm=True)
        if l != tri[2]:
            variants[f"pair_{l}_{tri[2]}"] = lab.FixedFusion(H, sorted([l, tri[2]]), norm=True)
    variants[f"triplet_upstream"] = lab.FixedFusion(H, tri, norm=True)
    data = torch.load(out / "train_data.pt")[:n_train]
    base_out = torch.load(out / "baseline_outputs.pt")
    prompts = lab.eval_prompts(EAGLE, tok, BENCHES + [["humaneval", None]])
    test = []
    for p in prompts:
        gen = base_out[p["id"]]
        test.append({"ids": torch.cat([p["ids"][0], gen]),
                     "mask": torch.cat([torch.zeros(p["ids"].shape[1], dtype=torch.long),
                                        torch.ones(gen.numel(), dtype=torch.long)])})
    total = n_train // accum
    state = {}
    for name, m in variants.items():
        m.cuda()
        if name == "official":
            continue
        opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + np.cos(np.pi * min(s / total, 1.0))))
        state[name] = (m, opt, sched)
    log(f"sweep {pair}: {len(state)} trainable variants, {n_train} sequences")
    started = time.time()
    for i, seq in enumerate(data):
        states, tp, pm, nxt = lab.target_batch(model, seq)
        if pm.sum() == 0:
            continue
        feats = states.float()
        for name, (m, opt, sched) in state.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = lab.draft_step0_logits(ea, m(feats), nxt)
            loss, _ = lab.step0_loss(logits, tp, pm)
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
        if (i + 1) % 200 == 0:
            log(f"sweep {i + 1}/{n_train} ({time.time() - started:.0f}s)")
    result = {"n_layers": N, "triplet": tri, "n_train": n_train, "train_seconds": time.time() - started,
              "step0": {}}
    import phase_b
    for name, m in variants.items():
        result["step0"][name] = phase_b.evaluate_step0(model, m.eval(), test, torch.bfloat16)
    log("sweep: " + " ".join(f"{k}={v['acc']:.3f}" for k, v in result["step0"].items() if k.startswith("single")))
    lab.dump(result, out / "sweep.json")


LANGS = ["en", "ru", "uk", "de", "zh", "ja"]
SCRIPTS = {"cyr": ("Ѐ", "ӿ"), "latin": ("a", "z"), "cjk": ("一", "鿿"), "kana": ("぀", "ヿ")}


def script_shares(text):
    counts = {k: 0 for k in SCRIPTS}
    for ch in text.lower():
        for k, (lo, hi) in SCRIPTS.items():
            if lo <= ch <= hi:
                counts[k] += 1
    total = max(sum(counts.values()), 1)
    return {k: v / total for k, v in counts.items()}


def lang_prompts(tok, lang, n_arena=60, n_mgsm=60):
    """Parallel prompts: same m-ArenaHard question ids and same MGSM rows in every language."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rows = []
    arena = pd.read_parquet(hf_hub_download("CohereLabs/m-ArenaHard", f"{lang}/test-00000-of-00001.parquet",
                                            repo_type="dataset"))
    ids = sorted(pd.read_parquet(hf_hub_download("CohereLabs/m-ArenaHard", "en/test-00000-of-00001.parquet",
                                                 repo_type="dataset"))["question_id"])[:n_arena]
    arena = arena.set_index("question_id").loc[ids]
    for qid, r in arena.iterrows():
        rows.append({"id": f"arena/{qid}", "bench": "arena", "text": r["prompt"]})
    if lang != "uk":
        mgsm = pd.read_parquet(hf_hub_download("juletxara/mgsm", f"{lang}/test-00000-of-00001.parquet",
                                               repo_type="dataset"))
        for i in range(n_mgsm):
            rows.append({"id": f"mgsm/{i}", "bench": "mgsm", "text": mgsm.iloc[i]["question"]})
    for r in rows:
        r["ids"] = tok(lab.chat_prompt(tok, r["text"]), return_tensors="pt", add_special_tokens=False).input_ids
    return rows


def stage_lang(pair, n_vanilla=8):
    """Same official draft, parallel prompts in several languages: tau, draft-vocab coverage, why chains stop."""
    out = out_dir(pair) / "lang"
    out.mkdir(exist_ok=True)
    model = load(pair)
    tok = model.get_tokenizer()
    ea = model.ea_layer
    t2d = ea.t2d.bool().cpu() if hasattr(ea, "t2d") else torch.ones(len(tok), dtype=torch.bool)
    result = {"protocol": PROTOCOL, "draft_vocab": int(t2d.sum()), "target_vocab": int(t2d.numel()), "langs": {}}
    for lang in LANGS:
        prompts = lang_prompts(tok, lang)
        rows, outs = lab.run_tau(model, prompts, PROTOCOL["max_new_tokens"], gen_kwargs=gen_kwargs(pair))
        per = []
        for p, r in zip(prompts, rows):
            gen = outs[p["id"]]
            ids = gen.tolist()
            text = tok.decode(gen, skip_special_tokens=True)
            # Each verification cycle emits accepted draft tokens plus one target token; that last token is the
            # one the draft failed to propose. Position 0 comes from the prefill.
            start, oov_stops, stops = 1, 0, 0
            for a in r["accepted"]:
                end = start + a
                if end < len(ids):
                    stops += 1
                    oov_stops += int(not t2d[ids[end]])
                start = end + 1
            per.append({**{k: v for k, v in r.items() if k != "accepted"}, "accepted": r["accepted"],
                        "oov_share": float((~t2d[gen]).float().mean()) if len(ids) else 0.0,
                        "oov_stop_share": oov_stops / max(stops, 1), "stops": stops,
                        "chars": len(text), "scripts": script_shares(text),
                        "prompt_tokens": int(p["ids"].shape[1]), "prompt_chars": len(p["text"])})
        summary = {"all": lab.tau_summary(rows)}
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
        # Wall-clock speedup on a small subset: vanilla greedy vs EAGLE, same prompts.
        sub = prompts[:n_vanilla // 2] + [p for p in prompts if p["bench"] == "mgsm"][:n_vanilla // 2]
        v_tok = v_sec = e_tok = e_sec = 0
        for p in sub:
            ids = p["ids"].cuda()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            o = model.naivegenerate(ids.clone(), temperature=0.0, max_new_tokens=PROTOCOL["max_new_tokens"],
                                    **gen_kwargs(pair))
            torch.cuda.synchronize(); v_sec += time.perf_counter() - t0
            v_tok += o.shape[1] - ids.shape[1]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            o = model.eagenerate(ids.clone(), temperature=0.0, max_new_tokens=PROTOCOL["max_new_tokens"],
                                 **gen_kwargs(pair))
            torch.cuda.synchronize(); e_sec += time.perf_counter() - t0
            e_tok += o.shape[1] - ids.shape[1]
        summary["speed_subset"] = {"prompts": len(sub), "vanilla_tok_s": v_tok / v_sec, "eagle_tok_s": e_tok / e_sec,
                                   "speedup": (e_tok / e_sec) / (v_tok / v_sec)}
        result["langs"][lang] = summary
        lab.dump(per, out / f"rows_{lang}.json")
        lab.dump(result, out / "lang.json")
        log(f"{pair} {lang}: tau={summary['all']['tau']:.3f} " + " ".join(
            f"{b}: tau={summary[b]['tau']:.2f} oov={summary[b]['oov_share']:.3f} oovstop={summary[b]['oov_stop_share']:.3f} "
            f"cyr={summary[b]['scripts']['cyr']:.2f} lat={summary[b]['scripts']['latin']:.2f} cjk={summary[b]['scripts']['cjk']:.2f}"
            for b in ("arena", "mgsm") if b in summary) + f" speedup={summary['speed_subset']['speedup']:.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "lang":
        stage_lang(sys.argv[2])
        sys.exit(0)
    stage, pair = sys.argv[1], sys.argv[2]
    {"eval_official": stage_eval_official, "gen_data": stage_gen_data, "analysis": stage_analysis,
     "fusion": stage_fusion, "sweep": stage_sweep}[stage](pair)
