"""Stage 1: evaluate a zoo of public EAGLE-3 drafts for one target on identical target-generated texts.

python zoo_eval.py gen <mode>            # mode: nothink | think  -> texts_<mode>.pt (greedy target outputs)
python zoo_eval.py eval                  # chain acceptance of every draft on every mode
python zoo_eval.py check                 # numerical check of the evaluator against EAGLE's own draft code
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import zoo  # noqa: E402

TARGET = ROOT / "target/qwen3-8b"
OUT = ROOT / "results/zoo"
DEPTH = 6
DRAFTS = {
    "angelslim": {}, "tengyunw": {}, "thoughtworks": {}, "redhat": {}, "redhat_thinking": {},
    "deepseek_ttt7": {}, "deepseek_ttt7_plus1": {"dir": "deepseek_ttt7", "shift": 1},
    "io_e3_qwen3arch": {}, "io_e31_qwen3arch": {}, "io_e31_llamaarch": {}, "io_e31_qwen3arch_3e4": {},
    "io_e31_fcnorm": {},
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def prompts(tok, mode):
    rows = []
    for bench in ("mt_bench", "gsm8k", "humaneval"):
        for line in (ROOT / f"EAGLE/eagle/data/{bench}/question.jsonl").read_text().splitlines():
            q = json.loads(line)
            text = tok.apply_chat_template([{"role": "user", "content": q["turns"][0]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=(mode == "think"))
            rows.append({"id": f"{bench}/{q['question_id']}", "bench": bench, "text": text})
    return rows


def gen(mode, max_new=512, batch=48):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TARGET)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                 attn_implementation="sdpa").eval()
    rows = prompts(tok, mode)
    order = np.argsort([len(r["text"]) for r in rows])
    seqs = [None] * len(rows)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([rows[i]["text"] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            p = enc.input_ids[j][enc.attention_mask[j].bool()].cpu()
            r = g[j, enc.input_ids.shape[1]:].cpu()
            ends = (r == tok.eos_token_id) | (r == tok.pad_token_id)
            if ends.any():
                r = r[: int(ends.nonzero()[0]) + 1]
            ids = torch.cat([p, r])
            mask = torch.zeros_like(ids)
            mask[len(p):] = 1
            seqs[i] = {**{k: rows[i][k] for k in ("id", "bench")}, "ids": ids, "mask": mask}
        log(f"gen {mode} {b + len(idx)}/{len(rows)}")
    torch.save(seqs, OUT / f"texts_{mode}.pt")


class Target:
    def __init__(self):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                          attn_implementation="sdpa").eval()
        self.N = self.model.config.num_hidden_layers
        self._pre = None
        self.model.model.norm.register_forward_pre_hook(lambda m, a: setattr(self, "_pre", a[0]))

    @torch.no_grad()
    def feats(self, ids):
        out = self.model(input_ids=ids[None].cuda(), output_hidden_states=True)
        hs = list(out.hidden_states[: self.N]) + [self._pre]
        return torch.stack([h[0] for h in hs], 1), out.logits[0].argmax(-1)


def load_drafts(N, embed):
    drafts = {}
    for name, spec in DRAFTS.items():
        d = ROOT / "zoo" / spec.get("dir", name)
        if not (d / "config.json").exists():
            log(f"skip {name}: not downloaded")
            continue
        cfg = json.loads((d / "config.json").read_text())
        ids = cfg.get("target_layer_ids")
        layer_ids = [i + spec["shift"] for i in ids] if spec.get("shift") else None
        try:
            drafts[name] = zoo.Draft(d, N, target_embed=embed, layer_ids=layer_ids, name=name)
            dr = drafts[name]
            log(f"{name}: layers={dr.layer_ids} vocab={dr.draft_vocab} nbr={dr.norm_before_residual} "
                f"nbf={dr.norm_before_fc} fc_norm={dr.fc_norms is not None} norm_out={dr.norm_output} qk_norm={dr.W['q_norm'] is not None} theta={dr.theta}")
        except Exception as e:  # noqa: BLE001
            log(f"FAIL {name}: {e!r}")
    return drafts


def evaluate():
    tgt = Target()
    drafts = load_drafts(tgt.N, tgt.model.model.embed_tokens.weight)
    result = {"depth": DEPTH, "drafts": {n: {"layer_ids": d.layer_ids, "draft_vocab": d.draft_vocab,
                                             "params": d.n_params()} for n, d in drafts.items()}, "modes": {}}
    for mode in ("nothink", "think"):
        path = OUT / f"texts_{mode}.pt"
        if not path.exists():
            continue
        seqs = torch.load(path)
        acc = {n: {} for n in drafts}
        agree = 0
        total = 0
        for k, s in enumerate(seqs):
            ids, mask = s["ids"].cuda(), s["mask"].cuda()
            feats, greedy = tgt.feats(ids)
            # The text is the target's own greedy output, so its tokens are the target's predictions.
            agree += int((greedy[:-1][mask[1:].bool()] == ids[1:][mask[1:].bool()]).sum())
            total += int(mask[1:].sum())
            for n, d in drafts.items():
                pred = d.chain(feats, ids, DEPTH)
                lead, correct = zoo.chain_stats(pred, greedy, mask, DEPTH)
                oov = (~d.in_vocab[greedy[1:-1]])[mask[1:-1].bool()].float()
                a = acc[n].setdefault(s["bench"], {"lead": [], "correct": [], "oov": []})
                a["lead"].append(lead.cpu())
                a["correct"].append(correct.float().sum(1).cpu())
                a["oov"].append(oov.mean().item() if oov.numel() else 0.0)
            if k % 40 == 0:
                log(f"{mode} {k}/{len(seqs)}")
        res = {"teacher_forced_greedy_agreement": agree / max(total, 1)}
        for n in drafts:
            res[n] = {}
            for bench, a in acc[n].items():
                lead = torch.cat(a["lead"]).float()
                res[n][bench] = {"tau_chain": 1 + float(lead.mean()), "positions": int(lead.numel()),
                                 "depth1_acc": float((lead >= 1).float().mean()),
                                 "accept_by_depth": [float((lead >= s).float().mean()) for s in range(1, DEPTH + 1)],
                                 "oov_rate": float(np.mean(a["oov"]))}
            allp = torch.cat([torch.cat(a["lead"]) for a in acc[n].values()]).float()
            res[n]["all"] = {"tau_chain": 1 + float(allp.mean()), "positions": int(allp.numel())}
            log(f"{mode} {n}: tau_chain={res[n]['all']['tau_chain']:.3f} " +
                " ".join(f"{b}={v['tau_chain']:.2f}" for b, v in res[n].items() if b != "all"))
        result["modes"][mode] = res
        (OUT / "zoo_eval.json").write_text(json.dumps(result, indent=2))


def check():
    """Depth-1 logits of zoo.Draft must match EAGLE's own cnets.Model on an AngelSlim draft (Qwen3-1.7B pair)."""
    import run
    import layer_lab as lab
    model = run.load("qwen3-1.7b")
    ea = model.ea_layer
    N = len(model.base_model.model.layers)
    lab.set_capture(model, tuple(range(N + 1)))
    tok = model.get_tokenizer()
    ids = tok("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").input_ids[0].cuda()
    states, _ = lab.target_forward(model, ids)
    d = zoo.Draft(ROOT / "models/qwen3-1.7b/draft", N, target_embed=ea.embed_tokens.weight, name="angelslim_1.7b")
    pred = d.chain(states, ids, 3)
    tri = lab.default_triplet(N)
    g = states[:, list(tri)].flatten(1)[None].to(ea.fc.weight.dtype)
    nxt = torch.zeros_like(ids)
    nxt[:-1] = ids[1:]
    with torch.no_grad():
        h = ea.fc(g)
        out = ea.midlayer(input_emb=ea.embed_tokens(nxt[None]), hidden_states=h,
                          attention_mask=lab.causal_mask(ids.shape[0], ids.device), position_ids=torch.arange(ids.shape[0], device=ids.device)[None],
                          use_cache=False)[0]
        logits = ea.lm_head(ea.norm(out))[0]
    ref = logits.argmax(-1) + ea.d2t[logits.argmax(-1)]
    agree = (ref[:-1] == pred[0][:-1]).float().mean().item()
    print("depth-1 argmax agreement with EAGLE cnets:", agree)


def validate(n_prompts=12, depth=5):
    """EAGLE chain decoding (top_k=1) vs the offline evaluator on the same greedy texts (Qwen3-1.7B pair)."""
    import run
    import layer_lab as lab
    lab.patch_eagle(run.EAGLE)
    p = run.PAIRS["qwen3-1.7b"]
    model = lab.load_pair(str(p["base"]), str(p["draft"]), total_token=depth + 1, depth=depth, top_k=1,
                          dtype=torch.bfloat16)
    tok = model.get_tokenizer()
    N = len(model.base_model.model.layers)
    prompts = lab.eval_prompts(run.EAGLE, tok, [["mt_bench", n_prompts // 2], ["gsm8k", n_prompts // 2]])
    rows, outs = lab.run_tau(model, prompts, 256, gen_kwargs={"max_length": 2048})
    eagle_cycles = [a + 1 for r in rows for a in r["accepted"]]
    d = zoo.Draft(p["draft"], N, target_embed=model.ea_layer.embed_tokens.weight, name="angelslim_1.7b")
    lab.set_capture(model, tuple(range(N + 1)))
    ours = []
    max_acc = max(max(r["accepted"]) for r in rows)
    for pr in prompts:
        ids = torch.cat([pr["ids"][0], outs[pr["id"]]]).cuda()
        mask = torch.zeros_like(ids)
        mask[pr["ids"].shape[1]:] = 1
        states, logits = lab.target_forward(model, ids)
        greedy = logits.argmax(-1)
        pred = d.chain(states, ids, max_acc)
        lead = zoo.chain_lead_full(pred, greedy, max_acc)
        valid = torch.zeros_like(mask, dtype=torch.bool)
        valid[pr["ids"].shape[1] - 1: ids.shape[0] - max_acc - 1] = True
        ours += zoo.renewal_tau(lead, valid, max_acc)
    print(f"max accepted per cycle in EAGLE chain: {max_acc}")
    print(f"EAGLE chain tau: {np.mean(eagle_cycles):.4f} over {len(eagle_cycles)} cycles")
    print(f"offline renewal tau: {np.mean(ours):.4f} over {len(ours)} cycles")


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate()
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "gen":
        gen(sys.argv[2])
    elif cmd == "eval":
        evaluate()
    elif cmd == "check":
        check()
