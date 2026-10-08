"""Texts for training and evaluation. Every response is the target's own greedy output, as in EAGLE-3.

    python prepare_data.py train <pair>     -> data/<pair>/train.pt   (6000 UltraChat + 2000 GSM8K-train prompts)
    python prepare_data.py eval  <pair>     -> data/<pair>/eval.pt    (MT-bench, GSM8K, HumanEval from the EAGLE repo)

Each file is a list of {"ids": prompt+response token ids, "mask": 1 on response tokens, ...}.
Prompts: one user turn, no system prompt (EAGLE eval protocol), Qwen3 in non-thinking mode.
Responses: greedy, at most 512 new tokens.

`eval --eagle-outputs FILE` reuses responses produced by EAGLE generation instead of generating them again
(that is what the server run used: results/<pair>/baseline_outputs.pt from the layer study). At temperature 0
EAGLE is lossless, so both give the target's greedy text, up to rare bf16 ties.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PAIRS, DATA, EAGLE_REPO

MAX_NEW_TOKENS = 512
EVAL_BENCHES = ["mt_bench", "gsm8k", "humaneval"]


def chat_prompt(tokenizer, text):
    extra = {"enable_thinking": False} if "enable_thinking" in (tokenizer.chat_template or "") else {}
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, **extra)


def load_target(pair):
    tok = AutoTokenizer.from_pretrained(PAIRS[pair]["target"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(PAIRS[pair]["target"], torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", attn_implementation="sdpa").eval()
    stop = [tok.eos_token_id]
    if PAIRS[pair].get("llama3"):
        stop.append(tok.convert_tokens_to_ids("<|eot_id|>"))
    stop = [t for t in stop if t is not None and t != tok.unk_token_id]
    return tok, model, stop


@torch.no_grad()
def generate(tok, model, stop, texts, batch=96):
    """Greedy responses for a list of prompt strings. Returns [(prompt_ids, response_ids)] in input order."""
    order = np.argsort([len(t) for t in texts])        # similar lengths in one batch
    out = [None] * len(texts)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, eos_token_id=stop,
                             pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            prompt = enc.input_ids[j][enc.attention_mask[j].bool()].cpu()
            response = gen[j, enc.input_ids.shape[1]:].cpu()
            ends = [n for n, t in enumerate(response.tolist()) if t in stop]
            out[i] = (prompt, response[: ends[0] + 1] if ends else response)
        print(f"generated {b + len(idx)}/{len(texts)}", flush=True)
    return out


def with_mask(prompt, response, **info):
    ids = torch.cat([prompt, response])
    mask = torch.zeros_like(ids)
    mask[len(prompt):] = 1
    return {"ids": ids, "mask": mask, **info}


def make_train(pair, n_chat=6000, n_math=2000):
    from datasets import load_dataset
    prompts = []
    for item in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True):
        m = item["messages"]
        if m and m[0]["role"] == "user" and len(m[0]["content"]) < 4000:
            prompts.append(m[0]["content"])
        if len(prompts) >= n_chat:
            break
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    prompts += [x["question"] for x in gsm.select(range(n_math))]

    tok, model, stop = load_target(pair)
    pairs = generate(tok, model, stop, [chat_prompt(tok, p) for p in prompts])
    seqs = [with_mask(p, r, source="gsm8k" if i >= n_chat else "ultrachat") for i, (p, r) in enumerate(pairs)]
    seqs = [seqs[i] for i in np.random.default_rng(0).permutation(len(seqs))]
    return seqs


def eval_questions(tok):
    """First turns of the benchmark questions shipped with EAGLE, in file order."""
    rows = []
    for bench in EVAL_BENCHES:
        lines = (EAGLE_REPO / f"eagle/data/{bench}/question.jsonl").read_text(encoding="utf-8").splitlines()
        for line in lines:
            item = json.loads(line)
            rows.append({"id": f"{bench}/{item['question_id']}", "bench": bench,
                         "text": chat_prompt(tok, item["turns"][0])})
    return rows


def make_eval(pair, eagle_outputs=None):
    if eagle_outputs:
        tok = AutoTokenizer.from_pretrained(PAIRS[pair]["target"])
        questions = eval_questions(tok)
        responses = torch.load(eagle_outputs)
        pairs = [(tok(q["text"], return_tensors="pt", add_special_tokens=False).input_ids[0], responses[q["id"]])
                 for q in questions]
    else:
        tok, model, stop = load_target(pair)
        questions = eval_questions(tok)
        pairs = generate(tok, model, stop, [q["text"] for q in questions], batch=16)
    return [with_mask(p, r, id=q["id"], bench=q["bench"]) for q, (p, r) in zip(questions, pairs)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "eval"])
    ap.add_argument("pair", choices=list(PAIRS))
    ap.add_argument("--eagle-outputs", default=None)
    a = ap.parse_args()
    out = DATA / a.pair
    out.mkdir(parents=True, exist_ok=True)
    seqs = make_train(a.pair) if a.what == "train" else make_eval(a.pair, a.eagle_outputs)
    torch.save(seqs, out / f"{a.what}.pt")
    lens = [int(s["mask"].sum()) for s in seqs]
    print(f"saved {len(seqs)} sequences to {out / (a.what + '.pt')}; mean response {np.mean(lens):.0f} tokens, "
          f"{np.mean([n >= MAX_NEW_TOKENS for n in lens]):.0%} hit the {MAX_NEW_TOKENS}-token limit")
