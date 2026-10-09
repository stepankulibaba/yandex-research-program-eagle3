"""Training texts: the target's own greedy answers to 6000 UltraChat and 2000 GSM8K-train prompts (as in EAGLE-3).

    python gen_data.py dsl-8b        -> results/h200/dsl-8b/train_data.pt
"""
import sys
import time

import numpy as np
import torch

from config import PAIRS, log, results_dir
from eagle_model import chat_prompt


def main(pair, n_chat=6000, n_math=2000, max_new=512, batch=96):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PAIRS[pair]["target"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(PAIRS[pair]["target"], torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", attn_implementation="sdpa").eval()
    prompts = []
    for item in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True):
        m = item["messages"]
        if m and m[0]["role"] == "user" and len(m[0]["content"]) < 4000:
            prompts.append(m[0]["content"])
        if len(prompts) >= n_chat:
            break
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    prompts += [x["question"] for x in gsm.select(range(n_math))]
    texts = [chat_prompt(tok, t) for t in prompts]
    stop = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|eot_id|>")] if PAIRS[pair]["llama3"] else [])

    seqs = [None] * len(texts)
    order = np.argsort([len(t) for t in texts])          # similar lengths in one batch
    started = time.time()
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=stop,
                                 pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            prompt_ids = enc.input_ids[j][enc.attention_mask[j].bool()].cpu()
            answer = gen[j, enc.input_ids.shape[1]:].cpu()
            ends = [k for k, t in enumerate(answer.tolist()) if t in stop]
            answer = answer[: ends[0] + 1] if ends else answer
            ids = torch.cat([prompt_ids, answer])
            mask = torch.zeros_like(ids)
            mask[len(prompt_ids):] = 1
            seqs[i] = {"ids": ids, "mask": mask, "source": "gsm8k" if i >= n_chat else "ultrachat"}
        if (b // batch) % 10 == 0:
            log(f"gen {b + len(idx)}/{len(order)} ({time.time() - started:.0f}s)")

    perm = np.random.default_rng(0).permutation(len(seqs))
    torch.save([seqs[i] for i in perm], results_dir(pair) / "train_data.pt")
    lens = [s["mask"].sum().item() for s in seqs]
    log(f"saved {len(seqs)} sequences, mean response {np.mean(lens):.0f} tokens, "
        f"truncated {np.mean([n >= max_new for n in lens]):.2%}")


if __name__ == "__main__":
    main(sys.argv[1])
