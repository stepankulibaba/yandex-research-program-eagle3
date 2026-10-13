"""Texts for training and evaluation. Every answer is the target's own greedy output (as EAGLE-3 trains).

    python prepare_data.py train          -> data/train_en.pt    6000 UltraChat + 2000 GSM8K-train prompts
    python prepare_data.py train --ml     -> data/train_ml.pt    the same + multilingual prompts (oasst2: ru uk de zh ja)
    python prepare_data.py eval           -> data/eval_en.pt     MT-bench, GSM8K, HumanEval (from the EAGLE repo)
    python prepare_data.py lang           -> data/eval_lang.pt   60 m-ArenaHard + 60 MGSM questions in 6 languages

Each file is a list of {"ids": prompt+answer token ids, "mask": 1 on answer tokens, "bench", "lang", ...}.
Prompts: one user turn, no system prompt, Qwen3 in non-thinking mode. Answers: greedy, at most 512 new tokens.
The language questions are the same items in every language (same m-ArenaHard ids, same MGSM rows; MGSM has no
Ukrainian), as in our earlier language study (layer-selection/frozen/code/languages.py).
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DATA, EAGLE_REPO, LANGS, MAX_NEW_TOKENS, TARGET

EVAL_BENCHES = ["mt_bench", "gsm8k", "humaneval"]
ML_LANGS = ["ru", "uk", "de", "zh", "ja"]


def chat_prompt(tokenizer, text):
    extra = {"enable_thinking": False} if "enable_thinking" in (tokenizer.chat_template or "") else {}
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, **extra)


def load_target():
    tok = AutoTokenizer.from_pretrained(TARGET)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                 attn_implementation="sdpa").eval()
    return tok, model, [tok.eos_token_id]


@torch.no_grad()
def generate(tok, model, stop, texts, batch=96):
    """Greedy answers for a list of chat-formatted prompts. Returns [(prompt_ids, answer_ids)] in input order."""
    order = np.argsort([len(t) for t in texts])        # similar lengths in one batch
    out = [None] * len(texts)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, eos_token_id=stop,
                             pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            prompt = enc.input_ids[j][enc.attention_mask[j].bool()].cpu()
            answer = gen[j, enc.input_ids.shape[1]:].cpu()
            ends = [n for n, t in enumerate(answer.tolist()) if t in stop]
            out[i] = (prompt, answer[: ends[0] + 1] if ends else answer)
        print(f"generated {b + len(idx)}/{len(texts)}", flush=True)
    return out


def with_mask(prompt, answer, **info):
    ids = torch.cat([prompt, answer])
    mask = torch.zeros_like(ids)
    mask[len(prompt):] = 1
    return {"ids": ids, "mask": mask, **info}


def train_prompts(ml, n_chat=6000, n_math=2000, n_per_lang=800):
    from datasets import load_dataset
    rows = []
    for item in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True):
        m = item["messages"]
        if m and m[0]["role"] == "user" and len(m[0]["content"]) < 4000:
            rows.append({"text": m[0]["content"], "source": "ultrachat", "lang": "en"})
        if len(rows) >= n_chat:
            break
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    rows += [{"text": x["question"], "source": "gsm8k", "lang": "en"} for x in gsm.select(range(n_math))]
    if ml:
        # first user turns of OpenAssistant conversations (Apache-2.0); "uk-UA" etc. -> "uk"
        counts = {lang: 0 for lang in ML_LANGS}
        for split in ("train", "validation"):
            for x in load_dataset("OpenAssistant/oasst2", split=split):
                lang = (x["lang"] or "").split("-")[0]
                if x["parent_id"] is None and x["role"] == "prompter" and lang in counts \
                        and counts[lang] < n_per_lang and len(x["text"]) < 4000:
                    rows.append({"text": x["text"], "source": "oasst2", "lang": lang})
                    counts[lang] += 1
        print("multilingual prompts:", counts, flush=True)
    return rows


def eval_questions():
    rows = []
    for bench in EVAL_BENCHES:
        for line in (EAGLE_REPO / f"eagle/data/{bench}/question.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            rows.append({"id": f"{bench}/{item['question_id']}", "bench": bench, "lang": "en",
                         "text": item["turns"][0]})
    return rows


def lang_questions(n_arena=60, n_mgsm=60):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def parquet(repo, lang):
        return pd.read_parquet(hf_hub_download(repo, f"{lang}/test-00000-of-00001.parquet", repo_type="dataset"))

    rows = []
    ids = sorted(parquet("CohereLabs/m-ArenaHard", "en")["question_id"])[:n_arena]
    for lang in LANGS:
        for qid, r in parquet("CohereLabs/m-ArenaHard", lang).set_index("question_id").loc[ids].iterrows():
            rows.append({"id": f"{lang}/arena/{qid}", "bench": "arena", "lang": lang, "text": r["prompt"]})
        if lang != "uk":
            mgsm = parquet("juletxara/mgsm", lang)
            for i in range(n_mgsm):
                rows.append({"id": f"{lang}/mgsm/{i}", "bench": "mgsm", "lang": lang,
                             "text": mgsm.iloc[i]["question"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "eval", "lang"])
    ap.add_argument("--ml", action="store_true", help="train: add multilingual prompts (oasst2)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: at most this many texts (per language for lang)")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if a.what == "train":
        rows, name, batch = train_prompts(a.ml), "train_ml.pt" if a.ml else "train_en.pt", 96
    else:
        rows = eval_questions() if a.what == "eval" else lang_questions()
        name, batch = ("eval_en.pt" if a.what == "eval" else "eval_lang.pt"), 32
    if a.limit:
        if a.what == "lang":
            rows = [r for lang in LANGS for r in [x for x in rows if x["lang"] == lang][:a.limit]]
        else:
            rows = rows[:: max(1, len(rows) // a.limit)][:a.limit]
    tok, model, stop = load_target()
    pairs = generate(tok, model, stop, [chat_prompt(tok, r["text"]) for r in rows], batch=batch)
    seqs = [with_mask(p, ans, **{k: v for k, v in r.items() if k != "text"}, prompt=r["text"])
            for r, (p, ans) in zip(rows, pairs)]
    if a.what == "train":
        seqs = [seqs[i] for i in np.random.default_rng(0).permutation(len(seqs))]
    for s in seqs:                                       # keep the answer text for reading (not used in training)
        s["answer"] = tok.decode(s["ids"][s["mask"].bool()], skip_special_tokens=True)
    torch.save(seqs, DATA / name)
    lens = [int(s["mask"].sum()) for s in seqs]
    print(f"saved {len(seqs)} sequences to {DATA / name}; mean answer {np.mean(lens):.0f} tokens, "
          f"{np.mean([n >= MAX_NEW_TOKENS for n in lens]):.0%} hit the {MAX_NEW_TOKENS}-token limit")


if __name__ == "__main__":
    main()
