"""The official EAGLE-3 implementation with one change: which target layers the draft reads is configurable.

Upstream EAGLE-3 hard-codes the captured layers (2, N/2, N-3) inside the target's forward pass. `patch_eagle`
rewrites that line so the set can be changed at run time with `set_capture(model, layers)`; with nothing set the
behaviour is exactly upstream.

Layer index convention used everywhere: index i in [0, N] is the hidden state that ENTERS decoder layer i
(0 = token embeddings), i.e. HF `hidden_states[i]`; index N is the output of the last layer before the final norm.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from config import EAGLE_DIR, PAIRS, PROTOCOL

# --- the patch: three small source edits in eagle/model/modeling_{qwen3,llama}_kv.py --------------------------------
_HELPER = '''

def _eagle_capture_layers(model):
    """Layers whose input hidden state is passed to the EAGLE-3 draft."""
    layers = getattr(model, "eagle_layers", None)
    if layers is None:
        n = len(model.layers)
        return (2, n // 2, n - 3)
    return layers
'''
_OLD_SELECT = ("            if idx==len(self.layers)-3 or idx==len(self.layers)//2 or idx==2:\n"
               "                all_hidden_states += (hidden_states,)\n")
_NEW_SELECT = ("            if idx in _eagle_capture_layers(self):\n"
               "                all_hidden_states += (hidden_states,)\n")
_OLD_TAIL = "        hidden_states = self.norm(hidden_states)\n\n        # add hidden states from the last decoder layer\n"
_NEW_TAIL = ("        if len(self.layers) in _eagle_capture_layers(self):\n"     # index N: last layer, before the norm
             "            all_hidden_states += (hidden_states,)\n" + _OLD_TAIL)


def patch_eagle(eagle_dir=None):
    """Make the captured layer set of the Qwen3 and Llama targets configurable (idempotent)."""
    eagle_dir = eagle_dir or EAGLE_DIR
    for name in ("modeling_qwen3_kv.py", "modeling_llama_kv.py"):
        path = Path(eagle_dir) / "eagle/model" / name
        src = path.read_text(encoding="utf-8")
        if "_eagle_capture_layers" in src:
            continue
        assert src.count(_OLD_SELECT) == 1 and src.count(_OLD_TAIL) == 1, f"unexpected upstream source {name}"
        src = src.replace(_OLD_SELECT, _NEW_SELECT).replace(_OLD_TAIL, _NEW_TAIL)
        cut = src.index("\nclass ")
        src = src[:cut] + _HELPER + src[cut:]
        path.write_text(src, encoding="utf-8")
    if str(eagle_dir) not in sys.path:
        sys.path.insert(0, str(eagle_dir))


def set_capture(model, layers):
    """Which target layers the draft receives: a sorted tuple of indices, or None for upstream (2, N/2, N-3)."""
    if layers is not None:
        layers = tuple(int(x) for x in layers)
        assert list(layers) == sorted(set(layers)), layers
    model.base_model.model.eagle_layers = layers


def default_triplet(n_layers):
    return (2, n_layers // 2, n_layers - 3)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_eagle(pair, dtype=torch.bfloat16):
    """Target + official draft as one EaModel, frozen, with the paper's tree settings."""
    patch_eagle()
    seed_all(0)
    from eagle.model.ea_model import EaModel
    model = EaModel.from_pretrained(
        base_model_path=str(PAIRS[pair]["target"]), ea_model_path=str(PAIRS[pair]["draft"]),
        total_token=PROTOCOL["total_token"], depth=PROTOCOL["depth"], top_k=PROTOCOL["top_k"],
        torch_dtype=dtype, low_cpu_mem_usage=True, device_map="cuda:0", use_eagle3=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def draft_vocab_mask(model):
    """Boolean mask over the target vocabulary: which tokens the draft can propose."""
    ea = model.ea_layer
    if hasattr(ea, "t2d"):
        return ea.t2d.bool()
    return torch.ones(model.base_model.lm_head.weight.shape[0], dtype=torch.bool, device="cuda")


# --- prompts and texts ----------------------------------------------------------------------------------------------
def chat_prompt(tokenizer, text):
    """Single user turn, no system prompt (as in EAGLE's DeepSeek eval script); Qwen3 in non-thinking mode."""
    kwargs = {"enable_thinking": False} if "enable_thinking" in (tokenizer.chat_template or "") else {}
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, **kwargs)


def benchmark_prompts(tokenizer, benches):
    """First-turn questions from the benchmark files shipped with EAGLE. benches: [(name, count or None)]."""
    rows = []
    for name, count in benches:
        lines = (EAGLE_DIR / f"eagle/data/{name}/question.jsonl").read_text(encoding="utf-8").splitlines()
        if count is not None:
            lines = lines[::max(1, len(lines) // count)][:count]      # evenly spaced subset
        for line in lines:
            item = json.loads(line)
            ids = tokenizer(chat_prompt(tokenizer, item["turns"][0]), return_tensors="pt",
                            add_special_tokens=False).input_ids
            rows.append({"id": f"{name}/{item['question_id']}", "bench": name, "ids": ids})
    return rows


def ultrachat_sequences(tokenizer, count, max_len=1024, skip=0):
    """Single-turn (prompt, human-written answer) token sequences with a mask over the answer."""
    from datasets import load_dataset
    seqs, seen = [], 0
    for item in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True):
        seen += 1
        if seen <= skip:
            continue
        msgs = item["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        prompt = tokenizer(chat_prompt(tokenizer, msgs[0]["content"]), add_special_tokens=False).input_ids
        answer = tokenizer(msgs[1]["content"] + tokenizer.eos_token, add_special_tokens=False).input_ids
        if len(prompt) + len(answer) > max_len or len(answer) < 16:
            continue
        seqs.append(with_answer_mask(torch.tensor(prompt + answer), len(prompt)))
        if len(seqs) >= count:
            break
    return seqs


def with_answer_mask(ids, prompt_len):
    mask = torch.zeros_like(ids)
    mask[prompt_len:] = 1
    return {"ids": ids, "mask": mask}


def prompt_plus_answer(prompt, answer):
    """A benchmark prompt followed by a generated answer, as a sequence with an answer mask."""
    return {"ids": torch.cat([prompt["ids"][0], answer]),
            "mask": torch.cat([torch.zeros(prompt["ids"].shape[1], dtype=torch.long),
                               torch.ones(answer.numel(), dtype=torch.long)])}
