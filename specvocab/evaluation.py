"""Offline evaluation of draft variants on the target's own greedy texts (temperature 0).

At temperature 0 the target accepts a draft guess iff it equals the target's argmax, so one target forward pass
over a greedy text tells, for every position and draft step, whether the guess would be accepted (metrics.py).

Per variant and per mode (static/full: one mode; SpecVocab: exact over the full vocabulary and with k
candidates for every k) we collect
    tau_chain   1 + mean number of leading accepted steps of a 6-step chain started at each answer position
    tau_cycles  the same in a simulation of real decoding (each region of the text counted once)
and, independent of the variant,
    oov_share   share of answer tokens outside the official 32k draft vocabulary (the static head can never propose them)
and for SpecVocab
    recall@k    share of draft steps where the token the target wants is among the k candidates.
"""
from collections import defaultdict

import numpy as np
import torch

from config import DEPTH
from metrics import chain_positions, cycle_lengths, leading_accepted


class Target:
    """Frozen target: the three captured hidden states (inputs of layers 2, N/2, N-3) and the logits."""

    def __init__(self, path, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from draft import capture_layers
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device,
                                                          attn_implementation="sdpa").eval()
        self.model.requires_grad_(False)
        self.device = device
        self.layers = capture_layers(self.model.config.num_hidden_layers)
        self.embedding = self.model.model.embed_tokens.weight
        self.out_weight = self.model.lm_head.weight          # tied to the embeddings in Qwen3-1.7B

    @torch.no_grad()
    def __call__(self, ids):
        out = self.model(input_ids=ids[None].to(self.device), output_hidden_states=True)
        states = torch.cat([out.hidden_states[i][0] for i in self.layers], -1)     # [T, 3H]
        return states, out.logits[0]                                               # [T, |V|]


class Scores:
    """Accumulates leading-accepted counts and cycles per (variant, mode, group)."""

    def __init__(self):
        self.lead = defaultdict(list)
        self.cycles = defaultdict(list)
        self.recall = defaultdict(list)          # (variant, k, group) -> per-step hit tensors
        self.oov = defaultdict(list)             # group -> per-token flags

    def add(self, key, lead, mask, groups):
        first = int(mask.nonzero()[0]) - 1      # the cycle that produces the first answer token
        last = mask.shape[0] - DEPTH - 2
        chain = lead[chain_positions(mask, DEPTH)].cpu()
        cycles = cycle_lengths(lead, first, last, DEPTH) if last >= first else []
        for g in groups:
            self.lead[key + (g,)].append(chain)
            self.cycles[key + (g,)] += cycles

    def summary(self):
        out = defaultdict(dict)
        for key, chains in self.lead.items():
            variant, mode, group = key
            lead = torch.cat(chains).float()
            out[f"{variant}|{mode}"][group] = {
                "tau_chain": 1 + float(lead.mean()) if len(lead) else None,
                "tau_cycles": float(np.mean(self.cycles[key])) if self.cycles[key] else None,
                "accept_by_depth": [float((lead >= s).float().mean()) for s in range(1, DEPTH + 1)],
                "positions": int(len(lead))}
        for (variant, k, group), hits in self.recall.items():
            h = torch.cat(hits).float()
            out[f"{variant}|k={k}"].setdefault(group, {})["recall_by_step"] = h.mean(0).tolist()
        oov = {g: float(torch.cat(v).float().mean()) for g, v in self.oov.items()}
        return dict(out), oov


@torch.no_grad()
def score_texts(target, variants, seqs, ks=(), official_ids=None, log=None):
    """variants: {name: (body, head)}; seqs: texts with ids/mask/bench/lang. Groups: bench, lang and 'all'."""
    scores = Scores()
    in_official = None
    if official_ids is not None:
        in_official = torch.zeros(target.out_weight.shape[0], dtype=torch.bool, device=target.device)
        in_official[official_ids.to(target.device)] = True
    for n, seq in enumerate(seqs):
        ids, mask = seq["ids"].to(target.device), seq["mask"].to(target.device)
        groups = {"all", seq.get("lang", "en"), f"{seq.get('lang', 'en')}/{seq.get('bench', '?')}"}
        states, logits = target(ids)
        greedy = logits.argmax(-1)
        if in_official is not None:
            answer = ids[mask.bool()]
            for g in groups:
                scores.oov[g].append((~in_official[answer]).cpu())
        T = ids.shape[0]
        for name, (body, head) in variants.items():
            with torch.autocast(target.device if target.device != "cpu" else "cpu", dtype=torch.bfloat16,
                                enabled=target.device != "cpu"):
                hs = body.unroll(states, ids, DEPTH)
            modes = [None] + (list(ks) if head.kind == "specvocab" else [])
            for k in modes:
                predicted = torch.stack([head.predict(h, k) for h in hs])            # [DEPTH, T]
                lead = leading_accepted(predicted, greedy, DEPTH)
                mode = "exact" if k is None else f"k={k}"
                scores.add((name, mode), lead, mask, groups)
                if k is not None:
                    # recall: is the token the target wants for this step among the candidates?
                    hits = []
                    for s, h in enumerate(hs, start=1):
                        wanted = torch.full_like(greedy, -1)
                        wanted[: T - s] = greedy[s:]
                        cand = head.candidates(h, k)
                        hit = (cand == wanted[:, None]).any(-1)
                        hits.append(hit[chain_positions(mask, DEPTH)].cpu())
                    hits = torch.stack(hits, -1)                                       # [P, DEPTH]
                    for g in groups:
                        scores.recall[(name, k, g)].append(hits)
        if log and (n + 1) % 50 == 0:
            log(f"   scored {n + 1}/{len(seqs)} texts")
    return scores.summary()
