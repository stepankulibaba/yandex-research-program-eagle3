"""Final evaluation of the trained variants on all 240 held-out questions, with per-question data for CIs.

    python final_eval.py qwen3-1.7b

For per-token routers it also records the chosen layer set at every response token and runs one causal test:
"modal" = the same trained router, but every token gets the router's most frequent layer set
(gates still computed by the router). If tau does not drop, per-token routing carries no information.
Output: results/<pair>/final_eval.pt
"""
import sys
from collections import Counter

import torch

from config import PAIRS, DATA, RESULTS
from draft import EagleDraft
from fusion import build_fusions
from metrics import leading_accepted, chain_positions, cycle_lengths
from train import Target, DEPTH, ALL_VARIANTS, token_kind

pair = sys.argv[1]
out_dir = RESULTS / pair
target = Target(PAIRS[pair]["target"])
N, H = target.n_layers, target.hidden
eval_seqs = torch.load(DATA / pair / "eval.pt")


def load(name):
    ck = torch.load(out_dir / f"{name}.pt")
    fusion = build_fusions(N, H, [name])[name]
    fusion.load_state_dict(ck["fusion"])
    fusion.eval()
    draft = EagleDraft(PAIRS[pair]["draft"], target.embedding)
    for k, v in ck["draft"].items():
        k = {"norm": "final_norm"}.get(k, k)
        if k in draft.weights:
            draft.weights[k] = v.to("cuda", torch.bfloat16)
    draft.fusion = fusion
    return draft


def force_set(fusion, layer_set):
    """Replace per-token choice by one fixed set; gates still come from the router's scores."""
    ids = torch.tensor(sorted(layer_set), device="cuda")

    def choose(states):
        scores = fusion.router_scores(states)
        sel = ids[None].expand(states.shape[0], -1)
        return sel, torch.softmax(scores.gather(-1, sel).float(), -1) * fusion.k
    fusion.choose_layers = choose


drafts = {n: load(n) for n in ALL_VARIANTS}
routers = [n for n in ALL_VARIANTS if n.startswith("token")]


@torch.no_grad()
def run(drafts, record_routing):
    per_seq = {n: [] for n in drafts}
    routing = {n: [] for n in drafts if n in routers and record_routing}
    for i, seq in enumerate(eval_seqs):
        ids, mask = seq["ids"].cuda(), seq["mask"].cuda()
        states, logits = target(ids)
        greedy = logits.argmax(-1)
        resp = mask.bool()
        first = int(mask.nonzero()[0]) - 1
        for name, draft in drafts.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = draft.predict_tokens(draft.unroll(states, ids, DEPTH))
            lead = leading_accepted(pred, greedy, DEPTH)
            chain = lead[chain_positions(mask, DEPTH)]
            last = ids.shape[0] - DEPTH - 2
            cyc = cycle_lengths(lead, first, last, DEPTH) if last >= first else []
            per_seq[name].append({"id": seq["id"], "bench": seq["bench"], "lead_sum": int(chain.sum()),
                                  "n": int(chain.numel()), "by_depth": [int((chain >= s).sum()) for s in range(1, DEPTH + 1)],
                                  "cyc_tokens": int(sum(cyc)), "cyc_n": len(cyc)})
            if name in routing:
                conf = torch.softmax(logits.float(), -1).max(-1).values[resp].cpu()
                routing[name].append({
                    "sets": draft.fusion.last_choice[resp].cpu(),
                    "kind": token_kind(target.tokenizer, ids[resp].cpu()),
                    "conf": conf, "pos": torch.arange(ids.shape[0])[resp.cpu()] - first,
                    "bench": seq["bench"],
                    # was the first draft guess at this position accepted? (same position grid as the routing)
                    "acc1": (pred[0] == torch.cat([greedy[1:], greedy[:1] * 0 - 1]))[resp].cpu(),
                })
        if i % 40 == 0:
            print(f"{i}/{len(eval_seqs)}", flush=True)
    return per_seq, routing


per_seq, routing = run(drafts, True)
modal = {}
for n in routers:
    counts = Counter(tuple(s.tolist()) for r in routing[n] for s in r["sets"])
    modal[n] = counts.most_common(1)[0][0]
    print(n, "modal set", modal[n], f"{counts.most_common(1)[0][1] / sum(counts.values()):.1%}", flush=True)
    force_set(drafts[n].fusion, modal[n])
forced, _ = run({f"{n}@modal": drafts[n] for n in routers}, False)
per_seq.update(forced)
torch.save({"per_seq": per_seq, "routing": routing, "modal": modal, "n_layers": N}, out_dir / "final_eval.pt")
for n, rows in per_seq.items():
    print(f"{n:20s} tau={1 + sum(r['lead_sum'] for r in rows) / sum(r['n'] for r in rows):.4f}")
