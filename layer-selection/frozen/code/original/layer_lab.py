"""Experiments on the target-feature fusion of EAGLE-3 drafts.

Upstream EAGLE-3 captures the residual stream entering decoder layers
(2, N//2, N-3) of the target, concatenates the three vectors and projects them
with one bias-free Linear(3H -> H). This module makes the captured set
configurable, analyses the released fusion weights and trains alternative
fusion modules against a frozen official draft.

Layer index convention used everywhere here: index i in [0, N] is the hidden
state that ENTERS decoder layer i (0 = token embeddings), i.e. HF
`hidden_states[i]`. Index N is the output of the last decoder layer before the
final norm. Upstream EAGLE-3 therefore uses (2, N//2, N-3).
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Source patch: configurable capture of target hidden states
# --------------------------------------------------------------------------

_HELPER = '''

def _eagle_capture_layers(model):
    """Layers whose input hidden state is passed to the EAGLE-3 draft."""
    layers = getattr(model, "eagle_layers", None)
    if layers is None:
        n = len(model.layers)
        return (2, n // 2, n - 3)
    return layers
'''
_OLD_SELECT = "            if idx==len(self.layers)-3 or idx==len(self.layers)//2 or idx==2:\n                all_hidden_states += (hidden_states,)\n"
_NEW_SELECT = "            if idx in _eagle_capture_layers(self):\n                all_hidden_states += (hidden_states,)\n"
_OLD_TAIL = "        hidden_states = self.norm(hidden_states)\n\n        # add hidden states from the last decoder layer\n"
_NEW_TAIL = ("        if len(self.layers) in _eagle_capture_layers(self):\n"
             "            all_hidden_states += (hidden_states,)\n" + _OLD_TAIL)


def patch_eagle(eagle_dir):
    """Make the captured layer set of the Qwen3 and Llama targets configurable.

    With `eagle_layers` unset the behaviour is identical to upstream.
    """
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
    """None restores the upstream triplet; otherwise a sorted tuple of indices."""
    if layers is not None:
        layers = tuple(int(x) for x in layers)
        assert list(layers) == sorted(set(layers)), layers
    model.base_model.model.eagle_layers = layers


def default_triplet(n_layers):
    return (2, n_layers // 2, n_layers - 3)


# --------------------------------------------------------------------------
# Loading, prompts, data
# --------------------------------------------------------------------------

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pair(base_path, draft_path, total_token=60, depth=7, top_k=10, dtype=torch.float16):
    from eagle.model.ea_model import EaModel
    model = EaModel.from_pretrained(
        base_model_path=base_path, ea_model_path=draft_path, total_token=total_token,
        depth=depth, top_k=top_k, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map="cuda:0", use_eagle3=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def chat_prompt(tokenizer, text):
    """Single user turn, no system prompt (as in EAGLE's DeepSeek eval script); Qwen3 in non-thinking mode."""
    kwargs = {"enable_thinking": False} if "enable_thinking" in (tokenizer.chat_template or "") else {}
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, **kwargs)


def eval_prompts(eagle_dir, tokenizer, counts=(("mt_bench", 24), ("gsm8k", 8), ("humaneval", 8))):
    """First-turn questions from the benchmark files shipped with EAGLE."""
    rows = []
    for name, count in counts:
        lines = (Path(eagle_dir) / f"eagle/data/{name}/question.jsonl").read_text(encoding="utf-8").splitlines()
        step = 1 if count is None else max(1, len(lines) // count)
        for line in (lines if count is None else lines[::step][:count]):
            item = json.loads(line)
            ids = tokenizer(chat_prompt(tokenizer, item["turns"][0]), return_tensors="pt",
                            add_special_tokens=False).input_ids
            rows.append({"id": f"{name}/{item['question_id']}", "bench": name, "ids": ids})
    return rows


def ultrachat_sequences(tokenizer, count, max_len=1024, skip=0, seed=0):
    """Single-turn (prompt, response) token sequences with a response mask."""
    from datasets import load_dataset
    stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    seqs, seen = [], 0
    for item in stream:
        seen += 1
        if seen <= skip:
            continue
        msgs = item["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        p = tokenizer(chat_prompt(tokenizer, msgs[0]["content"]), add_special_tokens=False).input_ids
        r = tokenizer(msgs[1]["content"] + tokenizer.eos_token, add_special_tokens=False).input_ids
        if len(p) + len(r) > max_len or len(r) < 16:
            continue
        ids = torch.tensor(p + r)
        mask = torch.zeros_like(ids)
        mask[len(p):] = 1
        seqs.append({"ids": ids, "mask": mask})
        if len(seqs) >= count:
            break
    return seqs


# --------------------------------------------------------------------------
# Acceptance length measurement (temperature 0)
# --------------------------------------------------------------------------

@torch.no_grad()
def run_tau(model, prompts, max_new_tokens=128, reference=None, gen_kwargs=None):
    """Runs EAGLE generation and records accepted draft tokens per cycle.

    tau = mean(accepted + 1) over verification cycles (bonus token included).
    `reference` maps prompt id -> generated ids to count token mismatches.
    """
    import eagle.model.ea_model as ea_module
    rows, outputs = [], {}
    original = ea_module.evaluate_posterior
    for prompt in prompts:
        accepted = []

        def traced(*args, **kwargs):
            result = original(*args, **kwargs)
            accepted.append(int(result[1]))
            return result

        ea_module.evaluate_posterior = traced
        try:
            ids = prompt["ids"].to("cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = model.eagenerate(ids.clone(), temperature=0.0, max_new_tokens=max_new_tokens, **(gen_kwargs or {}))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        finally:
            ea_module.evaluate_posterior = original
        gen = out[0, ids.shape[1]:].cpu()
        outputs[prompt["id"]] = gen
        row = {"id": prompt["id"], "bench": prompt["bench"], "cycles": len(accepted),
               "accepted_sum": int(sum(accepted)), "new_tokens": int(gen.numel()), "seconds": seconds,
               "tau": float(np.mean([a + 1 for a in accepted])) if accepted else float("nan"),
               "accepted": accepted}
        if reference is not None:
            ref = reference[prompt["id"]]
            n = min(len(ref), len(gen))
            diff = (ref[:n] != gen[:n]).nonzero()
            first = int(diff[0]) if len(diff) else (n if len(ref) != len(gen) else None)
            row["mismatch"] = first is not None
            row["first_mismatch"] = first
            # Cycles that start before the first divergent token run on text identical to the reference.
            start, pre_cycles, pre_acc = 1, 0, 0
            for a in accepted:
                if first is not None and start > first:
                    break
                pre_cycles += 1
                pre_acc += a
                start += a + 1
            row["prefix_cycles"], row["prefix_accepted"] = pre_cycles, pre_acc
        rows.append(row)
    return rows, outputs


def tau_summary(rows):
    cycles = sum(r["cycles"] for r in rows)
    acc = sum(r["accepted_sum"] for r in rows)
    out = {"tau": (acc + cycles) / max(cycles, 1), "cycles": cycles,
           "new_tokens": sum(r["new_tokens"] for r in rows), "seconds": sum(r["seconds"] for r in rows)}
    for bench in sorted({r["bench"] for r in rows}):
        sub = [r for r in rows if r["bench"] == bench]
        c = sum(r["cycles"] for r in sub)
        out[f"tau_{bench}"] = (sum(r["accepted_sum"] for r in sub) + c) / max(c, 1)
    if rows and "mismatch" in rows[0]:
        out["mismatched_prompts"] = int(sum(r["mismatch"] for r in rows))
        pc = sum(r["prefix_cycles"] for r in rows)
        out["tau_prefix"] = (sum(r["prefix_accepted"] for r in rows) + pc) / max(pc, 1)
        out["prefix_cycles"] = pc
    return out


def prefix_tau_pair(ref_rows, rows):
    """tau of the reference and of a variant over the same identical-text prefix (per prompt)."""
    ref = {r["id"]: r for r in ref_rows}
    a_c = a_s = b_c = b_s = 0
    for r in rows:
        first = r.get("first_mismatch")
        start, cyc, acc = 1, 0, 0
        for a in ref[r["id"]]["accepted"]:
            if first is not None and start > first:
                break
            cyc, acc, start = cyc + 1, acc + a, start + a + 1
        a_c, a_s = a_c + cyc, a_s + acc
        b_c, b_s = b_c + r["prefix_cycles"], b_s + r["prefix_accepted"]
    return (a_s + a_c) / max(a_c, 1), (b_s + b_c) / max(b_c, 1)


def paired_bootstrap(rows_a, rows_b, n=2000, seed=0):
    """95% interval of tau(b) - tau(a), resampling prompts (pooled ratio estimator)."""
    a = {r["id"]: r for r in rows_a}
    ids = [r["id"] for r in rows_b if r["id"] in a]
    b = {r["id"]: r for r in rows_b}
    rng = np.random.default_rng(seed)

    def pooled(rows):
        c = sum(r["cycles"] for r in rows)
        return (sum(r["accepted_sum"] for r in rows) + c) / max(c, 1)

    diffs = []
    for _ in range(n):
        pick = rng.choice(len(ids), len(ids), replace=True)
        diffs.append(pooled([b[ids[i]] for i in pick]) - pooled([a[ids[i]] for i in pick]))
    return [float(x) for x in np.quantile(diffs, [0.025, 0.975])]


# --------------------------------------------------------------------------
# Hidden-state collection
# --------------------------------------------------------------------------

@torch.no_grad()
def target_forward(model, ids):
    """All captured hidden states [T, L, H] (fp16) and target logits [T, V]."""
    from eagle.model.utils import reset_tree_mode
    reset_tree_mode(model)
    out = model.base_model.model(input_ids=ids[None].to("cuda"))
    states = torch.stack(out.hidden_states, dim=-2)[0]
    logits = model.base_model.lm_head(out[0])[0]
    return states, logits


@torch.no_grad()
def collect_positions(model, seqs, max_positions, horizons=3, seed=0):
    """Hidden states of all N+1 layers at response positions.

    labels[:, k-1] = target greedy prediction made at position t+k-1, i.e. the
    token the target would emit k steps after position t (k=1: next token).
    """
    n_layers = len(model.base_model.model.layers)
    set_capture(model, tuple(range(n_layers + 1)))
    per_seq = max(1, max_positions // max(len(seqs), 1))
    rng = np.random.default_rng(seed)
    states, labels, next_ids = [], [], []
    for seq in seqs:
        ids, mask = seq["ids"], seq["mask"]
        hs, logits = target_forward(model, ids)
        pred = logits.argmax(-1).cpu()
        T = ids.numel()
        valid = [t for t in range(T - horizons) if mask[t + 1] == 1]
        if not valid:
            continue
        pick = np.sort(rng.choice(valid, min(per_seq, len(valid)), replace=False))
        pick_t = torch.as_tensor(pick)
        states.append(hs[pick_t.to(hs.device)].cpu())
        labels.append(torch.stack([pred[pick_t + k] for k in range(horizons)], -1))
        next_ids.append(ids[pick_t + 1])
    set_capture(model, None)
    return {"states": torch.cat(states), "labels": torch.cat(labels), "next_ids": torch.cat(next_ids)}


# --------------------------------------------------------------------------
# Phase A analyses
# --------------------------------------------------------------------------

def layer_norm_stats(states):
    x = states.float()
    rms = x.pow(2).mean(-1).sqrt()
    return {"rms_mean": rms.mean(0).tolist(), "rms_p95": rms.quantile(0.95, dim=0).tolist(),
            "absmax_p99": x.abs().amax(-1).quantile(0.99, dim=0).tolist()}


def effective_rank(singular_values, energy):
    e = singular_values.double().pow(2)
    c = torch.cumsum(e, 0) / e.sum()
    return int((c < energy).sum().item() + 1)


@torch.no_grad()
def fc_decomposition(W, states, triplet, chunk=4096):
    """g = sum_k W_k h_k. Contribution shares, pairwise cosines and spectra."""
    H = W.shape[0]
    blocks = [W[:, k * H:(k + 1) * H].float().cuda() for k in range(3)]
    sq = torch.zeros(3, dtype=torch.float64)
    g_sq = 0.0
    cos = torch.zeros(3, 3, dtype=torch.float64)
    n = 0
    for i in range(0, states.shape[0], chunk):
        x = states[i:i + chunk].cuda().float()
        cs = [x[:, triplet[k]] @ blocks[k].T for k in range(3)]
        g = cs[0] + cs[1] + cs[2]
        g_sq += g.pow(2).sum().item()
        for a in range(3):
            sq[a] += cs[a].pow(2).sum().item()
            for b in range(3):
                cos[a, b] += F.cosine_similarity(cs[a], cs[b], dim=-1).sum().item()
        n += x.shape[0]
    out = {"contribution_energy_share": (sq / sq.sum()).tolist(),
           "contribution_over_g_energy": (sq / g_sq).tolist(),
           "mean_cosine": (cos / n).tolist(), "spectra": {}, "effective_rank": {}}
    for name, M in [("low", blocks[0]), ("mid", blocks[1]), ("high", blocks[2]), ("full", W.float().cuda())]:
        s = torch.linalg.svdvals(M).cpu()
        out["spectra"][name] = s.tolist()
        out["effective_rank"][name] = {str(e): effective_rank(s, e) for e in (0.5, 0.9, 0.99)}
    # Spectrum of the data-weighted map: singular values of W_k X_k^T (what the draft actually receives)
    sub = states[: min(8192, states.shape[0])].cuda().float()
    for k, name in enumerate(["low", "mid", "high"]):
        s = torch.linalg.svdvals(sub[:, triplet[k]] @ blocks[k].T).cpu()
        out["effective_rank"][name + "_data"] = {str(e): effective_rank(s, e) for e in (0.5, 0.9, 0.99)}
    return out


@torch.no_grad()
def official_fused(W, states, triplet, chunk=4096):
    Wc = W.float().cuda()
    outs = []
    for i in range(0, states.shape[0], chunk):
        x = states[i:i + chunk, list(triplet)].cuda().float().flatten(1)
        outs.append((x @ Wc.T).cpu())
    return torch.cat(outs)


@torch.no_grad()
def linear_cka(states, n=4096, seed=0):
    rng = np.random.default_rng(seed)
    idx = torch.as_tensor(rng.choice(states.shape[0], min(n, states.shape[0]), replace=False))
    x = states[idx].cuda().float()
    x = x - x.mean(0, keepdim=True)
    L = x.shape[1]
    self_hsic = torch.stack([(x[:, i].T @ x[:, i]).pow(2).sum() for i in range(L)])
    cka = torch.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            v = (x[:, i].T @ x[:, j]).pow(2).sum() / torch.sqrt(self_hsic[i] * self_hsic[j])
            cka[i, j] = cka[j, i] = v.item()
    return cka.tolist()


@torch.no_grad()
def ridge_r2(train_x, train_y, test_x, test_y, lam=1e-2):
    """Ridge regression from one layer to the official fused vector; test R^2."""
    X = train_x.cuda().double()
    Y = train_y.cuda().double()
    mx, my = X.mean(0), Y.mean(0)
    X, Y = X - mx, Y - my
    scale = X.pow(2).sum(0).mean()
    A = X.T @ X + lam * scale * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    B = torch.linalg.solve(A, X.T @ Y)
    Xt = test_x.cuda().double() - mx
    Yt = test_y.cuda().double() - my
    res = (Yt - Xt @ B).pow(2).sum()
    return float(1 - res / Yt.pow(2).sum())


def train_future_probe(train_x, train_y, test_x, test_y, head_w, final_norm, epochs=3, bs=1024, lr=1e-3, seed=0):
    """Tuned-lens style probe: logits = head(final_norm(h + A h + b)), A,b init 0.

    Labels are draft-vocabulary indices (-100 = not in draft vocab).
    Returns (held-out top-1 accuracy after each epoch, logit-lens accuracy before training).
    """
    torch.manual_seed(seed)
    H = train_x.shape[1]
    A = nn.Linear(H, H).cuda()
    nn.init.zeros_(A.weight)
    nn.init.zeros_(A.bias)
    opt = torch.optim.Adam(A.parameters(), lr=lr)
    head = head_w.cuda()

    def logits_of(x):
        x = x.cuda().float()
        return final_norm(x + A(x)) @ head.T

    def accuracy():
        hit = tot = 0
        with torch.no_grad():
            for i in range(0, test_x.shape[0], bs):
                y = test_y[i:i + bs].cuda()
                keep = y >= 0
                if keep.any():
                    pred = logits_of(test_x[i:i + bs]).argmax(-1)
                    hit += (pred[keep] == y[keep]).sum().item()
                    tot += keep.sum().item()
        return hit / max(tot, 1)

    lens = accuracy()
    per_epoch = []
    n = train_x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            y = train_y[sel].cuda()
            if (y >= 0).sum() == 0:
                continue
            loss = F.cross_entropy(logits_of(train_x[sel]), y, ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        per_epoch.append(accuracy())
    return per_epoch, lens


# --------------------------------------------------------------------------
# Fusion replacements for inference
# --------------------------------------------------------------------------

class SlotEdit(nn.Module):
    """Official fc with one slot replaced by a constant or rescaled, or W truncated."""

    def __init__(self, W, mean=None, slot=None, scale=None, rank=None):
        super().__init__()
        W = W.float()
        if rank is not None:
            U, S, Vh = torch.linalg.svd(W.cuda(), full_matrices=False)
            W = (U[:, :rank] * S[:rank]) @ Vh[:rank]
        self.register_buffer("W", W.cuda())
        self.H = W.shape[0]
        self.slot, self.scale = slot, scale
        self.register_buffer("mean", None if mean is None else mean.float().cuda())

    def forward(self, x):
        dtype = x.dtype
        x = x.float().clone()
        if self.slot is not None:
            sl = slice(self.slot * self.H, (self.slot + 1) * self.H)
            if self.mean is not None:
                x[..., sl] = self.mean
            if self.scale is not None:
                x[..., sl] = x[..., sl] * self.scale
        return (x @ self.W.T).to(dtype)


class StackAdapter(nn.Module):
    """Feeds the concatenated capture of all N+1 layers to a trained fusion module."""

    def __init__(self, fusion, n_states, hidden):
        super().__init__()
        self.fusion, self.L, self.H = fusion, n_states, hidden

    def forward(self, x):
        out = self.fusion(x.float().view(*x.shape[:-1], self.L, self.H))
        return out.to(x.dtype)


# --------------------------------------------------------------------------
# Trainable fusion variants (phase B)
# --------------------------------------------------------------------------

def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class FixedFusion(nn.Module):
    """Upstream-style concat + Linear over a fixed layer set, optional per-source RMSNorm."""

    def __init__(self, hidden, layers, norm=False, init_weight=None):
        super().__init__()
        self.layers, self.norm = list(layers), norm
        self.gain = nn.Parameter(torch.ones(len(layers), hidden)) if norm else None
        self.fc = nn.Linear(len(layers) * hidden, hidden, bias=False)
        if init_weight is not None:
            self.fc.weight.data.copy_(init_weight.float())

    def forward(self, stack):
        x = stack[..., self.layers, :]
        if self.norm:
            x = rms(x) * self.gain
        return self.fc(x.flatten(-2))


class MixFusion(nn.Module):
    """S slots, each a softmax-weighted mix of all RMS-normalised layers, then Linear(S*H -> H).

    router=True adds a token-dependent term computed from the last layer
    (a soft mixture-of-experts over layers).
    """

    def __init__(self, hidden, n_states, slots=3, init=None, router=False):
        super().__init__()
        self.S, self.L = slots, n_states
        self.logits = nn.Parameter(torch.zeros(slots, n_states))
        if init is not None:
            for s, layer in enumerate(init):
                self.logits.data[s, layer] = 4.0
        self.gain = nn.Parameter(torch.ones(slots, hidden))
        self.router = nn.Linear(hidden, slots * n_states) if router else None
        if router:
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
        self.fc = nn.Linear(slots * hidden, hidden, bias=False)

    def weights(self, stack):
        logits = self.logits
        if self.router is not None:
            logits = logits + self.router(rms(stack[..., -1, :])).view(*stack.shape[:-2], self.S, self.L)
        return torch.softmax(logits.float(), -1)

    def forward(self, stack):
        w = self.weights(stack).to(stack.dtype)
        x = rms(stack)
        if w.dim() == 2:
            mixed = torch.einsum("sl,...lh->...sh", w, x)
        else:
            mixed = torch.einsum("...sl,...lh->...sh", w, x)
        return self.fc((mixed * self.gain).flatten(-2))


class LowRankFusion(nn.Module):
    def __init__(self, hidden, layers, rank):
        super().__init__()
        self.layers = list(layers)
        self.gain = nn.Parameter(torch.ones(len(layers), hidden))
        self.down = nn.Linear(len(layers) * hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)

    def forward(self, stack):
        x = (rms(stack[..., self.layers, :]) * self.gain).flatten(-2)
        return self.up(self.down(x))


class LowRankMoEFusion(nn.Module):
    """Shared low-rank map plus top-1 routed low-rank experts (Switch-style balance loss)."""

    def __init__(self, hidden, layers, shared_rank, experts, expert_rank):
        super().__init__()
        self.layers, self.E = list(layers), experts
        d_in = len(layers) * hidden
        self.gain = nn.Parameter(torch.ones(len(layers), hidden))
        self.down = nn.Linear(d_in, shared_rank, bias=False)
        self.up = nn.Linear(shared_rank, hidden, bias=False)
        self.e_down = nn.Parameter(torch.randn(experts, d_in, expert_rank) / math.sqrt(d_in))
        self.e_up = nn.Parameter(torch.randn(experts, expert_rank, hidden) / math.sqrt(expert_rank))
        self.router = nn.Linear(d_in, experts, bias=False)
        self.aux_loss = torch.tensor(0.0)
        self.last_load = None

    def forward(self, stack):
        x = (rms(stack[..., self.layers, :]) * self.gain).flatten(-2)
        probs = torch.softmax(self.router(x).float(), -1)
        top = probs.argmax(-1)
        gate = probs.gather(-1, top[..., None]).to(x.dtype)
        expert_out = torch.einsum("...d,edr->...er", x, self.e_down)
        expert_out = torch.einsum("...er,erh->...eh", expert_out, self.e_up)
        chosen = expert_out.gather(-2, top[..., None, None].expand(*top.shape, 1, expert_out.shape[-1]))[..., 0, :]
        onehot = F.one_hot(top, self.E).float()
        load = onehot.flatten(0, -2).mean(0)
        importance = probs.flatten(0, -2).mean(0)
        self.aux_loss = self.E * (load * importance).sum()
        self.last_load = load.detach()
        return self.up(self.down(x)) + gate * chosen


def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))


def build_variants(hidden, n_layers, official_w):
    """Name -> (module, note). n_layers = number of target decoder layers (N)."""
    L = n_layers + 1
    tri = default_triplet(n_layers)
    q = n_layers // 3
    four = (2, 2 + q, 2 + 2 * q, n_layers - 3)
    full_rank_equiv = (3 * hidden * hidden) // (4 * hidden)  # rank with the same params as fc
    shared = full_rank_equiv // 2
    expert_rank = (full_rank_equiv - shared) // 4
    v = {
        "fc3_scratch": (FixedFusion(hidden, tri), f"upstream layers {tri}, no norm, random init"),
        "fc3_official_ft": (FixedFusion(hidden, tri, init_weight=official_w), "official fc fine-tuned with our recipe"),
        "fc3_norm": (FixedFusion(hidden, tri, norm=True), "upstream layers + per-source RMSNorm (EAGLE-3.1 style)"),
        "fc3_last": (FixedFusion(hidden, (2, n_layers // 2, n_layers), norm=True), "high slot = last layer output (N)"),
        "fc4_norm": (FixedFusion(hidden, four, norm=True), f"four layers {four}; +33% fusion params"),
        "mix3_uniform": (MixFusion(hidden, L), "3 slots, softmax over all layers, uniform init"),
        "mix3_init": (MixFusion(hidden, L, init=tri), "3 slots, softmax over all layers, init at upstream"),
        "router_mix": (MixFusion(hidden, L, init=tri, router=True), "token-wise router over layers (soft MoE over layers)"),
        f"lowrank_{full_rank_equiv}": (LowRankFusion(hidden, tri, full_rank_equiv), "dense low-rank, same total params as fc"),
        f"lowrank_{shared + expert_rank}": (LowRankFusion(hidden, tri, shared + expert_rank), "dense low-rank, same active params as MoE"),
        "moe_lowrank": (LowRankMoEFusion(hidden, tri, shared, 4, expert_rank),
                        f"shared rank {shared} + 4 experts rank {expert_rank}, top-1"),
    }
    return v


# --------------------------------------------------------------------------
# Draft step-0 forward used for training and held-out evaluation
# --------------------------------------------------------------------------

def causal_mask(T, device):
    m = torch.full((T, T), torch.finfo(torch.float32).min, device=device)
    return torch.triu(m, 1)[None, None]


def draft_step0_logits(ea_layer, fused, next_ids):
    """Single draft step with teacher-forced inputs: fused[t] + emb(x_{t+1}) -> x_{t+2}."""
    T = fused.shape[1]
    emb = ea_layer.embed_tokens(next_ids)
    pos = torch.arange(T, device=fused.device)[None]
    out = ea_layer.midlayer(input_emb=emb, hidden_states=fused.to(emb.dtype),
                            attention_mask=causal_mask(T, fused.device), position_ids=pos, use_cache=False)[0]
    return ea_layer.lm_head(ea_layer.norm(out)).float()


@torch.no_grad()
def target_batch(model, seq):
    """Target features of all layers plus EAGLE-3 step-0 training targets for one sequence."""
    ea = model.ea_layer
    ids, mask = seq["ids"].cuda(), seq["mask"].cuda()
    states, logits = target_forward(model, ids)
    T = ids.numel()
    t2d = ea.t2d.bool()
    target_logits = torch.zeros_like(logits[:, t2d])
    target_logits[:-1] = logits[1:, t2d]
    target_p = torch.softmax(target_logits.float(), -1)
    in_vocab = torch.zeros(T, device=ids.device)
    in_vocab[:-1] = t2d[logits[1:].argmax(-1)].float()
    pos_mask = torch.zeros(T, device=ids.device)
    pos_mask[:-2] = mask[2:].float()
    pos_mask = pos_mask * in_vocab
    next_ids = torch.zeros_like(ids)
    next_ids[:-1] = ids[1:]
    return states[None], target_p[None], pos_mask[None], next_ids[None]


def step0_loss(logits, target_p, pos_mask):
    logp = torch.log_softmax(logits, -1)
    per_pos = -(target_p * logp).sum(-1)
    denom = pos_mask.sum().clamp_min(1)
    loss = (per_pos * pos_mask).sum() / denom
    acc = (((logits.argmax(-1) == target_p.argmax(-1)).float() * pos_mask).sum() / denom)
    return loss, acc


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
