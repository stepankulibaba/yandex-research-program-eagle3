"""Draft zoo: load EAGLE-3 drafts of different formats and evaluate them offline on the same target texts.

Supported checkpoint layouts
  * EAGLE / AngelSlim / thoughtworks: ``midlayer.*``, ``fc.weight``, ``norm.weight``, ``lm_head.weight``, d2t/t2d
  * speculators (RedHat, inference-optimization): ``layers.0.*`` plus config flags
    (norm_before_residual, norm_before_fc, fc_norm, eagle_aux_hidden_state_layer_ids)
  * deepseek-ai eagle3_*_ttt7: ``layers.0.*`` with q/k norms, ``target_layer_ids``, full-vocab lm_head

The evaluator reproduces inference-time drafting along the true continuation:
draft depth s (s = 1..K) at context position t takes the target features of x_1..x_t (step 1) or its own
previous output (s > 1), the embedding of x_{t+s}, attends to the context keys j <= t and to its own earlier
depths at this position, and predicts x_{t+s+1}. RoPE positions follow inference: depth s sits at t+s-1.
With greedy target text, the chain acceptance at t is the number of leading correct depths.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def _rms(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (w.float() * xf).to(x.dtype) if w is not None else xf.to(x.dtype)


def _rope(x, pos, theta):
    """x: [..., T, D] with rotate-half convention (Llama/Qwen)."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * inv[None]
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(x.dtype)
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(x.dtype)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Draft:
    """Weights of one EAGLE-3 draft in a normalised layout (bf16, on device)."""

    def __init__(self, path, n_target_layers, target_embed=None, device="cuda", layer_ids=None, name=None):
        path = Path(path)
        self.name = name or path.name
        cfg = json.loads((path / "config.json").read_text())
        files = sorted(path.glob("*.safetensors"))
        if files:
            sd = {}
            for f in files:
                sd.update(load_file(str(f)))
        else:
            sd = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        tl = cfg.get("transformer_layer_config", cfg)
        self.H = tl["hidden_size"]
        self.n_heads = tl["num_attention_heads"]
        self.n_kv = tl["num_key_value_heads"]
        self.head_dim = tl.get("head_dim") or self.H // self.n_heads
        self.eps = tl.get("rms_norm_eps", 1e-6)
        rp = tl.get("rope_parameters") or {}
        self.theta = rp.get("rope_theta") or tl.get("rope_theta") or 10000.0  # EAGLE cnets default when config has none
        self.norm_before_residual = bool(cfg.get("norm_before_residual", False))
        self.norm_before_fc = bool(cfg.get("norm_before_fc", False))
        if layer_ids is None:
            layer_ids = cfg.get("eagle_aux_hidden_state_layer_ids") or cfg.get("target_layer_ids")
        if layer_ids is None:
            layer_ids = [2, n_target_layers // 2, n_target_layers - 3]
        self.layer_ids = list(layer_ids)
        self.cfg = cfg

        def pick(*names):
            for n in names:
                if n in sd:
                    return sd[n]
            return None

        pre = "midlayer." if any(k.startswith("midlayer.") for k in sd) else "layers.0."
        get = lambda n: pick(pre + n)  # noqa: E731
        W = {
            "fc": pick("fc.weight"),
            "hidden_norm": get("hidden_norm.weight"), "input_norm": get("input_layernorm.weight"),
            "q": get("self_attn.q_proj.weight"), "k": get("self_attn.k_proj.weight"),
            "v": get("self_attn.v_proj.weight"), "o": get("self_attn.o_proj.weight"),
            "q_norm": get("self_attn.q_norm.weight"), "k_norm": get("self_attn.k_norm.weight"),
            "post_norm": get("post_attention_layernorm.weight"),
            "gate": get("mlp.gate_proj.weight"), "up": get("mlp.up_proj.weight"), "down": get("mlp.down_proj.weight"),
            "norm": pick("norm.weight"), "lm_head": pick("lm_head.weight"),
            "fc_in_norm": pick("input_norm.weight"),
        }
        fc_norms = [pick(f"fc_norms.{i}.weight", f"fc_norm.{i}.weight") for i in range(len(self.layer_ids))]
        self.norm_output = bool(cfg.get("norm_output", False))
        self.fc_norms = [w.to(device, torch.bfloat16) for w in fc_norms] if all(w is not None for w in fc_norms) else None
        missing = [k for k in ("fc", "q", "k", "v", "o", "gate", "up", "down", "lm_head", "norm") if W[k] is None]
        assert not missing, (self.name, missing, list(sd)[:20])
        assert W["fc"].shape[1] == len(self.layer_ids) * self.H, (self.name, W["fc"].shape, self.layer_ids)
        self.W = {k: (v.to(device, torch.bfloat16) if v is not None else None) for k, v in W.items()}
        emb = pick("embed_tokens.weight")
        self.embed = emb.to(device, torch.bfloat16) if emb is not None else target_embed
        self.vocab_target = self.embed.shape[0]
        d2t = pick("d2t")
        self.draft_vocab = self.W["lm_head"].shape[0]
        if self.draft_vocab == self.vocab_target:
            self.d2t_ids = torch.arange(self.draft_vocab, device=device)
        else:
            assert d2t is not None, self.name
            self.d2t_ids = torch.arange(self.draft_vocab, device=device) + d2t.to(device).long()
        self.in_vocab = torch.zeros(self.vocab_target, dtype=torch.bool, device=device)
        self.in_vocab[self.d2t_ids] = True

    def n_params(self):
        return {k: int(v.numel()) for k, v in self.W.items() if v is not None}

    # ------------------------------------------------------------------
    def fuse(self, feats):
        """feats: [T, L_all, H] captured target states (index = layer input id). Returns g: [T, H]."""
        x = feats[:, self.layer_ids, :].to(torch.bfloat16)
        if self.fc_norms is not None:
            x = torch.stack([_rms(x[:, i], w, self.eps) for i, w in enumerate(self.fc_norms)], 1)
        x = x.flatten(1)
        if self.norm_before_fc:
            x = _rms(x, self.W["fc_in_norm"], self.eps)
        return x @ self.W["fc"].to(torch.bfloat16).T

    def _qkv(self, x, pos):
        W = {k: (v.to(torch.bfloat16) if v is not None else None) for k, v in self.W.items()}
        T = x.shape[0]
        q = (x @ W["q"].T).view(T, self.n_heads, self.head_dim)
        k = (x @ W["k"].T).view(T, self.n_kv, self.head_dim)
        v = (x @ W["v"].T).view(T, self.n_kv, self.head_dim)
        if W["q_norm"] is not None:
            q = _rms(q, W["q_norm"], self.eps)
            k = _rms(k, W["k_norm"], self.eps)
        q = _rope(q.transpose(0, 1), pos, self.theta)   # [h, T, d]
        k = _rope(k.transpose(0, 1), pos, self.theta)
        return q, k, v.transpose(0, 1)

    @torch.no_grad()
    def chain(self, feats, ids, depth=6):
        return torch.stack([self.d2t_ids[l.argmax(-1)] for l in self.unroll(feats, ids, depth)])

    def trainable(self, groups):
        """Make the named weight groups trainable (fp32 master copies). groups: subset of
        {'fc', 'attn', 'mlp', 'norms', 'lm_head'}; returns the list of parameters."""
        sets = {"fc": ["fc"], "attn": ["q", "k", "v", "o", "q_norm", "k_norm"], "mlp": ["gate", "up", "down"],
                "norms": ["hidden_norm", "input_norm", "post_norm", "norm"], "lm_head": ["lm_head"]}
        params = []
        for g in groups:
            for k in sets[g]:
                if self.W.get(k) is not None:
                    self.W[k] = torch.nn.Parameter(self.W[k].float())
                    params.append(self.W[k])
        return params

    def state(self):
        return {k: v.detach().to(torch.bfloat16).cpu() for k, v in self.W.items() if v is not None}

    def unroll(self, feats, ids, depth=6):
        """Teacher-forced drafting along the true continuation (differentiable).

        feats: [T, L_all, H] target states for tokens x_0..x_{T-1}; ids: [T] token ids.
        Returns a list over depths s = 1..depth of draft-vocab logits [T, V_draft]; entry s-1 at position t
        predicts token x_{t+s+1}.
        """
        W = {k: (v.to(torch.bfloat16) if v is not None else None) for k, v in self.W.items()}
        eps = self.eps
        T = ids.shape[0]
        dev = ids.device
        rep = self.n_heads // self.n_kv
        scale = 1.0 / math.sqrt(self.head_dim)
        hidden = self.fuse(feats)
        ctx_k = ctx_v = None
        diag_k, diag_v = [], []
        preds = []
        base = torch.arange(T, device=dev)
        for s in range(1, depth + 1):
            tok = torch.zeros_like(ids)
            tok[: T - s] = ids[s:]
            emb = self.embed[tok]
            residual = hidden
            h_n = _rms(hidden, W["hidden_norm"], eps)
            if self.norm_before_residual:
                residual = h_n
            x = torch.cat([_rms(emb, W["input_norm"], eps), h_n], -1)
            q, k, v = self._qkv(x, base + s - 1)
            k = k.repeat_interleave(rep, 0)
            v = v.repeat_interleave(rep, 0)
            if s == 1:
                ctx_k, ctx_v = k, v
                att = (q @ k.transpose(1, 2)) * scale
                att = att.masked_fill(torch.ones(T, T, dtype=torch.bool, device=dev).triu(1), float("-inf"))
                p = torch.softmax(att.float(), -1).to(v.dtype)
                out = p @ v
            else:
                att_ctx = (q @ ctx_k.transpose(1, 2)) * scale
                att_ctx = att_ctx.masked_fill(torch.ones(T, T, dtype=torch.bool, device=dev).triu(1), float("-inf"))
                extra = [(q * dk).sum(-1, keepdim=True) * scale for dk in diag_k] + [(q * k).sum(-1, keepdim=True) * scale]
                att = torch.cat([att_ctx] + extra, -1)
                p = torch.softmax(att.float(), -1).to(v.dtype)
                out = p[..., :T] @ ctx_v
                for i, dv in enumerate(diag_v + [v]):
                    out = out + p[..., T + i: T + i + 1] * dv
            if s > 1:
                diag_k.append(k)
                diag_v.append(v)
            out = out.transpose(0, 1).reshape(T, -1) @ W["o"].T
            hidden_new = residual + out
            r2 = hidden_new
            m = _rms(hidden_new, W["post_norm"], eps)
            m = (F.silu(m @ W["gate"].T) * (m @ W["up"].T)) @ W["down"].T
            hidden = r2 + m
            if self.norm_output:
                # speculators norm_output: the normalised state is both the head input and the next-depth input
                hidden = _rms(hidden, W["norm"], eps)
                logits = hidden @ W["lm_head"].T
            else:
                logits = _rms(hidden, W["norm"], eps) @ W["lm_head"].T
            preds.append(logits)
        return preds


def chain_stats(pred, greedy, mask, depth):
    """Chain acceptance per position from teacher-forced predictions.

    greedy[j] = target's argmax prediction for token j+1 given the true prefix x_0..x_j (what verification
    checks). Depth s at position t is correct when pred[s-1, t] == greedy[t+s].
    mask[t] = 1 when x_t is a response token; the draft starts at t once x_{t+1} is a response token.
    Returns accepted lengths (number of leading correct depths) for valid positions.
    """
    T = greedy.shape[0]
    correct = torch.zeros(depth, T, dtype=torch.bool, device=greedy.device)
    for s in range(1, depth + 1):
        tgt = torch.full_like(greedy, -1)
        tgt[: T - s] = greedy[s:]
        correct[s - 1] = (pred[s - 1] == tgt) & (tgt >= 0)
    lead = torch.cumprod(correct.int(), 0).sum(0)
    valid = torch.zeros(T, dtype=torch.bool, device=greedy.device)
    valid[: T - depth - 1] = mask[1: T - depth].bool()
    return lead[valid], correct[:, valid]


def renewal_tau(lead_full, valid, depth):
    """Cycle-level tau as in real chain decoding: a cycle at t accepts L=lead[t] draft tokens plus one target
    token, and the next cycle starts at t + L + 1."""
    idx = valid.nonzero()
    if len(idx) == 0:
        return []
    t, end = int(idx[0]), int(idx[-1])
    lead_full = lead_full.tolist()
    out = []
    while t <= end:
        L = min(lead_full[t], depth)
        out.append(L + 1)
        t += L + 1
    return out


def chain_lead_full(pred, greedy, depth):
    T = greedy.shape[0]
    correct = torch.zeros(depth, T, dtype=torch.bool, device=greedy.device)
    for s in range(1, depth + 1):
        tgt = torch.full_like(greedy, -1)
        tgt[: T - s] = greedy[s:]
        correct[s - 1] = (pred[s - 1] == tgt) & (tgt >= 0)
    return torch.cumprod(correct.int(), 0).sum(0)
