"""EAGLE-3 draft (one decoder layer) on a known text, run for several steps ahead as at inference.

The draft body is the official checkpoint (AngelSlim/Qwen3-1.7B_eagle3): fusion fc, one decoder layer, final norm.
The output head is separate (heads.py), so that the same body can be paired with a fixed 32k vocabulary,
the full vocabulary or SpecVocab.

On a text x_0 .. x_{T-1} the draft runs for all positions t in parallel:

    step 1:  input = fc(target states of layers 2, N/2, N-3 at t) + embedding of x_{t+1}  ->  predicts x_{t+2}
    step s:  input = draft's own hidden state from step s-1      + embedding of x_{t+s}  ->  predicts x_{t+s+1}

Attention at step s for position t sees the step-1 keys/values of positions 0..t (the context) and the
keys/values position t itself produced at steps 2..s: the KV cache of the EAGLE drafter at inference.
This is the code of our layer study (routed_release/draft.py), checked there against the EAGLE repository.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x, weight=None, eps=1e-6):
    """RMSNorm computed in fp32 and returned in the input dtype."""
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        x32 = weight.float() * x32
    return x32.to(x.dtype)


def apply_rope(x, positions, theta):
    """Rotary position embedding, rotate-half convention (Llama / Qwen). x: [heads, T, head_dim]."""
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    angles = positions.float()[:, None] * inv_freq[None]
    cos = torch.cat([angles.cos(), angles.cos()], -1).to(x.dtype)
    sin = torch.cat([angles.sin(), angles.sin()], -1).to(x.dtype)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


# weight names inside an EAGLE-3 checkpoint ("midlayer" = the one layer)
CHECKPOINT_KEYS = {
    "fc": "fc.weight",                                     # [H, 3H]: fusion of the three target states
    "q": "midlayer.self_attn.q_proj.weight",
    "k": "midlayer.self_attn.k_proj.weight",
    "v": "midlayer.self_attn.v_proj.weight",
    "o": "midlayer.self_attn.o_proj.weight",
    "gate": "midlayer.mlp.gate_proj.weight",
    "up": "midlayer.mlp.up_proj.weight",
    "down": "midlayer.mlp.down_proj.weight",
    "hidden_norm": "midlayer.hidden_norm.weight",          # norm of the incoming hidden state
    "input_norm": "midlayer.input_layernorm.weight",       # norm of the token embedding
    "post_norm": "midlayer.post_attention_layernorm.weight",
    "final_norm": "norm.weight",
}


def load_checkpoint(path):
    """State dict and config of an EAGLE-3 draft directory (pytorch_model.bin or model.safetensors)."""
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    if (path / "pytorch_model.bin").exists():
        state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        state = load_file(str(path / "model.safetensors"))
    return state, cfg


class DraftBody(nn.Module):
    """Everything of the draft except the output head. Parameters are fp32; the forward runs in bf16."""

    def __init__(self, state, cfg, target_embedding):
        super().__init__()
        self.hidden = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or self.hidden // self.n_heads
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        self.rope_theta = cfg.get("rope_theta", 10000.0)    # EAGLE's own default when the config has none
        self.w = nn.ParameterDict({name: nn.Parameter(state[key].float().clone())
                                   for name, key in CHECKPOINT_KEYS.items()})
        self.embedding = target_embedding                   # EAGLE-3 drafts reuse the target's token embeddings

    def unroll(self, states, ids, steps):
        """states: [T, 3H] concatenated target states (layers 2, N/2, N-3); ids: [T].
        Returns `steps` tensors [T, H]: the final-normed hidden state of each step (input of the head)."""
        w = {name: p.to(torch.bfloat16) for name, p in self.w.items()}
        T = ids.shape[0]
        positions = torch.arange(T, device=ids.device)
        future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
        scale = 1.0 / math.sqrt(self.head_dim)

        hidden = states.to(torch.bfloat16) @ w["fc"].T       # [T, H]
        context_k = context_v = None
        own_k, own_v = [], []
        outputs = []
        for step in range(1, steps + 1):
            next_token = torch.zeros_like(ids)               # past the end of the text: 0 (never scored)
            next_token[: T - step] = ids[step:]
            residual = hidden
            x = torch.cat([rms_norm(self.embedding[next_token].to(torch.bfloat16), w["input_norm"], self.eps),
                           rms_norm(hidden, w["hidden_norm"], self.eps)], -1)
            q, k, v = self._qkv(x, positions + step - 1, w)   # step-s guess for t sits at RoPE position t+s-1
            if step == 1:
                context_k, context_v = k, v
                scores = ((q @ k.transpose(1, 2)) * scale).masked_fill(future, float("-inf"))
                attn = torch.softmax(scores.float(), -1).to(v.dtype) @ v
            else:
                ctx = ((q @ context_k.transpose(1, 2)) * scale).masked_fill(future, float("-inf"))
                own = [(q * kk).sum(-1, keepdim=True) * scale for kk in own_k + [k]]
                probs = torch.softmax(torch.cat([ctx] + own, -1).float(), -1).to(v.dtype)
                attn = probs[..., :T] @ context_v
                for i, vv in enumerate(own_v + [v]):
                    attn = attn + probs[..., T + i: T + i + 1] * vv
                own_k.append(k)
                own_v.append(v)
            attn = attn.transpose(0, 1).reshape(T, -1) @ w["o"].T
            hidden = residual + attn
            m = rms_norm(hidden, w["post_norm"], self.eps)
            hidden = hidden + (F.silu(m @ w["gate"].T) * (m @ w["up"].T)) @ w["down"].T
            outputs.append(rms_norm(hidden, w["final_norm"], self.eps))
        return outputs

    def _qkv(self, x, positions, w):
        T = x.shape[0]
        q = (x @ w["q"].T).view(T, self.n_heads, self.head_dim).transpose(0, 1)
        k = (x @ w["k"].T).view(T, self.n_kv_heads, self.head_dim).transpose(0, 1)
        v = (x @ w["v"].T).view(T, self.n_kv_heads, self.head_dim).transpose(0, 1)
        q = apply_rope(q, positions, self.rope_theta)
        k = apply_rope(k, positions, self.rope_theta)
        repeat = self.n_heads // self.n_kv_heads
        return q, k.repeat_interleave(repeat, 0), v.repeat_interleave(repeat, 0)


def capture_layers(n_layers):
    """Target layers whose input states feed the EAGLE-3 draft (the authors' hard-coded triplet)."""
    return [2, n_layers // 2, n_layers - 3]
