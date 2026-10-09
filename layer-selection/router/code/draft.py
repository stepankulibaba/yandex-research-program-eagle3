"""EAGLE-3 draft model: loading the checkpoint and running it for several steps ahead.

The draft is a single decoder layer. On a known text x_0 .. x_{T-1} it is run exactly as at inference,
for all positions t in parallel:

    step 1:  input = fusion(target hidden states at t)   + embedding of x_{t+1}   ->  predicts x_{t+2}
    step s:  input = draft's own hidden state from s-1  + embedding of x_{t+s}   ->  predicts x_{t+s+1}

Attention at step s for position t sees
    * the keys/values of step 1 at positions 0..t      (the context, causal), and
    * the keys/values that position t itself produced at steps 2..s   (the draft's own earlier guesses).
That is the KV cache the EAGLE tree drafter has at inference time. Validated against the EAGLE repository:
step-1 argmax agreement 1.0, chain tau 2.395 here vs 2.406 in real EAGLE generation.
"""
import json
import math
from pathlib import Path

import torch
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


# Weight names inside an EAGLE-3 checkpoint (official EAGLE and AngelSlim format; "midlayer" = the one layer).
# The original fusion matrix "fc.weight" is not loaded: here the fusion is a separate module (fusion.py).
CHECKPOINT_KEYS = {
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
    "lm_head": "lm_head.weight",                           # over the reduced draft vocabulary (32k)
}


class EagleDraft:
    def __init__(self, path, target_embedding, device="cuda"):
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        self.hidden = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or self.hidden // self.n_heads
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        self.rope_theta = cfg.get("rope_theta", 10000.0)   # EAGLE's own default when the config has none

        state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        # bf16 copies; make_trainable() turns them into fp32 parameters
        self.weights = {name: state[key].to(device, torch.bfloat16) for name, key in CHECKPOINT_KEYS.items()}

        # EAGLE-3 drafts reuse the target's token embeddings
        self.embedding = target_embedding
        # draft token i corresponds to target token i + d2t[i]
        n_draft_vocab = self.weights["lm_head"].shape[0]
        self.draft_to_target = torch.arange(n_draft_vocab, device=device) + state["d2t"].to(device).long()

        # module that maps target hidden states [T, N+1, H] to the draft input [T, H]; set from outside
        self.fusion = None

    def make_trainable(self):
        """Turns all draft weights into fp32 parameters and returns them (the fusion is trained separately)."""
        for name in self.weights:
            self.weights[name] = torch.nn.Parameter(self.weights[name].float())
        return list(self.weights.values())

    def state(self):
        return {name: w.detach().to(torch.bfloat16).cpu() for name, w in self.weights.items()}

    def predict_tokens(self, step_logits):
        """Draft-vocab logits of every step -> target token ids, [steps, T]."""
        return torch.stack([self.draft_to_target[logits.argmax(-1)] for logits in step_logits])

    # ------------------------------------------------------------------------------------------
    def unroll(self, target_states, ids, steps):
        """Runs the draft `steps` steps ahead from every position of the text.

        target_states: [T, N+1, H] hidden states of the target for x_0..x_{T-1}; ids: [T] token ids.
        Returns a list of `steps` tensors [T, draft_vocab]; element s-1 at position t predicts x_{t+s+1}.
        """
        w = {name: p.to(torch.bfloat16) for name, p in self.weights.items()}
        T = ids.shape[0]
        positions = torch.arange(T, device=ids.device)
        future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)   # causal mask
        scale = 1.0 / math.sqrt(self.head_dim)

        hidden = self.fusion(target_states)          # [T, H]: step-1 input built from the target
        context_k = context_v = None                  # step-1 keys/values = the context cache
        own_k, own_v = [], []                         # keys/values of steps 2..s at the same position
        step_logits = []

        for step in range(1, steps + 1):
            # token fed at this step is x_{t+step}; past the end of the text: 0 (those positions are never scored)
            next_token = torch.zeros_like(ids)
            next_token[: T - step] = ids[step:]

            residual = hidden
            x = torch.cat([rms_norm(self.embedding[next_token], w["input_norm"], self.eps),
                           rms_norm(hidden, w["hidden_norm"], self.eps)], -1)          # [T, 2H]
            # the step-s guess for position t sits at RoPE position t + s - 1, as at inference
            q, k, v = self._qkv(x, positions + step - 1, w)                           # [heads, T, head_dim]

            if step == 1:
                context_k, context_v = k, v
                scores = (q @ k.transpose(1, 2)) * scale
                scores = scores.masked_fill(future, float("-inf"))
                probs = torch.softmax(scores.float(), -1).to(v.dtype)
                attn = probs @ v
            else:
                ctx_scores = (q @ context_k.transpose(1, 2)) * scale                    # [heads, T, T]
                ctx_scores = ctx_scores.masked_fill(future, float("-inf"))
                # each position also attends to its own keys from steps 2..step: one score per key
                self_scores = [(q * kk).sum(-1, keepdim=True) * scale for kk in own_k + [k]]
                probs = torch.softmax(torch.cat([ctx_scores] + self_scores, -1).float(), -1).to(v.dtype)
                attn = probs[..., :T] @ context_v
                for i, vv in enumerate(own_v + [v]):
                    attn = attn + probs[..., T + i: T + i + 1] * vv
                own_k.append(k)
                own_v.append(v)

            attn = attn.transpose(0, 1).reshape(T, -1) @ w["o"].T
            hidden = residual + attn
            m = rms_norm(hidden, w["post_norm"], self.eps)
            hidden = hidden + (F.silu(m @ w["gate"].T) * (m @ w["up"].T)) @ w["down"].T
            step_logits.append(rms_norm(hidden, w["final_norm"], self.eps) @ w["lm_head"].T)
        return step_logits

    def _qkv(self, x, positions, w):
        """Projections + RoPE; grouped-query heads are repeated to the full number of heads."""
        T = x.shape[0]
        q = (x @ w["q"].T).view(T, self.n_heads, self.head_dim).transpose(0, 1)
        k = (x @ w["k"].T).view(T, self.n_kv_heads, self.head_dim).transpose(0, 1)
        v = (x @ w["v"].T).view(T, self.n_kv_heads, self.head_dim).transpose(0, 1)
        q = apply_rope(q, positions, self.rope_theta)
        k = apply_rope(k, positions, self.rope_theta)
        repeat = self.n_heads // self.n_kv_heads
        return q, k.repeat_interleave(repeat, 0), v.repeat_interleave(repeat, 0)
