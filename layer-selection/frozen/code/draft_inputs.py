"""Replacements for the draft's input matrix fc (g = W [h_a; h_b; h_c]).

Two kinds:
  * AblatedFC        the official W with some of its three inputs replaced by a constant mean vector,
                     rescaled, or W truncated to a lower rank. Used at inference, no training.
  * trainable inputs (FixedFusion, MixFusion, LowRankFusion, LowRankMoEFusion) that read the states of ALL target
                     layers [.., N+1, H] and are trained while the draft stays frozen.
AllLayersAdapter plugs a trainable input into EAGLE's generation loop.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eagle_model import default_triplet


class AblatedFC(nn.Module):
    """The official fc with its input edited.

    keep:  which of the three inputs stay; the others are replaced by `fill` (their mean vectors over many tokens).
    scale: {slot: factor} multiplies an input (used when a slot reads another layer: match its typical length).
    rank:  keep only the top-`rank` singular directions of W.
    """

    def __init__(self, W, fill, keep=(0, 1, 2), scale=None, rank=None):
        super().__init__()
        Wf = W.float()
        if rank:
            U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
            Wf = (U[:, :rank] * S[:rank]) @ Vh[:rank]
        self.register_buffer("W", Wf.cuda())
        self.register_buffer("fill", fill.cuda())
        self.keep, self.scale = keep, scale or {}
        self.H = W.shape[0]

    def forward(self, x):                         # x: [.., 3H], the three captured states concatenated
        dtype = x.dtype
        x = x.float().clone()
        for s in range(3):
            sl = slice(s * self.H, (s + 1) * self.H)
            if s not in self.keep:
                x[..., sl] = self.fill[sl]
            if s in self.scale:
                x[..., sl] = x[..., sl] * self.scale[s]
        return (x @ self.W.T).to(dtype)


class AllLayersAdapter(nn.Module):
    """EAGLE passes the captured states concatenated [.., L*H]; trainable inputs expect [.., L, H]."""

    def __init__(self, fusion, n_states, hidden):
        super().__init__()
        self.fusion, self.L, self.H = fusion, n_states, hidden

    def forward(self, x):
        return self.fusion(x.float().view(*x.shape[:-1], self.L, self.H)).to(x.dtype)


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class FixedFusion(nn.Module):
    """As upstream: concatenate a fixed set of layers and project with one matrix; optional per-layer RMSNorm."""

    def __init__(self, hidden, layers, norm=False, init_weight=None):
        super().__init__()
        self.layers, self.norm = list(layers), norm
        self.gain = nn.Parameter(torch.ones(len(layers), hidden)) if norm else None
        self.fc = nn.Linear(len(layers) * hidden, hidden, bias=False)
        if init_weight is not None:
            self.fc.weight.data.copy_(init_weight.float())

    def forward(self, stack):                     # stack: [.., N+1, H]
        x = stack[..., self.layers, :]
        if self.norm:
            x = rms(x) * self.gain
        return self.fc(x.flatten(-2))


class MixFusion(nn.Module):
    """S slots; each slot is a learned softmax-weighted mix of ALL (normalised) layers, then Linear(S*H -> H).

    router=True adds a per-token term computed from the last layer: a soft mixture-of-experts over layers.
    """

    def __init__(self, hidden, n_states, slots=3, init=None, router=False):
        super().__init__()
        self.S, self.L = slots, n_states
        self.logits = nn.Parameter(torch.zeros(slots, n_states))
        if init is not None:                      # start each slot concentrated on one layer
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
    """Upstream layers, normalised, projected through a rank-r bottleneck instead of the full matrix."""

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
    """A shared low-rank map plus one of E low-rank experts per token (top-1 routing, Switch-style balance loss)."""

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
        self.aux_loss = torch.tensor(0.0)         # load-balance loss of the last forward
        self.last_load = None                     # share of tokens per expert in the last forward

    def forward(self, stack):
        x = (rms(stack[..., self.layers, :]) * self.gain).flatten(-2)
        probs = torch.softmax(self.router(x).float(), -1)
        top = probs.argmax(-1)
        gate = probs.gather(-1, top[..., None]).to(x.dtype)
        expert_out = torch.einsum("...d,edr->...er", x, self.e_down)
        expert_out = torch.einsum("...er,erh->...eh", expert_out, self.e_up)
        chosen = expert_out.gather(-2, top[..., None, None].expand(*top.shape, 1, expert_out.shape[-1]))[..., 0, :]
        load = F.one_hot(top, self.E).float().flatten(0, -2).mean(0)
        importance = probs.flatten(0, -2).mean(0)
        self.aux_loss = self.E * (load * importance).sum()
        self.last_load = load.detach()
        return self.up(self.down(x)) + gate * chosen


def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))


def build_variants(hidden, n_layers, official_w):
    """The compared draft inputs: name -> (module, description). n_layers = N of the target.

    Construction order matters for reproducibility (random initialisation).
    """
    L = n_layers + 1
    tri = default_triplet(n_layers)
    q = n_layers // 3
    four = (2, 2 + q, 2 + 2 * q, n_layers - 3)
    same_params_rank = (3 * hidden * hidden) // (4 * hidden)   # low rank with as many parameters as the full fc
    shared = same_params_rank // 2
    expert_rank = (same_params_rank - shared) // 4
    return {
        "fc3_scratch": (FixedFusion(hidden, tri), f"upstream layers {tri}, no norm, random init"),
        "fc3_official_ft": (FixedFusion(hidden, tri, init_weight=official_w), "official fc fine-tuned with our recipe"),
        "fc3_norm": (FixedFusion(hidden, tri, norm=True), "upstream layers + per-source RMSNorm (EAGLE-3.1 style)"),
        "fc3_last": (FixedFusion(hidden, (2, n_layers // 2, n_layers), norm=True), "high slot = last layer output (N)"),
        "fc4_norm": (FixedFusion(hidden, four, norm=True), f"four layers {four}; +33% fusion params"),
        "mix3_uniform": (MixFusion(hidden, L), "3 slots, softmax over all layers, uniform init"),
        "mix3_init": (MixFusion(hidden, L, init=tri), "3 slots, softmax over all layers, init at upstream"),
        "router_mix": (MixFusion(hidden, L, init=tri, router=True), "token-wise router over layers (soft MoE over layers)"),
        f"lowrank_{same_params_rank}": (LowRankFusion(hidden, tri, same_params_rank), "dense low-rank, same total params as fc"),
        f"lowrank_{shared + expert_rank}": (LowRankFusion(hidden, tri, shared + expert_rank),
                                            "dense low-rank, same active params as MoE"),
        "moe_lowrank": (LowRankMoEFusion(hidden, tri, shared, 4, expert_rank),
                        f"shared rank {shared} + 4 experts rank {expert_rank}, top-1"),
    }
