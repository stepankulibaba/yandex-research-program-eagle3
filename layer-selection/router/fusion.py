"""Which target layers feed the draft, and how they are combined.

Original EAGLE-3:   g = W [h_2 ; h_{N/2} ; h_{N-3}]          (W: 3H -> H, layers hard-coded)

Here:               g = W [gate_1 * norm(h_{i1}) ; ... ; gate_k * norm(h_{ik})]     (W: kH -> H)

where the k layers i1 < ... < ik are chosen out of all N+1 hidden states of the target
(0..N-1 = input of layer i, N = output of the last layer before the final norm) in one of three ways:

    fixed   a hand-picked set (the original triplet, or another one)
    static  one learned score per layer; the k best layers are used for every token
    token   a router reads the last target state of the token and picks k layers for that token

Selection is hard top-k. The token router can read a unit-length input (unit_input=True, variants *_s): with a
2048-dim input of unit-scale coordinates and Adam, every step can move the scores by ~lr*2048, which freezes the
choice after a few dozen steps; dividing by sqrt(H) (and a lower lr) keeps the scores moving slowly.
 The router learns through the gates: gate = softmax over the scores of the
selected layers, rescaled to mean 1. Early in training Gaussian noise is added to the scores so that
other layers get tried (set `noise` from the training loop; it is ignored in eval mode).
"""
import torch
import torch.nn as nn

from draft import rms_norm


class LayerFusion(nn.Module):
    def __init__(self, hidden, n_states, mode, k, layers=None, normalize=True, unit_input=False):
        super().__init__()
        assert mode in ("fixed", "static", "token")
        self.mode, self.k, self.normalize = mode, k, normalize
        self.hidden, self.unit_input = hidden, unit_input
        self.register_buffer("layers", torch.tensor(layers if layers is not None else [0] * k))   # for "fixed"
        self.W = nn.Linear(k * hidden, hidden, bias=False)                     # the projection, as in EAGLE-3
        self.gain = nn.Parameter(torch.ones(n_states, hidden)) if normalize else None   # per-layer norm scale
        if mode == "static":
            self.logits = nn.Parameter(torch.zeros(n_states))                  # one score per layer
        if mode == "token":
            self.router = nn.Linear(hidden, n_states)                          # scores from the token's last state
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)
        self.noise = 0.0
        self.last_choice = None    # [T, k] layers chosen in the last forward call (for analysis)
        self.last_gates = None     # [T, k] their gates
        self.last_score_std = None # [T] spread of the scores over layers

    def router_parameters(self):
        return [p for n, p in self.named_parameters() if n == "logits" or n.startswith("router")]

    def projection_parameters(self):
        return [p for n, p in self.named_parameters() if not (n == "logits" or n.startswith("router"))]

    def router_scores(self, states):
        x = rms_norm(states[:, -1]).float()
        if self.unit_input:
            x = x / self.hidden ** 0.5
        return self.router(x)

    def choose_layers(self, states):
        """states: [T, N+1, H]. Returns layer ids [T, k] (sorted by depth) and gates [T, k]."""
        T = states.shape[0]
        if self.mode == "fixed":
            return self.layers[None].expand(T, -1), torch.ones(T, self.k, device=states.device)
        if self.mode == "static":
            scores = self.logits[None].expand(T, -1)
        else:
            scores = self.router_scores(states)
        self.last_score_std = scores.detach().float().std(-1)
        if self.training and self.noise > 0:
            noisy = scores + self.noise * torch.randn_like(scores)
        else:
            noisy = scores
        # sort by depth so that each block of W always sees "the shallowest chosen layer", "the next one", ...
        layer_ids = noisy.topk(self.k, -1).indices.sort(-1).values
        gates = torch.softmax(scores.gather(-1, layer_ids).float(), -1) * self.k
        return layer_ids, gates

    def forward(self, states):
        T = states.shape[0]
        layer_ids, gates = self.choose_layers(states)
        x = states[torch.arange(T, device=states.device)[:, None], layer_ids]        # [T, k, H]
        if self.normalize:
            x = rms_norm(x) * self.gain[layer_ids].to(x.dtype)
        x = x * gates[..., None].to(x.dtype)
        self.last_choice = layer_ids.detach()
        self.last_gates = gates.detach()
        return self.W(x.flatten(1).to(self.W.weight.dtype)).to(torch.bfloat16)      # [T, H]


def build_fusions(n_layers, hidden, names):
    """The variants compared in the experiment. n_layers = N of the target; there are N+1 candidate states."""
    N = n_layers
    original = [2, N // 2, N - 3]
    shifted = [N // 2, round(0.7 * N), N - 3]      # best triplet of the frozen-draft study
    spec = {                 # mode, k, layers, normalize, unit_input
        "fixed_raw":   ("fixed", 3, original, False),    # EAGLE-3 as is
        "fixed_norm":  ("fixed", 3, original, True),
        "fixed_alt":   ("fixed", 3, shifted, True),
        "static_top3": ("static", 3, None, True),
        "static_top2": ("static", 2, None, True),
        "token_top3":  ("token", 3, None, True),
        "token_top2":  ("token", 2, None, True),
        "token_top3_s": ("token", 3, None, True, True),     # unit-length router input, lower router lr
        "token_top2_s": ("token", 2, None, True, True),
    }
    return {name: LayerFusion(hidden, N + 1, *spec[name]).cuda() for name in names}
