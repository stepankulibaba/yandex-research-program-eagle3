"""Offline acceptance metrics on a greedy target text.

At temperature 0 the target accepts a draft guess iff it equals the target's own argmax. So from one target
forward pass we know, for every position t and draft step s, whether the s-th guess would be accepted.

    chain tau   = 1 + mean over positions of the number of leading accepted steps (out of `depth`);
                  the 1 is the token the target adds itself in every verification cycle.
    cycle tau   = the same, but simulating real decoding: start a cycle at t, accept L tokens,
                  jump to t + L + 1, repeat. Counts each region of the text once.
"""
import torch


def leading_accepted(predicted, target_greedy, depth):
    """predicted: [depth, T], step s-1 at position t is the draft's guess for x_{t+s+1}.
    target_greedy: [T], the target's argmax at t, i.e. its prediction of x_{t+1}.
    Returns [T]: number of leading correct steps at each position (0..depth)."""
    T = target_greedy.shape[0]
    correct = torch.zeros(depth, T, dtype=torch.bool, device=target_greedy.device)
    for s in range(1, depth + 1):
        wanted = torch.full_like(target_greedy, -1)
        wanted[: T - s] = target_greedy[s:]          # target's choice for x_{t+s+1}
        correct[s - 1] = (predicted[s - 1] == wanted) & (wanted >= 0)
    return torch.cumprod(correct.int(), 0).sum(0)


def chain_positions(response_mask, depth):
    """Positions that start a full-depth chain whose first guess is a response token."""
    T = response_mask.shape[0]
    ok = torch.zeros(T, dtype=torch.bool, device=response_mask.device)
    ok[: T - depth - 1] = response_mask[1: T - depth].bool()
    return ok


def cycle_lengths(leading, first, last, depth):
    """Decoding simulation from position `first` to `last`: list of tokens produced per cycle."""
    leading = leading.tolist()
    t, out = first, []
    while t <= last:
        accepted = min(leading[t], depth)
        out.append(accepted + 1)
        t += accepted + 1
    return out
