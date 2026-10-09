"""Analyses of the target layers and of the official input matrix (no generation, no training of the draft)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_norm_stats(states):
    """How 'loud' each layer is: RMS of the hidden state, and its tails."""
    x = states.float()
    rms = x.pow(2).mean(-1).sqrt()
    return {"rms_mean": rms.mean(0).tolist(), "rms_p95": rms.quantile(0.95, dim=0).tolist(),
            "absmax_p99": x.abs().amax(-1).quantile(0.99, dim=0).tolist()}


def effective_rank(singular_values, energy):
    """Number of singular directions that hold `energy` of the total squared mass."""
    e = singular_values.double().pow(2)
    c = torch.cumsum(e, 0) / e.sum()
    return int((c < energy).sum().item() + 1)


@torch.no_grad()
def fc_decomposition(W, states, triplet, chunk=4096):
    """g = W_low h_low + W_mid h_mid + W_high h_high: how much each term contributes on real data.

    Returns the energy share of each term, their mean pairwise cosines, and the spectra / effective ranks of the
    three blocks of W (alone and multiplied by real states, i.e. what the draft actually receives).
    """
    H = W.shape[0]
    blocks = [W[:, k * H:(k + 1) * H].float().cuda() for k in range(3)]
    sq = torch.zeros(3, dtype=torch.float64)
    g_sq = 0.0
    cos = torch.zeros(3, 3, dtype=torch.float64)
    n = 0
    for i in range(0, states.shape[0], chunk):
        x = states[i:i + chunk].cuda().float()
        terms = [x[:, triplet[k]] @ blocks[k].T for k in range(3)]
        g = terms[0] + terms[1] + terms[2]
        g_sq += g.pow(2).sum().item()
        for a in range(3):
            sq[a] += terms[a].pow(2).sum().item()
            for b in range(3):
                cos[a, b] += F.cosine_similarity(terms[a], terms[b], dim=-1).sum().item()
        n += x.shape[0]
    out = {"contribution_energy_share": (sq / sq.sum()).tolist(),
           "contribution_over_g_energy": (sq / g_sq).tolist(),
           "mean_cosine": (cos / n).tolist(), "spectra": {}, "effective_rank": {}}
    for name, M in [("low", blocks[0]), ("mid", blocks[1]), ("high", blocks[2]), ("full", W.float().cuda())]:
        s = torch.linalg.svdvals(M).cpu()
        out["spectra"][name] = s.tolist()
        out["effective_rank"][name] = {str(e): effective_rank(s, e) for e in (0.5, 0.9, 0.99)}
    sub = states[: min(8192, states.shape[0])].cuda().float()
    for k, name in enumerate(["low", "mid", "high"]):
        s = torch.linalg.svdvals(sub[:, triplet[k]] @ blocks[k].T).cpu()
        out["effective_rank"][name + "_data"] = {str(e): effective_rank(s, e) for e in (0.5, 0.9, 0.99)}
    return out


@torch.no_grad()
def official_fused(W, states, triplet, chunk=4096):
    """The draft input g computed with the official W for every collected position."""
    Wc = W.float().cuda()
    outs = []
    for i in range(0, states.shape[0], chunk):
        x = states[i:i + chunk, list(triplet)].cuda().float().flatten(1)
        outs.append((x @ Wc.T).cpu())
    return torch.cat(outs)


@torch.no_grad()
def linear_cka(states, n=4096, seed=0):
    """Similarity of every pair of layers (linear CKA, 1 = same representation up to rotation and scale)."""
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
    """How well one layer linearly predicts the official draft input g (ridge regression, held-out R^2)."""
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
    """How much a layer knows about the token k steps ahead (tuned-lens style probe).

    logits = head(final_norm(h + A h + b)) with A, b starting at 0, i.e. the target's own output head applied to an
    intermediate layer plus a learned linear correction. Labels are draft-vocabulary indices (-100 = not in it).
    Returns (held-out top-1 accuracy after each epoch, accuracy before training = logit lens).
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
