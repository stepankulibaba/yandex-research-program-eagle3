"""Output heads of the draft: a fixed 32k vocabulary (EAGLE-3), the full vocabulary, and SpecVocab.

SpecVocab (Williams et al., 2026, arXiv 2602.13836) keeps the full output matrix U [|V|, d] but at every step
looks only at k candidate tokens:

    s_t = W_vocab (W_down h_t)        a cheap low-rank ranking of the whole vocabulary   (d' << d)
    K_t = top-k(s_t)                  candidates
    z_t = U[K_t] h_t                  exact logits of the candidates; everything else gets probability 0

Training uses no top-k: L = L_TTT(p, softmax(U h)) + lambda * L_TTT(p, softmax(s)) (lambda = 0.1 in the paper),
so both U and the ranker learn the full target distribution and the ranker's loss also shapes h.

Initialisation (ours; the paper trains from scratch). All heads start from the official draft:
  * rows of the official 32k vocabulary are the official head's rows;
  * the other ~120k rows come from the target's own output matrix, mapped into the draft head's space by a
    linear map fitted on the 32k shared tokens (least squares): draft_row ~= target_row @ A;
  * the SpecVocab ranker is the best rank-d' approximation of that full head (truncated SVD).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticHead(nn.Module):
    """EAGLE-3 as published: logits over a fixed subset of the vocabulary (draft id i = target id vocab_ids[i])."""
    kind = "static"

    def __init__(self, weight, vocab_ids):
        super().__init__()
        self.weight = nn.Parameter(weight.float().clone())          # [Vd, H]
        self.register_buffer("vocab_ids", vocab_ids.long().clone())

    def logits(self, h):
        return h @ self.weight.to(h.dtype).T

    def loss(self, h, teacher_logp):
        """Soft cross-entropy against the target distribution restricted to this vocabulary and renormalised
        (as EAGLE-3 trains its reduced head). h: [n, H]; teacher_logp: [n, |V|] full log-probabilities."""
        teacher = torch.softmax(teacher_logp[:, self.vocab_ids], -1)
        return F.cross_entropy(self.logits(h).float(), teacher, reduction="none")

    @torch.no_grad()
    def predict(self, h, k=None):
        return self.vocab_ids[self.logits(h).argmax(-1)]

    def cost(self):
        """Multiply-adds per drafted token."""
        return self.weight.numel()


class FullHead(nn.Module):
    """Logits over the whole target vocabulary."""
    kind = "full"

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.float().clone())          # [|V|, H]

    def logits(self, h):
        return h @ self.weight.to(h.dtype).T

    def loss(self, h, teacher_logp):
        return F.cross_entropy(self.logits(h).float(), teacher_logp.exp(), reduction="none")

    @torch.no_grad()
    def predict(self, h, k=None):
        return self.logits(h).argmax(-1)

    def cost(self):
        return self.weight.numel()


class SpecVocabHead(FullHead):
    """Full head U plus the low-rank ranker that picks k candidates per step."""
    kind = "specvocab"

    def __init__(self, weight, rank, aux_weight=0.1):
        super().__init__(weight)
        self.aux_weight = aux_weight
        # truncated SVD of the full head: U ~= (P S^1/2) (S^1/2 Q^T)
        P, S, Q = torch.svd_lowrank(self.weight.detach(), q=rank + 32, niter=6)
        root = S[:rank].sqrt()
        self.vocab = nn.Parameter((P[:, :rank] * root).contiguous())             # W_vocab [|V|, d']
        self.down = nn.Parameter((root[:, None] * Q[:, :rank].T).contiguous())  # W_down  [d', H]

    def rank_scores(self, h):
        return (h @ self.down.to(h.dtype).T) @ self.vocab.to(h.dtype).T

    def loss(self, h, teacher_logp):
        teacher = teacher_logp.exp()
        main = F.cross_entropy(self.logits(h).float(), teacher, reduction="none")
        aux = F.cross_entropy(self.rank_scores(h).float(), teacher, reduction="none")
        return main + self.aux_weight * aux

    @torch.no_grad()
    def candidates(self, h, k):
        return self.rank_scores(h).topk(k, -1).indices                         # [n, k]

    @torch.no_grad()
    def predict(self, h, k=None):
        """k=None: exact argmax over the full vocabulary; otherwise the SpecVocab decision over k candidates."""
        if k is None:
            return super().predict(h)
        cand = self.candidates(h, k)
        U = self.weight.to(h.dtype)
        out = []
        for i in range(0, h.shape[0], 32):          # U[K_t] is [n, k, H]: in chunks of positions to bound memory
            exact = torch.einsum("nkd,nd->nk", U[cand[i:i + 32]], h[i:i + 32])     # U[K_t] h_t
            out.append(cand[i:i + 32].gather(-1, exact.argmax(-1, keepdim=True))[:, 0])
        return torch.cat(out)

    def cost(self, k=2048):
        return self.vocab.numel() + self.down.numel() + k * self.weight.shape[1]


@torch.no_grad()
def extend_to_full(official_weight, official_ids, target_out, log=print):
    """Full-vocabulary head that keeps the official rows and fills the rest from the target's output matrix.

    official_weight: [Vd, H] official draft head; official_ids: [Vd] their target ids;
    target_out: [|V|, H] target's output matrix (lm_head; tied to the embeddings in Qwen3-1.7B)."""
    E = target_out.float()
    U_s = official_weight.float().to(E.device)
    ids = official_ids.to(E.device)
    A = torch.linalg.lstsq(E[ids], U_s, driver="gelsd" if E.device.type == "cpu" else None).solution   # [H, H]
    fit = E[ids] @ A
    r2 = 1 - (fit - U_s).pow(2).sum() / (U_s - U_s.mean(0)).pow(2).sum()
    full = E @ A
    # same row norms on average as the official rows, so new tokens neither dominate nor vanish at the start
    full *= U_s.norm(dim=-1).mean() / full[ids].norm(dim=-1).mean()
    full[ids] = U_s
    log(f"full head: {E.shape[0]} rows, {ids.numel()} official, others mapped from the target head "
        f"(least-squares fit on the shared rows: R^2 = {float(r2):.3f})")
    return full


def build_head(kind, official_weight, official_ids, full_weight=None, rank=256, aux_weight=0.1):
    if kind == "static":
        return StaticHead(official_weight, official_ids)
    if kind == "full":
        return FullHead(full_weight)
    if kind == "specvocab":
        return SpecVocabHead(full_weight, rank, aux_weight)
    raise ValueError(kind)
