"""Hidden states of the target and one teacher-forced draft step ("step 0"), used for analysis and for training
new draft inputs while the draft itself stays frozen.

Step 0: at position t the draft gets the fused target features g_t and the embedding of the next token x_{t+1}
and predicts x_{t+2}. The training target is the target model's own distribution for x_{t+2}, restricted to the
draft vocabulary. This is the first step of EAGLE-3's training objective (without the multi-step unroll).
"""
import numpy as np
import torch

from eagle_model import set_capture


@torch.no_grad()
def target_forward(model, ids):
    """All currently captured hidden states [T, L, H] and the target logits [T, V] for one sequence."""
    from eagle.model.utils import reset_tree_mode
    reset_tree_mode(model)
    out = model.base_model.model(input_ids=ids[None].to("cuda"))
    states = torch.stack(out.hidden_states, dim=-2)[0]
    logits = model.base_model.lm_head(out[0])[0]
    return states, logits


@torch.no_grad()
def collect_positions(model, seqs, max_positions, horizons=3, seed=0):
    """Hidden states of all N+1 layers at randomly chosen answer positions.

    labels[:, k-1] = the target's greedy token k steps after position t (k = 1: the next token).
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
        valid = [t for t in range(ids.numel() - horizons) if mask[t + 1] == 1]
        if not valid:
            continue
        pick = torch.as_tensor(np.sort(rng.choice(valid, min(per_seq, len(valid)), replace=False)))
        states.append(hs[pick.to(hs.device)].cpu())
        labels.append(torch.stack([pred[pick + k] for k in range(horizons)], -1))
        next_ids.append(ids[pick + 1])
    set_capture(model, None)
    return {"states": torch.cat(states), "labels": torch.cat(labels), "next_ids": torch.cat(next_ids)}


# --- step 0 ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def step0_batch(model, seq):
    """Everything needed for step 0 on one sequence (batch of 1).

    Returns states [1, T, L, H] of all captured layers, target_p [1, T, V_draft] (the target's distribution for
    x_{t+2}), pos_mask [1, T] (positions whose x_{t+2} is an answer token the draft can propose) and
    next_ids [1, T] (x_{t+1}, the token fed to the draft).
    """
    ea = model.ea_layer
    ids, mask = seq["ids"].cuda(), seq["mask"].cuda()
    states, logits = target_forward(model, ids)
    T = ids.numel()
    t2d = ea.t2d.bool()
    target_logits = torch.zeros_like(logits[:, t2d])
    target_logits[:-1] = logits[1:, t2d]                       # target's logits at t+1 predict x_{t+2}
    target_p = torch.softmax(target_logits.float(), -1)
    in_vocab = torch.zeros(T, device=ids.device)
    in_vocab[:-1] = t2d[logits[1:].argmax(-1)].float()
    pos_mask = torch.zeros(T, device=ids.device)
    pos_mask[:-2] = mask[2:].float()
    pos_mask = pos_mask * in_vocab
    next_ids = torch.zeros_like(ids)
    next_ids[:-1] = ids[1:]
    return states[None], target_p[None], pos_mask[None], next_ids[None]


def causal_mask(T, device):
    m = torch.full((T, T), torch.finfo(torch.float32).min, device=device)
    return torch.triu(m, 1)[None, None]


def draft_step0_logits(ea_layer, fused, next_ids):
    """One draft step with teacher forcing: fused[t] + embedding(x_{t+1}) -> logits for x_{t+2}."""
    T = fused.shape[1]
    emb = ea_layer.embed_tokens(next_ids)
    pos = torch.arange(T, device=fused.device)[None]
    out = ea_layer.midlayer(input_emb=emb, hidden_states=fused.to(emb.dtype),
                            attention_mask=causal_mask(T, fused.device), position_ids=pos, use_cache=False)[0]
    return ea_layer.lm_head(ea_layer.norm(out)).float()


def step0_loss(logits, target_p, pos_mask):
    """Soft cross-entropy to the target distribution and top-1 agreement, averaged over pos_mask."""
    logp = torch.log_softmax(logits, -1)
    per_pos = -(target_p * logp).sum(-1)
    denom = pos_mask.sum().clamp_min(1)
    loss = (per_pos * pos_mask).sum() / denom
    acc = ((logits.argmax(-1) == target_p.argmax(-1)).float() * pos_mask).sum() / denom
    return loss, acc


@torch.no_grad()
def evaluate_step0(model, fusion, seqs, amp_dtype=torch.float16):
    """Held-out step-0 loss and accuracy of a draft input module (weighted by the number of positions)."""
    loss_sum = acc_sum = weight = 0.0
    for seq in seqs:
        states, target_p, pos_mask, next_ids = step0_batch(model, seq)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = draft_step0_logits(model.ea_layer, fusion(states.float()), next_ids)
        loss, acc = step0_loss(logits, target_p, pos_mask)
        w = pos_mask.sum().item()
        loss_sum += loss.item() * w
        acc_sum += acc.item() * w
        weight += w
    return {"loss": loss_sum / max(weight, 1), "acc": acc_sum / max(weight, 1), "positions": weight}
