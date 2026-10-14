"""Reuse a supplied NeMo fused autograd function for pitched batch targets.

No fallback kernel: production callers must pass the actual fused apply.
The count is global, so splitting the batch does not average per-batch means.
"""
import torch

def batch_sliced_soft_ce(logits,target_probs,position_mask,*,fused_apply):
    if logits.ndim!=3 or target_probs.shape!=logits.shape:
        raise ValueError('Expected identical [B,S,V] logits and teacher shapes')
    B,S,V=logits.shape
    if not B or not S or position_mask.shape!=(B,S,1):
        raise ValueError('Expected positive batch/sequence and [B,S,1] mask')
    if target_probs.requires_grad:
        raise ValueError('Teacher must be frozen')
    if target_probs.stride(2)!=1 or target_probs.stride(1)!=V or target_probs.stride(0)<S*V:
        raise ValueError('Only dense rows with positive batch pitch are supported')
    if not logits.is_contiguous():
        raise ValueError('Draft logits must be contiguous; no implicit logits copy')
    mask=position_mask.squeeze(-1).float()
    if not mask.is_contiguous():
        raise ValueError('Mask must be contiguous; no implicit mask copy')
    count=mask.sum().clamp_min(1.0)
    # Preserve the original single dispatch whenever the teacher is flattenable.
    if B==1 or target_probs.stride(0)==S*V:
        return fused_apply(logits,target_probs,mask,count)
    losses=[]
    for batch_index in range(B):
        x=logits[batch_index:batch_index+1]
        p=target_probs[batch_index:batch_index+1]
        m=mask[batch_index:batch_index+1]
        assert p.is_contiguous() and p.untyped_storage().data_ptr()==target_probs.untyped_storage().data_ptr()
        losses.append(fused_apply(x,p,m,count))
    return torch.stack(losses).sum()
