"""Equivalent CPU tensor construction of EAGLE tree metadata.

The parent device-to-host transfer is deliberately retained. Besides replacing
Python loops, this variant replaces upstream's GPU unique(parent).tolist()
with CPU leaf detection. It does not eliminate all GPU synchronizations.
"""
import torch


def tensor_metadata(mask_index, gpu_unique=False):
    """Input: parent indices for non-root nodes, in topological node order.

    Output devices/shapes match upstream greedy metadata before positions move
    to the draft device. Sampling path ordering is intentionally unsupported.
    """
    parent = torch.cat((torch.zeros(1, dtype=torch.long),
                        mask_index.detach().to(device="cpu", dtype=torch.long)))
    n = parent.numel()
    mask = torch.eye(n, dtype=torch.bool)
    mask[:, 0] = True
    # Each round doubles the covered ancestor distance. Root points to itself.
    jump = parent
    for _ in range((n - 1).bit_length()):
        mask = mask | mask[jump]
        jump = jump[jump]
    positions = mask.sum(dim=1) - 1
    width = int(positions.max()) + 1
    is_leaf = torch.ones(n, dtype=torch.bool)
    if gpu_unique:
        # Factorial control: retain upstream's second device-to-host operation.
        is_leaf[torch.unique(mask_index).tolist()] = False
    else:
        is_leaf[parent[1:]] = False
    leaves = is_leaf.nonzero(as_tuple=True)[0]
    # Walk every leaf upwards in parallel, then reverse its valid prefix.
    current = leaves
    reverse_paths = [current]
    for _ in range(width - 1):
        current = parent[current]
        reverse_paths.append(current)
    reverse_paths = torch.stack(reverse_paths, dim=1)
    cols = torch.arange(width)[None, :]
    gather_index = (positions[leaves, None] - cols).clamp_min(0)
    paths = reverse_paths.gather(1, gather_index)
    paths = paths.masked_fill(cols > positions[leaves, None], -1)
    return paths, mask.float()[None, None], positions


def finish_tensor_metadata(draft_tokens, mask_index, device, logits_processor, gpu_unique=False):
    if logits_processor is not None:
        raise ValueError("Pilot supports greedy only; sampling ordering is untouched.")
    paths, mask, positions = tensor_metadata(mask_index, gpu_unique=gpu_unique)
    return draft_tokens[None], paths, mask, positions.to(device)
