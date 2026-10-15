"""Cost of the draft head on the GPU: fixed 32k vocabulary vs full vocabulary vs SpecVocab (plain PyTorch).

    python head_speed.py            -> results/head_speed.json

Times one head call for a batch of draft hidden states: B = 1 (a chain) and B = 60 (the tree of 60 tokens EAGLE-3
drafts per cycle). SpecVocab = ranking (h W_down^T W_vocab^T), top-k, gather of the k candidate rows, exact logits.
The paper's fused kernel (one read of each candidate row) is not reproduced here, so SpecVocab is slower than it
could be; the multiply-add counts are given next to the times.
"""
import json

import torch

from config import RESULTS, K_VALUES

V, H, R, V_STATIC = 151936, 2048, 256, 32000      # Qwen3-1.7B


def timed(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000          # microseconds


def main():
    dev, dt = "cuda", torch.bfloat16
    U = torch.randn(V, H, device=dev, dtype=dt) * 0.02
    Us = U[:V_STATIC].contiguous()
    down = torch.randn(R, H, device=dev, dtype=dt) * 0.02
    vocab = torch.randn(V, R, device=dev, dtype=dt) * 0.02
    out = {"vocab": V, "hidden": H, "rank": R, "results": {}}
    for B in (1, 60):
        h = torch.randn(B, H, device=dev, dtype=dt)
        row = {"static32k": timed(lambda: (h @ Us.T).argmax(-1)), "full": timed(lambda: (h @ U.T).argmax(-1))}

        for k in K_VALUES:
            def spec():
                cand = ((h @ down.T) @ vocab.T).topk(k, -1).indices
                exact = torch.einsum("bkd,bd->bk", U[cand], h)
                return cand.gather(-1, exact.argmax(-1, keepdim=True))
            row[f"specvocab_k{k}"] = timed(spec)
        out["results"][f"B={B}"] = row
        print(f"B={B}: " + "  ".join(f"{n} {t:.0f} us" for n, t in row.items()), flush=True)
    out["multiply_adds"] = {"static32k": V_STATIC * H, "full": V * H,
                            **{f"specvocab_k{k}": V * R + R * H + k * H for k in K_VALUES}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "head_speed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
