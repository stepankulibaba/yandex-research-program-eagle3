"""How useful is each single layer for the draft? (The draft stays frozen; only a new input matrix is trained.)

    python sweep_layers.py dsl-8b        -> results/h200/dsl-8b/sweep.json
    needs train_data.pt (gen_data.py) and baseline_outputs.pt (ablations.py)

For every layer l a new input is trained from {l} alone and from {l, N-3}, with the step-0 objective
(see target_states.py) on 3000 target answers. Result: held-out step-0 accuracy on the benchmark answers.
"""
import sys
import time

import numpy as np
import torch

from config import ALL_BENCHES, log, results_dir, save_json
from draft_inputs import FixedFusion
from eagle_model import benchmark_prompts, default_triplet, load_eagle, prompt_plus_answer, set_capture
from target_states import draft_step0_logits, evaluate_step0, step0_batch, step0_loss


def main(pair, n_train=3000, accum=8, lr=1e-3):
    out = results_dir(pair)
    model = load_eagle(pair)
    ea = model.ea_layer
    tok = model.get_tokenizer()
    N = len(model.base_model.model.layers)
    H = model.base_model.config.hidden_size
    tri = default_triplet(N)
    set_capture(model, tuple(range(N + 1)))

    variants = {"official": FixedFusion(H, tri, init_weight=ea.fc.weight.detach().clone())}
    for layer in range(N + 1):
        variants[f"single_{layer}"] = FixedFusion(H, [layer], norm=True)
        if layer != tri[2]:
            variants[f"pair_{layer}_{tri[2]}"] = FixedFusion(H, sorted([layer, tri[2]]), norm=True)
    variants["triplet_upstream"] = FixedFusion(H, tri, norm=True)

    data = torch.load(out / "train_data.pt")[:n_train]
    answers = torch.load(out / "baseline_outputs.pt")
    test = [prompt_plus_answer(p, answers[p["id"]]) for p in benchmark_prompts(tok, ALL_BENCHES)]

    total = n_train // accum
    training = {}
    for name, m in variants.items():
        m.cuda()
        if name == "official":
            continue
        opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + np.cos(np.pi * min(s / total, 1.0))))
        training[name] = (m, opt, sched)
    log(f"sweep {pair}: {len(training)} trainable variants, {n_train} sequences")

    started = time.time()
    for i, seq in enumerate(data):
        states, target_p, pos_mask, next_ids = step0_batch(model, seq)     # one target pass, shared by all variants
        if pos_mask.sum() == 0:
            continue
        feats = states.float()
        for name, (m, opt, sched) in training.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = draft_step0_logits(ea, m(feats), next_ids)
            loss, _ = step0_loss(logits, target_p, pos_mask)
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
        if (i + 1) % 200 == 0:
            log(f"sweep {i + 1}/{n_train} ({time.time() - started:.0f}s)")

    result = {"n_layers": N, "triplet": tri, "n_train": n_train, "train_seconds": time.time() - started,
              "step0": {name: evaluate_step0(model, m.eval(), test, torch.bfloat16) for name, m in variants.items()}}
    log("sweep: " + " ".join(f"{k}={v['acc']:.3f}" for k, v in result["step0"].items() if k.startswith("single")))
    save_json(result, out / "sweep.json")


if __name__ == "__main__":
    main(sys.argv[1])
