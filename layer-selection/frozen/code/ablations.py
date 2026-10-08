"""Which of its three input layers does the official draft rely on? (No training.)

    python ablations.py dsl-8b        -> results/h200/dsl-8b/eval_official.json, rows_*.json, baseline_outputs.pt

1. Baseline: tau of the official draft on MT-bench + GSM8K + HumanEval, paper protocol.
2. Ablations on MT-bench + GSM8K, the draft and its matrix unchanged, only the input is edited:
     drop_low / drop_mid / only_high / only_low_mid   removed inputs are replaced by their mean vector
                                                       (averaged over ~30k tokens of UltraChat)
     mid_<a>to<b>, high_<a>to<b>                       a slot reads another layer, rescaled to the same typical length
     fc_rank<r>                                        the matrix truncated to rank r
"""
import sys

import torch

from acceptance import measure_acceptance, paired_bootstrap, prefix_tau_pair, tau_summary
from config import ALL_BENCHES, PROTOCOL, generation_kwargs, log, results_dir, save_json
from draft_inputs import AblatedFC
from eagle_model import benchmark_prompts, default_triplet, load_eagle, set_capture, ultrachat_sequences
from target_states import collect_positions


def main(pair):
    out = results_dir(pair)
    model = load_eagle(pair)
    tok = model.get_tokenizer()
    N = len(model.base_model.model.layers)
    tri = default_triplet(N)
    W = model.ea_layer.fc.weight.detach().clone()
    fc_orig = model.ea_layer.fc
    H = W.shape[0]
    gen = generation_kwargs(pair)
    prompts = benchmark_prompts(tok, ALL_BENCHES)
    log(f"{pair}: N={N} H={H} triplet={tri} prompts={len(prompts)}")

    # 1. baseline; the answers are kept and reused as evaluation texts by the other experiments
    base_rows, base_out = measure_acceptance(model, prompts, PROTOCOL["max_new_tokens"], gen_kwargs=gen)
    torch.save(dict(base_out), out / "baseline_outputs.pt")
    save_json(base_rows, out / "rows_baseline.json")
    result = {"protocol": PROTOCOL, "n_layers": N, "hidden": H, "triplet": tri, "baseline": tau_summary(base_rows)}
    log(f"baseline {result['baseline']}")
    repeat, _ = measure_acceptance(model, prompts[:20], PROTOCOL["max_new_tokens"], reference=base_out, gen_kwargs=gen)
    result["determinism_20"] = tau_summary(repeat)            # is generation deterministic run to run?

    # mean vector and typical length (RMS) of every layer
    stats = collect_positions(model, ultrachat_sequences(tok, 300, max_len=1024), 30000)["states"].float()
    means = stats.mean(0)
    rms = stats.pow(2).mean(-1).sqrt().mean(0)
    del stats
    fill = torch.cat([means[t] for t in tri])

    def rescale(old, new):                                    # make layer `new` as long as layer `old` on average
        return float(rms[old] / rms[new])

    q = round(0.7 * N)
    # (name, layers to capture or None for upstream, edited input)
    conditions = [
        ("only_high", None, AblatedFC(W, fill, keep=(2,))),
        ("drop_low", None, AblatedFC(W, fill, keep=(1, 2))),
        ("drop_mid", None, AblatedFC(W, fill, keep=(0, 2))),
        ("only_low_mid", None, AblatedFC(W, fill, keep=(0, 1))),
        (f"mid_{tri[1]}to{q}", (tri[0], q, tri[2]), AblatedFC(W, fill, scale={1: rescale(tri[1], q)})),
        (f"high_{tri[2]}to{N}", (tri[0], tri[1], N), AblatedFC(W, fill, scale={2: rescale(tri[2], N)})),
        (f"high_{tri[2]}to{N - 1}", (tri[0], tri[1], N - 1), AblatedFC(W, fill, scale={2: rescale(tri[2], N - 1)})),
        (f"fc_rank{H // 4}", None, AblatedFC(W, fill, rank=H // 4)),
        (f"fc_rank{H // 2}", None, AblatedFC(W, fill, rank=H // 2)),
    ]

    # 2. ablations, compared with the baseline on the same prompts
    prompts = [p for p in prompts if p["bench"] in ("mt_bench", "gsm8k")]
    base_rows = [r for r in base_rows if r["bench"] in ("mt_bench", "gsm8k")]
    result["ablations"] = {}
    for name, capture, edited_fc in conditions:
        model.ea_layer.fc = edited_fc
        set_capture(model, capture)
        try:
            rows, _ = measure_acceptance(model, prompts, PROTOCOL["max_new_tokens"], reference=base_out, gen_kwargs=gen)
        finally:
            model.ea_layer.fc = fc_orig
            set_capture(model, None)
        s = tau_summary(rows)
        s["delta_ci95"] = paired_bootstrap(base_rows, rows)
        s["prefix_pair"] = prefix_tau_pair(base_rows, rows)
        s["capture"] = list(capture) if capture else list(tri)
        result["ablations"][name] = s
        save_json(rows, out / f"rows_{name}.json")
        log(f"{name}: tau={s['tau']:.3f} ci={s['delta_ci95']} prefix={s['prefix_pair']}")
        save_json(result, out / "eval_official.json")
    save_json(result, out / "eval_official.json")


if __name__ == "__main__":
    main(sys.argv[1])
