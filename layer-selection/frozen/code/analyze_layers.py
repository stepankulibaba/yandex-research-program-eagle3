"""What is in each target layer, and what the official input matrix takes from it. (No generation.)

    python analyze_layers.py dsl-8b      -> results/h200/dsl-8b/analysis.json
    needs train_data.pt (gen_data.py) and baseline_outputs.pt (ablations.py)

  norms             typical length of every layer's hidden state
  fc_decomposition  share of the draft input coming from each of the three layers
  cka               similarity of all pairs of layers
  ridge_r2_to_fused how well each single layer predicts the draft input
  future_probes     how well each layer predicts the token 1, 2, 3 steps ahead
"""
import sys

import torch

from config import ALL_BENCHES, log, results_dir, save_json
from eagle_model import (benchmark_prompts, default_triplet, draft_vocab_mask, load_eagle,
                         prompt_plus_answer)
from layer_analysis import (fc_decomposition, layer_norm_stats, linear_cka, official_fused, ridge_r2,
                            train_future_probe)
from target_states import collect_positions


def main(pair):
    out = results_dir(pair)
    model = load_eagle(pair)
    tok = model.get_tokenizer()
    N = len(model.base_model.model.layers)
    tri = default_triplet(N)
    W = model.ea_layer.fc.weight.detach().clone()

    train_seqs = list(torch.load(out / "train_data.pt")[-1500:])
    answers = torch.load(out / "baseline_outputs.pt")
    test_seqs = [prompt_plus_answer(p, answers[p["id"]]) for p in benchmark_prompts(tok, ALL_BENCHES)]
    train = collect_positions(model, train_seqs, 60000)
    test = collect_positions(model, test_seqs, 30000)
    log(f"positions train={tuple(train['states'].shape)} test={tuple(test['states'].shape)}")

    res = {"n_layers": N, "triplet": tri, "norms": layer_norm_stats(train["states"])}
    res["fc_decomposition"] = fc_decomposition(W, test["states"], tri)
    log(f"fc shares {res['fc_decomposition']['contribution_energy_share']}")
    res["cka"] = linear_cka(train["states"], n=4096)

    g_train = official_fused(W, train["states"], tri)
    g_test = official_fused(W, test["states"], tri)
    r2 = {str(layer): ridge_r2(train["states"][:, layer].float(), g_train, test["states"][:, layer].float(), g_test)
          for layer in range(N + 1)}
    r2["triplet"] = ridge_r2(train["states"][:, list(tri)].float().flatten(1), g_train,
                             test["states"][:, list(tri)].float().flatten(1), g_test)
    res["ridge_r2_to_fused"] = r2
    log(f"ridge r2 single max at {max(range(N + 1), key=lambda l: r2[str(l)])}, triplet {r2['triplet']:.3f}")
    del g_train, g_test

    # future-token probes; labels are mapped to draft-vocabulary indices (-100 = not in the draft vocabulary)
    in_draft = draft_vocab_mask(model).cpu()
    draft_index = torch.cumsum(in_draft.long(), 0) - 1

    def to_draft(y):
        return torch.where(in_draft[y], draft_index[y], torch.full_like(y, -100))

    head_w = model.base_model.lm_head.weight[in_draft.cuda()].float()
    norm = model.base_model.model.norm
    probes = {}
    for layer in range(N + 1):
        for k in (1, 2, 3):
            per_epoch, lens = train_future_probe(
                train["states"][:, layer], to_draft(train["labels"][:, k - 1]),
                test["states"][:, layer], to_draft(test["labels"][:, k - 1]), head_w, norm, epochs=3)
            probes[f"{layer}:{k}"] = {"probe_acc": per_epoch[-1], "probe_acc_per_epoch": per_epoch, "lens_acc": lens}
        log(f"probe {layer}: " + " ".join(f"k{k}={probes[f'{layer}:{k}']['probe_acc']:.3f}" for k in (1, 2, 3)))
    res["future_probes"] = probes
    save_json(res, out / "analysis.json")


if __name__ == "__main__":
    main(sys.argv[1])
