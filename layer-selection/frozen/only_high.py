"""Official draft, no retraining: keep only the N-3 slot, low and mid replaced by their dataset means."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import layer_lab as lab
from phase_a import _AllMean  # noqa: F401  (same module family)

SP = Path(sys.argv[1])
OUT = Path(__file__).parent / "results/only_high"
OUT.mkdir(parents=True, exist_ok=True)
lab.patch_eagle(SP / "EAGLE")
lab.seed_all(0)
model = lab.load_pair(str(SP / "models/target"), str(SP / "models/draft"))
tok = model.get_tokenizer()
N = len(model.base_model.model.layers)
tri = lab.default_triplet(N)
W = model.ea_layer.fc.weight.detach().clone()
fc_orig = model.ea_layer.fc
H = W.shape[0]

prompts = lab.eval_prompts(SP / "EAGLE", tok, (("mt_bench", 32), ("gsm8k", 8), ("humaneval", 8)))
base_rows, base_out = lab.run_tau(model, prompts, 128)
print("baseline", lab.tau_summary(base_rows), flush=True)

seqs = lab.ultrachat_sequences(tok, 200, max_len=1024)
means = lab.collect_positions(model, seqs, 20000)["states"].float().mean(0)


class KeepSlots(torch.nn.Module):
    """Official fc; slots not in `keep` are replaced by their mean vectors."""

    def __init__(self, keep):
        super().__init__()
        self.keep = keep
        self.register_buffer("W", W.float().cuda())
        self.register_buffer("fill", torch.cat([means[tri[s]] for s in range(3)]).cuda())

    def forward(self, x):
        dtype = x.dtype
        x = x.float().clone()
        for s in range(3):
            if s not in self.keep:
                x[..., s * H:(s + 1) * H] = self.fill[s * H:(s + 1) * H]
        return (x @ self.W.T).to(dtype)


result = {"baseline": lab.tau_summary(base_rows), "triplet": tri}
for name, keep in [("only_high", [2]), ("only_low_mid", [0, 1]), ("drop_low", [1, 2]), ("drop_mid", [0, 2])]:
    model.ea_layer.fc = KeepSlots(keep)
    try:
        rows, _ = lab.run_tau(model, prompts, 128, reference=base_out)
    finally:
        model.ea_layer.fc = fc_orig
    result[name] = lab.tau_summary(rows)
    result[name]["delta_ci95"] = lab.paired_bootstrap(base_rows, rows)
    result[name]["prefix_pair"] = lab.prefix_tau_pair(base_rows, rows)
    print(name, json.dumps(result[name]), flush=True)
lab.dump(result, OUT / "only_high.json")
