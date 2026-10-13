"""CPU checks with a tiny random model (no downloads, no GPU):

    python tests/test_cpu.py

* the draft body runs 6 steps and every head gives logits / loss / predictions of the right shape;
* the full head built by extend_to_full keeps the official rows exactly;
* SpecVocab with k = |V| decides exactly like its full head, and its ranker starts as the best low-rank head;
* the training loss decreases on one text for every head;
* evaluation produces tau, candidate recall and the out-of-vocabulary share.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from draft import DraftBody  # noqa: E402
from evaluation import score_texts  # noqa: E402
from heads import FullHead, SpecVocabHead, StaticHead, extend_to_full  # noqa: E402
from train import ttt_loss  # noqa: E402

torch.manual_seed(0)
V, H, NH, KVH, HD, FF, VD = 512, 64, 4, 2, 16, 128, 96
cfg = {"hidden_size": H, "num_attention_heads": NH, "num_key_value_heads": KVH, "head_dim": HD,
       "rms_norm_eps": 1e-6, "rope_theta": 1e6}
r = lambda *s: torch.randn(*s) * 0.05  # noqa: E731
state = {"fc.weight": r(H, 3 * H), "midlayer.self_attn.q_proj.weight": r(NH * HD, 2 * H),
         "midlayer.self_attn.k_proj.weight": r(KVH * HD, 2 * H), "midlayer.self_attn.v_proj.weight": r(KVH * HD, 2 * H),
         "midlayer.self_attn.o_proj.weight": r(H, NH * HD), "midlayer.mlp.gate_proj.weight": r(FF, H),
         "midlayer.mlp.up_proj.weight": r(FF, H), "midlayer.mlp.down_proj.weight": r(H, FF),
         "midlayer.hidden_norm.weight": torch.ones(H), "midlayer.input_layernorm.weight": torch.ones(H),
         "midlayer.post_attention_layernorm.weight": torch.ones(H), "norm.weight": torch.ones(H)}
embedding = r(V, H) * 20
target_out = r(V, H) * 20
off_ids = torch.randperm(V)[:VD].sort().values
off_w = target_out[off_ids] @ r(H, H) * 5 + r(VD, H)          # official rows: roughly a linear map of the target's

# 1. full head keeps the official rows; ranker = low-rank head
full = extend_to_full(off_w, off_ids, target_out)
assert torch.allclose(full[off_ids], off_w), "official rows must be kept"
spec = SpecVocabHead(full, rank=16)
low = spec.vocab @ spec.down
assert (low - full).norm() < full.norm(), "ranker must approximate the full head"

heads = {"static": StaticHead(off_w, off_ids), "full": FullHead(full), "specvocab": spec}

# 2. shapes, k = |V| equivalence, loss decreases
T = 40
ids = torch.randint(0, V, (T,))
mask = torch.zeros(T, dtype=torch.long)
mask[10:] = 1
states = r(T, 3 * H) * 20
teacher_logp = torch.log_softmax(r(T, V) * 100, -1)
for name, head in heads.items():
    body = DraftBody(state, cfg, embedding)
    hs = body.unroll(states, ids, config.DEPTH)
    assert len(hs) == config.DEPTH and hs[0].shape == (T, H)
    pred = head.predict(hs[0].float())
    assert pred.shape == (T,) and pred.max() < V
    if name == "specvocab":
        assert torch.equal(head.predict(hs[0].float(), V), head.predict(hs[0].float())), "k=|V| must equal full"
        cand = head.candidates(hs[0].float(), 8)
        assert cand.shape == (T, 8)
    if name == "static":
        assert set(pred.tolist()) <= set(off_ids.tolist()), "static head proposes only its own vocabulary"
    opt = torch.optim.AdamW(list(body.parameters()) + list(head.parameters()), lr=1e-3)
    first = None
    for _ in range(30):
        loss = ttt_loss(body, head, states, ids, mask, teacher_logp)
        first = first if first is not None else float(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert float(loss) < first, f"{name}: loss did not decrease ({first:.3f} -> {float(loss):.3f})"
    print(f"{name:9s} loss {first:.3f} -> {float(loss):.3f}")


# 3. evaluation end to end with a fake target
class FakeTarget:
    device = "cpu"
    out_weight = target_out

    def __call__(self, ids_):
        g = torch.Generator().manual_seed(int(ids_.sum()))
        return torch.randn(len(ids_), 3 * H, generator=g), torch.randn(len(ids_), V, generator=g)


seqs = [{"ids": torch.randint(0, V, (30,)), "mask": torch.cat([torch.zeros(8), torch.ones(22)]).long(),
         "bench": "arena", "lang": lang} for lang in ("en", "ru")]
variants = {n: (DraftBody(state, cfg, embedding), h) for n, h in heads.items()}
res, oov = score_texts(FakeTarget(), variants, seqs, ks=(8, V), official_ids=off_ids)
assert "specvocab|k=8" in res and "recall_by_step" in res["specvocab|k=8"]["ru"]
assert abs(res[f"specvocab|k={V}"]["all"]["recall_by_step"][0] - 1.0) < 1e-6, "k=|V| recall must be 1"
assert res[f"specvocab|k={V}"]["all"]["tau_chain"] == res["specvocab|exact"]["all"]["tau_chain"]
assert 0 <= oov["ru"] <= 1
print("tau:", {k: round(v["all"]["tau_chain"], 3) for k, v in res.items()}, "oov:", oov)
print("ALL CPU CHECKS PASSED")
