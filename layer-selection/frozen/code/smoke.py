"""Tiny local run of both phases to catch bugs before spending Kaggle GPU time."""
import sys
from pathlib import Path

SP = Path(sys.argv[1])
which = sys.argv[2] if len(sys.argv) > 2 else "ab"
sys.path.insert(0, str(Path(__file__).parent))

common = {"eagle_dir": str(SP / "EAGLE"), "base": str(SP / "models/target"), "draft": str(SP / "models/draft"),
          "seed": 0, "total_token": 60, "depth": 7, "top_k": 10, "max_new_tokens": 24,
          "prompt_counts": [["mt_bench", 2], ["gsm8k", 1], ["humaneval", 1]], "max_len": 384}

if "a" in which:
    import phase_a
    phase_a.run(dict(common, out=str(SP / "smoke_a"), n_train_seqs=8, train_positions=600, test_positions=400,
                     cka_tokens=256, horizons=2, probe_epochs=1, probe_layers=[2, 14, 28],
                     shifts=[[1], [16], [28]], unscaled_shifts={"2": [28]}, fc_ranks=[256]))
if "b" in which:
    import phase_b
    phase_b.run(dict(common, out=str(SP / "smoke_b"), skip=100, epochs=1, accum=2, lr=1e-3,
                     fast_lr_mult=10.0, lr_mult={"fc3_official_ft": 0.1}, warmup_frac=0.1, clip=1.0,
                     aux_coef=0.01, log_every=4, variants=None, max_train_seconds=3600, max_len=1024, n_train_seqs=24))
