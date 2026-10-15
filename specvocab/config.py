"""Paths, the model pair and the variants of the experiment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / os.environ.get("SPECVOCAB_DATA", "data")            # SMOKE=1 in run.sh: data_smoke
RESULTS = ROOT / os.environ.get("SPECVOCAB_RESULTS", "results")   # SMOKE=1 in run.sh: results/smoke
EAGLE_REPO = ROOT / "EAGLE"        # git clone of SafeAILab/EAGLE: only its benchmark question files are used

# target Qwen/Qwen3-1.7B (N = 28, |V| = 151 936, H = 2048) and its public EAGLE-3 draft (32k vocabulary)
TARGET = ROOT / "models/qwen3-1.7b/target"
DRAFT = ROOT / "models/qwen3-1.7b/draft"
TARGET_REPO = "Qwen/Qwen3-1.7B"
DRAFT_REPO = "AngelSlim/Qwen3-1.7B_eagle3"

DEPTH = 6              # draft steps per position (training-time test), as in our earlier runs
STEP_DECAY = 0.8       # loss weight of step s: 0.8^(s-1), as in EAGLE-3
MAX_NEW_TOKENS = 512   # length of the target's greedy answers

# variants trained side by side from the same official draft:  name -> (head kind, extra arguments)
VARIANTS = {
    "static32k": ("static", {}),                              # EAGLE-3 as is: fixed 32k vocabulary
    "full":      ("full", {}),                                # full vocabulary (upper bound, slow head)
    "specvocab": ("specvocab", {"rank": 256, "aux_weight": 0.1}),   # d' = d/8, lambda = 0.1 (paper's best)
}
K_VALUES = [256, 512, 1024, 2048, 4096]      # SpecVocab candidates evaluated (the paper uses k = 2048)

LANGS = ["en", "ru", "uk", "de", "zh", "ja"]
