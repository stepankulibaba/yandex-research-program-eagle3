"""Paths, model pairs and the evaluation protocol shared by all experiments of the frozen-draft study."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # layer-selection/frozen: models/, EAGLE/, results/ live here
EAGLE_DIR = ROOT / "EAGLE"                         # git clone of SafeAILab/EAGLE at commit cb7e0841

# target model and its EAGLE-3 draft
PAIRS = {
    # deepseek-ai/DeepSeek-R1-Distill-Llama-8B + yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B (N = 32)
    "dsl-8b": {"target": ROOT / "models/dsl-8b/base", "draft": ROOT / "models/dsl-8b/draft", "llama3": True},
    # Qwen/Qwen3-1.7B + AngelSlim/Qwen3-1.7B_eagle3 (N = 28)
    "qwen3-1.7b": {"target": ROOT / "models/qwen3-1.7b/base", "draft": ROOT / "models/qwen3-1.7b/draft", "llama3": False},
}

# Protocol of the EAGLE-3 paper (official DeepSeek eval script): first turn only, no system prompt, temperature 0,
# tree of 60 tokens, depth 5, top-k 10. eagenerate's defaults cap the answer at 512 new tokens / 2048 in total.
PROTOCOL = {"total_token": 60, "depth": 5, "top_k": 10, "max_new_tokens": 512, "max_length": 2048}

# benchmark questions shipped with EAGLE; None = all questions of the file
BENCHES = [["mt_bench", None], ["gsm8k", None]]
ALL_BENCHES = BENCHES + [["humaneval", None]]


def generation_kwargs(pair):
    """Extra arguments of EaModel.eagenerate for this pair."""
    kwargs = {"is_llama3": True} if PAIRS[pair]["llama3"] else {}
    return dict(kwargs, max_length=PROTOCOL["max_length"])


def results_dir(pair):
    path = ROOT / "results/h200" / pair
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
