"""Paths. Models are downloaded into models/<pair>/{target,draft} (see README)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # models/, data/, results/, EAGLE/ sit next to code/
DATA = ROOT / "data"
RESULTS = ROOT / "results"
EAGLE_REPO = ROOT / "EAGLE"        # git clone of SafeAILab/EAGLE, only its benchmark question files are used

PAIRS = {
    # target Qwen/Qwen3-1.7B (N = 28), draft AngelSlim/Qwen3-1.7B_eagle3
    "qwen3-1.7b": {"target": ROOT / "models/qwen3-1.7b/target", "draft": ROOT / "models/qwen3-1.7b/draft"},
    # target deepseek-ai/DeepSeek-R1-Distill-Llama-8B (N = 32), draft yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B
    "dsl-8b": {"target": ROOT / "models/dsl-8b/target", "draft": ROOT / "models/dsl-8b/draft", "llama3": True},
}
