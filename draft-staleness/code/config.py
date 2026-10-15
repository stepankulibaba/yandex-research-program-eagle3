"""Paths and protocol of the drafter-staleness map (official EAGLE-3 Llama-3.1 draft on fine-tuned targets)."""
import json
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent              # code/: scripts and the prompt files
ROOT = CODE.parent                                  # draft-staleness/: models/, EAGLE/, results/, figures/
EAGLE_DIR = ROOT / "EAGLE"                          # git clone of SafeAILab/EAGLE at commit cb7e0841
MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "llama31-8b"
PAIRS = {}                                          # eagle_model.load_eagle is not used here

# EAGLE-3 tree for LLaMA-3.1-8B as in our reproduction (eval_tau.py): 60 tokens, depth 7, top-k 10
PROTOCOL = {"total_token": 60, "depth": 7, "top_k": 10, "max_new_tokens": 512, "max_length": 2048}

DRAFTS = {
    "inst": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",            # official draft of Llama-3.1-8B-Instruct
    "r1": "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",      # official draft of R1-Distill-Llama-8B
}

# name -> HF repo (full weights) or {"lora": adapter repo, "base": base repo}; all share the Llama-3 tokenizer ids
TARGETS = {
    "base": "unsloth/Meta-Llama-3.1-8B-Instruct",              # = meta-llama/Llama-3.1-8B-Instruct, ungated mirror
    "abliterated": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    "limo_lora": {"lora": "t83714/llama-3.1-8b-instruct-limo-lora-adapter", "base": "unsloth/Meta-Llama-3.1-8B-Instruct"},
    "toolace": "Team-ACE/ToolACE-8B",
    "huatuo_o1": "FreedomIntelligence/HuatuoGPT-o1-8B",
    "ultramedical": "TsinghuaC3I/Llama-3.1-8B-UltraMedical",
    "supernova": "arcee-ai/Llama-3.1-SuperNova-Lite",
    "vikhr": "Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24",
    "nemotron_nano": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "r1_distill": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",  # sister model: from Llama-3.1-8B base, not Instruct
}

# runs in order: (target, draft); the last two complete the 2x2 (Instruct, R1) x (draft inst, draft r1)
RUNS = [(t, "inst") for t in TARGETS] + [("r1_distill", "r1"), ("base", "r1")]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
