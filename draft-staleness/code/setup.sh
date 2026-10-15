#!/bin/bash
# Environment for the staleness experiments on a fresh GPU machine (Colab A100 / H200): Python 3.11 venv with the
# versions the authors' EAGLE code runs on, the EAGLE repo at the pinned commit, and the prompt file.
# Everything is created in draft-staleness/ (the parent of code/), as in the other directions of the repository.
set -e
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
if [ ! -x .venv/bin/python ]; then
  uv venv -q -p 3.11 .venv
  uv pip install -q -p .venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  uv pip install -q -p .venv/bin/python transformers==4.53.2 accelerate==1.10.1 peft==0.15.2 safetensors \
    huggingface_hub hf_transfer pandas pyarrow numpy protobuf sentencepiece tiktoken blobfile
fi
if [ ! -d EAGLE ]; then
  git clone -q https://github.com/SafeAILab/EAGLE EAGLE
  (cd EAGLE && git checkout -q cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b)
fi
[ -f code/prompts.jsonl ] || .venv/bin/python code/build_prompts.py
.venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'transformers', transformers.__version__)"
