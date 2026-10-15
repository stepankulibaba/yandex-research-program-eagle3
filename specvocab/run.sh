#!/bin/bash
# SpecVocab vs fixed 32k vocabulary on Qwen3-1.7B, tested on six languages. One command, restartable.
#   bash run.sh                 # English training data, as in our earlier runs (6000 UltraChat + 2000 GSM8K)
#   DATA=ml bash run.sh         # + multilingual prompts (oasst2: ru uk de zh ja), results in results/ml
#   SMOKE=1 bash run.sh         # 10-minute check of every step on a few texts (results in results/smoke_*)
# Every step is skipped if its output exists, so after a failure just run the same command again.
# Needs: Python 3.10+, CUDA 12 driver, one GPU with >= 40 GB (A100/H100/H200), internet (Hugging Face).
set -e
cd "$(dirname "$0")"
RUN=${DATA:-en}
mkdir -p logs
log() { echo "$(date '+%F %T') $*" | tee -a logs/run.log; }

# 1. environment: PYTHON=/path/to/python uses an existing interpreter with requirements.txt installed;
#    otherwise .venv is created with uv, venv or virtualenv (whichever works) and the requirements installed
if [ -n "$PYTHON" ]; then
  PY=$PYTHON
else
  if [ ! -x .venv/bin/python ]; then
    log "creating .venv"
    if command -v uv > /dev/null; then
      uv venv -q -p 3.11 .venv && uv pip install -q -p .venv/bin/python -r requirements.txt
    elif python3 -m venv .venv 2> /dev/null; then
      .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt
    else
      python3 -m pip install -q --user virtualenv && python3 -m virtualenv -q .venv && \
        .venv/bin/pip install -q -r requirements.txt
    fi
  fi
  PY=.venv/bin/python
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY tests/test_cpu.py > logs/test_cpu.log 2>&1 || { log "CPU checks failed: logs/test_cpu.log"; exit 1; }

# 2. benchmark questions (MT-bench, GSM8K, HumanEval) from the EAGLE repository, pinned commit
if [ ! -d EAGLE ]; then
  git clone -q https://github.com/SafeAILab/EAGLE EAGLE
  (cd EAGLE && git checkout -q cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b)
fi

# 3. models: Qwen3-1.7B (~4 GB) and its public EAGLE-3 draft
$PY - <<'EOF'
from pathlib import Path
from huggingface_hub import snapshot_download
for repo, path in [("Qwen/Qwen3-1.7B", "models/qwen3-1.7b/target"),
                   ("AngelSlim/Qwen3-1.7B_eagle3", "models/qwen3-1.7b/draft")]:
    if not (Path(path) / "config.json").exists():
        print("downloading", repo, flush=True)
        snapshot_download(repo, local_dir=path, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"])
EOF

if [ -n "$SMOKE" ]; then
  # every step on a few texts, into data_smoke/ and results/smoke_<run>/
  export SPECVOCAB_DATA=data_smoke SPECVOCAB_RESULTS=results/smoke
  LIMIT="--limit 24"; TRAIN_ARGS="--steps-per-epoch 60 --epochs 1 --eval-every 30 --ckpt-every 40"
fi

# 4. texts: the target's own greedy answers (train ~1 h on H200, evaluation ~15 min)
D=${SPECVOCAB_DATA:-data}
[ -f $D/eval_en.pt ]   || { log "eval texts";     $PY prepare_data.py eval $LIMIT >> logs/prepare_eval.log 2>&1; }
[ -f $D/eval_lang.pt ] || { log "language texts"; $PY prepare_data.py lang $LIMIT >> logs/prepare_lang.log 2>&1; }
if [ "$RUN" = ml ]; then ML=--ml; else ML=; fi
[ -f $D/train_$RUN.pt ] || { log "train texts ($RUN)"; $PY prepare_data.py train $ML $LIMIT >> logs/prepare_train_$RUN.log 2>&1; }

# 5. the draft code must reproduce the official draft before anything is trained
log "sanity check"
$PY sanity.py 2>&1 | grep -v Warning | tee -a logs/sanity.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "sanity check failed, stopping"; exit 1; }

# 6. training: 3 variants side by side; speed in logs/train_$RUN.log after a few minutes
R=${SPECVOCAB_RESULTS:-results}/$RUN
if [ ! -f $R/specvocab.pt ]; then
  log "training ($RUN)"
  $PY train.py $RUN $TRAIN_ARGS >> logs/train_$RUN.log 2>&1
fi

# 7. final evaluation on all texts, head cost, tables and figures
[ -f $R/final.json ] || { log "final evaluation"; $PY evaluate.py $RUN >> logs/evaluate_$RUN.log 2>&1; }
[ -f ${SPECVOCAB_RESULTS:-results}/head_speed.json ] || $PY head_speed.py >> logs/head_speed.log 2>&1 || true
$PY analyze.py $RUN > logs/analyze_$RUN.log 2>&1 || true

# 8. everything for the report, without the large weights
tar czf results_$RUN.tgz $R/*.json $R/*.md $R/*.png ${SPECVOCAB_RESULTS:-results}/head_speed.json logs 2>/dev/null || true
log "done: results_$RUN.tgz ($R/summary.md)"
