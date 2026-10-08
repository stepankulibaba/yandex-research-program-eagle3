#!/bin/bash
# Router experiment on DSL-8B from scratch: environment -> models -> data -> sanity check -> training -> final eval.
# Usage:  bash run_dsl.sh            (full: 3 epochs x 8000 sequences)
#         EPOCHS=2 bash run_dsl.sh   (shorter)
# Every step is skipped if its output already exists, so the script can be restarted after a failure.
set -e
cd "$(dirname "$0")"
PAIR=dsl-8b
EPOCHS=${EPOCHS:-3}
STEPS=${STEPS_PER_EPOCH:-8000}
mkdir -p logs
log() { echo "$(date '+%F %T') $*" | tee -a logs/run_dsl.log; }

# 1. environment (Python 3.11+, CUDA 12.x driver)
if [ ! -x .venv/bin/python ]; then
  log "creating .venv"
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. benchmark questions (MT-bench, GSM8K, HumanEval) from the EAGLE repository, pinned commit
if [ ! -d EAGLE ]; then
  git clone -q https://github.com/SafeAILab/EAGLE EAGLE
  (cd EAGLE && git checkout -q cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b)
fi

# 3. models (~17 GB)
$PY - <<'EOF'
from pathlib import Path
from huggingface_hub import snapshot_download
for repo, path in [("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "models/dsl-8b/target"),
                   ("yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B", "models/dsl-8b/draft")]:
    if not (Path(path) / "config.json").exists():
        print("downloading", repo, flush=True)
        snapshot_download(repo, local_dir=path, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"])
EOF

# 4. data: target's own greedy answers (train ~1-1.5 h on H200, eval ~10 min)
[ -f data/$PAIR/eval.pt ]  || { log "eval texts";  $PY prepare_data.py eval  $PAIR >> logs/prepare_eval.log 2>&1; }
[ -f data/$PAIR/train.pt ] || { log "train texts"; $PY prepare_data.py train $PAIR >> logs/prepare_train.log 2>&1; }

# 5. the draft code must reproduce the official draft before we train anything
log "sanity check"
$PY sanity.py $PAIR 2>&1 | grep -v Warning | tee -a logs/sanity.log
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "sanity check failed, stopping"; exit 1; }

# 6. training: 7 variants side by side; check speed in logs/train.log after a few minutes
if [ ! -f results/$PAIR/token_top2.pt ]; then
  log "training ($EPOCHS x $STEPS)"
  $PY train.py $PAIR --epochs $EPOCHS --steps-per-epoch $STEPS --eval-every 1000 >> logs/train.log 2>&1
fi

# 7. final evaluation on all 240 questions (+ per-token routers forced to their most frequent set)
[ -f results/$PAIR/final_eval.pt ] || { log "final eval"; $PY final_eval.py $PAIR >> logs/final_eval.log 2>&1; }
$PY analyze.py results/$PAIR/history.json results/$PAIR/figures > logs/analyze.txt 2>&1 || true

# 8. everything needed for the report, without the large checkpoints
tar czf results_$PAIR.tgz results/$PAIR/history.json results/$PAIR/final_eval.pt results/$PAIR/figures logs
log "done: results_$PAIR.tgz"
