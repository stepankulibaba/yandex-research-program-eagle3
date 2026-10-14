#!/usr/bin/env bash
# The first session's check (~30-40 min of GPU): every piece once, on 4000 dialogues, before the real run.
#   nohup setsid bash smoke.sh > smoke.out 2>&1 &        then: tail -40 smoke.out
set -Eeuo pipefail
cd "$(dirname "$0")"
PY=../nemo_speed/venv_orig/bin/python
EVAL_PY=../inference_speed/venv_eagle/bin/python
command -v nvcc >/dev/null || export CUDA_HOME=../nemo_speed/venv_orig/stub_cuda   # DeepSpeed asks for nvcc -V

echo "== 1. tiny GPU check of the patched code"
"$PY" test_tiny_gpu.py
echo "== 2. both variants side by side, 150 steps; baseline is interrupted once and resumed"
"$PY" session.py --smoke
echo "== 3. the official draft on the smoke validation"
"$PY" reference_eval.py --data data_smoke --runs runs_smoke --val-eval 50
echo "== 4. real tau (the authors' eval, 2 MT-bench questions): smoke drafts and the official draft"
for v in baseline ours; do
    "$EVAL_PY" eval_tau.py runs_smoke/$v/milestones/step0000150 smoke_$v --questions 2
done
"$EVAL_PY" eval_tau.py ../inference_speed/models/eagle3-llama31-8b smoke_official --questions 2
echo "== 5. report"
"$PY" smoke_report.py
