#!/bin/bash
# Second part of the final evaluation, unattended: wait for ours to reach the comparison step, then
#   1. ours (SGLang + the authors' code) and the plain server (speed-up denominator) on a free GPU;
#   2. if the session has more than 50 min left: official and baseline again in SGLang, for clean speeds
#      (their first runs, done while ours was training, are kept as runs/final_eval/<draft>_gpu_shared/).
#   cd ~/data/big_training && nohup setsid bash final_eval_part2.sh > final_eval_part2.out 2>&1 &
cd "$(dirname "$0")"
PY=../nemo_speed/venv_orig/bin/python
log() { echo "$(date '+%H:%M:%S') $*"; }
until [ "$($PY -c 'import json; print(json.load(open("runs/ours/status.json"))["phase"])')" = done ]; do sleep 60; done
log "ours reached the comparison step"
sleep 30                                   # its process exits and frees the GPU
$PY final_eval.py --drafts ours
END=$($PY -c "import session; print(int(session.session_end()))")
if [ $(( END - $(date +%s) )) -gt 3000 ]; then
  for d in official baseline; do
    if [ ! -d runs/final_eval/${d}_gpu_shared ]; then
      mkdir -p runs/final_eval/${d}_gpu_shared
      mv runs/final_eval/$d/sglang_* runs/final_eval/${d}_gpu_shared/
    fi
  done
  log "clean speeds of official and baseline"
  $PY final_eval.py --drafts official,baseline --no-authors --no-plain
else
  log "not enough session time for clean speeds of official and baseline"
fi
log "done"
