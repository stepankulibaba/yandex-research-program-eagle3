#!/usr/bin/env bash
# One GPU session (run it in every new 8-hour SLURM session; it resumes by itself):
#   nohup setsid bash session.sh > session.out 2>&1 &
#   nohup setsid bash session.sh --smoke > smoke.out 2>&1 &      (the first check, ~30 min)
set -Eeuo pipefail
cd "$(dirname "$0")"
exec ../nemo_speed/venv_orig/bin/python session.py "$@"
