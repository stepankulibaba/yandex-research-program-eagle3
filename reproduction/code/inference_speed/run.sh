#!/usr/bin/env bash
# Inference only: 3a, 3b and the regeneration probe with the budget (task 5).
#   NIGHT_SECONDS=26400 nohup setsid bash run.sh > run.out 2>&1 &
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/orchestrate.py" inference "$@"
