#!/usr/bin/env bash
# Training speed only (task 4); the same as ../inference_speed/nemo_run.sh.
#   bash run.sh        (SMOKE=1 for 40 examples)
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/../inference_speed/orchestrate.py" train "$@"
