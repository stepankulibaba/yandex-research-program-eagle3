#!/usr/bin/env bash
# Training speed only (task 4): the authors' trainer and NeMo with its speed-ups.
#   bash nemo_run.sh        (SMOKE=1 for 40 examples)
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/orchestrate.py" train "$@"
