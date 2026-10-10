#!/usr/bin/env bash
# The full run: 3a, 4, 5, 3b and the NeMo speed-ups, with a common deadline (here 7 h 20 min).
#   NIGHT_SECONDS=26400 nohup setsid bash night.sh > night.out 2>&1 &
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/orchestrate.py" night "$@"
