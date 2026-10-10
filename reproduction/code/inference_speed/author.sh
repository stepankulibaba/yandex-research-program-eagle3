#!/usr/bin/env bash
# The authors' EAGLE-3 code on this machine (e.g. an A100): profile first (~15 min), then 3a (T=0, then T=1).
#   NIGHT_SECONDS=36000 nohup setsid bash author.sh > author.out 2>&1 &
#   AUTHOR_TEMPERATURES=0 ...        only T=0
# Results: results/main/profile/ (summary.md, figures/), results/main/summary.md (3a vs the paper).
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/orchestrate.py" author "$@"
