#!/usr/bin/env bash
# Short GPU check of everything on 2 questions / 40 training examples (~1 h), results in results/smoke.
#   bash check.sh
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/orchestrate.py" check "$@"
