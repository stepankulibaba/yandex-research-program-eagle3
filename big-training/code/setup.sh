#!/usr/bin/env bash
# One-time preparation, no GPU needed: ClearML client, patched copies of the authors' code, data.
#   bash setup.sh smoke     only the 4000-dialogue smoke data (minutes)
#   bash setup.sh           also the full data: 1.42M dialogues tokenized (~1 h on 32 CPU cores, ~10 GB disk)
set -Eeuo pipefail
cd "$(dirname "$0")"
PY=../nemo_speed/venv_orig/bin/python
"$PY" -m pip install -q 'clearml>=1.16,<2'
"$PY" patch_eagle.py
[ -d data_smoke/train ] || "$PY" prepare_data.py --limit 4000
if [ "${1:-}" != smoke ]; then
    [ -d data/train ] || "$PY" prepare_data.py
fi
echo "setup done"
