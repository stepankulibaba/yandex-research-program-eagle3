#!/bin/bash
# Copy draft-staleness/ (code, prompts; not results/models) to /content/staleness on the session and run setup.sh
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
tar czf - -C "$ROOT" --exclude=__pycache__ --exclude=results --exclude=models --exclude=figures --exclude=reports \
  --exclude=.venv --exclude=EAGLE . | ssh colab-eagle 'mkdir -p /content/staleness && tar xzf - -C /content/staleness'
ssh colab-eagle 'cd /content/staleness && bash code/setup.sh 2>&1 | tail -5'
