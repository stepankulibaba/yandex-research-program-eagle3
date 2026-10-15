#!/bin/bash
# Keep the session's kernel active (Colab drops idle runtimes; work over ssh does not count) and copy new result files
# to results/llama31-8b/ every 5 minutes. Never creates a runtime: if "eagle" is gone it stops
# (colab exec / colab ssh would otherwise silently auto-create a CPU runtime under the same name).
# usage: keepalive_pull.sh <result file whose arrival ends the job>
export PATH="$HOME/.local/bin:$PATH"
DONE_FILE=$1
DST="$(cd "$(dirname "$0")/../.." && pwd)/results/llama31-8b"
echo 'import time; print("alive", time.time())' > /tmp/ka.py
for i in $(seq 1 400); do
  colab sessions 2>/dev/null | grep '^\[eagle\]' | grep -q 'A100\|H100' || { echo "$(date +%T) session gone"; break; }
  timeout 90 colab exec -s eagle -f /tmp/ka.py --timeout 30 > /dev/null 2>&1 || echo "$(date +%T) keepalive failed"
  listing=$(ssh colab-eagle 'cd /content/staleness/results/llama31-8b && stat -c "%n %s" *.json *.npz 2>/dev/null' 2>/dev/null)
  new=""
  while read -r f s; do [ -n "$f" ] && [ "$(stat -c %s "$DST/$f" 2>/dev/null)" != "$s" ] && new="$new $f"; done <<< "$listing"
  [ -n "$new" ] && ssh colab-eagle "cd /content/staleness/results/llama31-8b && tar cf - $new" | tar xf - -C "$DST" && echo "$(date +%T) pulled$new"
  ssh colab-eagle 'cat /content/staleness/results/llama31-8b/run.log' > "$DST/run_latest.log" 2>/dev/null
  state=$(ssh colab-eagle 'pgrep -f -E "[p]ython code/" >/dev/null && echo RUNNING || echo IDLE' 2>/dev/null)
  [ "$state" = "IDLE" ] && [ -f "$DST/$DONE_FILE" ] && { echo "$(date +%T) job finished"; break; }
  sleep 300
done
