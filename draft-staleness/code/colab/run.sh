#!/bin/bash
# Start a job in the background on the session, e.g.:  run.sh code/staleness.py   |   run.sh code/fixfc.py nemotron_nano
# Never queue two jobs that download/delete models at the same time: they delete each other's model folders.
ssh colab-eagle "cd /content/staleness && mkdir -p results/llama31-8b && (HF_HUB_ENABLE_HF_TRANSFER=1 nohup setsid .venv/bin/python $* > results/llama31-8b/run.log 2>&1 < /dev/null &) && sleep 5 && tail -3 results/llama31-8b/run.log"
