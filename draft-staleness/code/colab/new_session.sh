#!/bin/bash
# GPU session "eagle": H100 if the account gets one, otherwise A100 High-RAM (~6.8 compute units / hour)
export PATH="$HOME/.local/bin:$PATH"
timeout 300 colab new -s eagle --gpu H100 --high-mem < /dev/null 2>&1 | tail -2
colab sessions 2>&1 | grep -q '^\[eagle\]' || timeout 300 colab new -s eagle --gpu A100 --high-mem < /dev/null 2>&1 | tail -2
colab sessions; colab usage | tail -3
