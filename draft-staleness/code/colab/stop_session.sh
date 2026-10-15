#!/bin/bash
# Stop only OUR session "eagle" (other sessions on the account may belong to colleagues)
export PATH="$HOME/.local/bin:$PATH"
colab sessions | grep -q '^\[eagle\]' && colab stop -s eagle
colab sessions; colab usage | tail -3
