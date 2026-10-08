#!/bin/bash
# Exactly the command running on the H200 (Qwen3-1.7B, 7 variants, 3 epochs x 8000 sequences).
cd "$(dirname "$0")"
mkdir -p logs
.venv/bin/python routed.py run qwen3-1.7b --variants fixed_raw,fixed_norm,fixed_alt,static_top3,static_top2,token_top3,token_top2 --steps-per-epoch 8000 --epochs 3 --eval-every 1000 > logs/routed_qwen3-1.7b.log 2>&1
echo "$(date +%T) END routed qwen3-1.7b exit=$?" >> logs/queue.log
