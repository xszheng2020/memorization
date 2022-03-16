#!/bin/bash -e
mkdir -p log/eval/mem/0

CUDA_VISIBLE_DEVICES=12 python -u evaluate.py --ORDER='mem' --SEED=42 > log/eval/mem/0/log_seed_42.txt &
PIDS[0]=$!

trap "kill ${PIDS[*]}" SIGINT

wait
