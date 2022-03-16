#!/bin/bash -e
mkdir -p log/eval/random/0

CUDA_VISIBLE_DEVICES=6 python -u evaluate.py --ORDER='random' --SEED=42 > log/eval/random/0/log_seed_42.txt &
PIDS[0]=$!

trap "kill ${PIDS[*]}" SIGINT

wait
