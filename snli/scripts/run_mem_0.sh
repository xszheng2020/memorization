#!/bin/bash -e
mkdir -p log/mem/0

CUDA_VISIBLE_DEVICES=3 python -u train.py --ORDER='mem' --SEED=0 > log/mem/0/log_seed_0.txt &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=4 python -u train.py --ORDER='mem' --SEED=1 > log/mem/0/log_seed_1.txt &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=8 python -u train.py --ORDER='mem' --SEED=2 > log/mem/0/log_seed_2.txt &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=9 python -u train.py --ORDER='mem' --SEED=3 > log/mem/0/log_seed_3.txt &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=10 python -u train.py --ORDER='mem' --SEED=42 > log/mem/0/log_seed_42.txt &
PIDS[4]=$!

trap "kill ${PIDS[*]}" SIGINT

wait
