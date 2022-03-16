#!/bin/bash -e
mkdir -p log/random/0

CUDA_VISIBLE_DEVICES=7 python -u train.py --SEED=0 --SAVE_CHECKPOINT=True > log/random/0/log_seed_0.txt &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=8 python -u train.py --SEED=1 --SAVE_CHECKPOINT=True > log/random/0/log_seed_1.txt &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=10 python -u train.py --SEED=2 --SAVE_CHECKPOINT=True > log/random/0/log_seed_2.txt &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=14 python -u train.py --SEED=3 --SAVE_CHECKPOINT=True > log/random/0/log_seed_3.txt &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=15 python -u train.py --SEED=42 --SAVE_CHECKPOINT=True > log/random/0/log_seed_42.txt &
PIDS[4]=$!

trap "kill ${PIDS[*]}" SIGINT

wait
