#!/bin/bash -e
mkdir -p log/random/30

CUDA_VISIBLE_DEVICES=7 python -u train.py --ORDER='random' --PERCENTAGE=30 --SEED=0 > log/random/30/log_seed_0.txt &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=8 python -u train.py --ORDER='random' --PERCENTAGE=30 --SEED=1 > log/random/30/log_seed_1.txt &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=10 python -u train.py --ORDER='random' --PERCENTAGE=30 --SEED=2 > log/random/30/log_seed_2.txt &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=12 python -u train.py --ORDER='random' --PERCENTAGE=30 --SEED=3 > log/random/30/log_seed_3.txt &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=13 python -u train.py --ORDER='random' --PERCENTAGE=30 --SEED=42 > log/random/30/log_seed_42.txt &
PIDS[4]=$!

trap "kill ${PIDS[*]}" SIGINT

wait
