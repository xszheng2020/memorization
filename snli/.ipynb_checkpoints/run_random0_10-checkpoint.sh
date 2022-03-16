#!/bin/bash -e
mkdir -p log/random_0/10

CUDA_VISIBLE_DEVICES=1 python -u train.py --ORDER='random_0' --PERCENTAGE=10 --SEED=0 > log/random_0/10/log_seed_0.txt &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=2 python -u train.py --ORDER='random_0' --PERCENTAGE=10 --SEED=1 > log/random_0/10/log_seed_1.txt &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=4 python -u train.py --ORDER='random_0' --PERCENTAGE=10 --SEED=2 > log/random_0/10/log_seed_2.txt &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=11 python -u train.py --ORDER='random_0' --PERCENTAGE=10 --SEED=3 > log/random_0/10/log_seed_3.txt &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=15 python -u train.py --ORDER='random_0' --PERCENTAGE=10 --SEED=42 > log/random_0/10/log_seed_42.txt &
PIDS[4]=$!

trap "kill ${PIDS[*]}" SIGINT

wait
