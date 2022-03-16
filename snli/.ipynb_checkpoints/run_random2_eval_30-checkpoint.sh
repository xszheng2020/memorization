#!/bin/bash -e
mkdir -p log/eval/random_2/30

CUDA_VISIBLE_DEVICES=3 python -u evaluate.py --ORDER='random_2' --PERCENTAGE=30 --SEED=0 > log/eval/random_2/30/log_seed_0.txt &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=7 python -u evaluate.py --ORDER='random_2' --PERCENTAGE=30 --SEED=1 > log/eval/random_2/30/log_seed_1.txt &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=8 python -u evaluate.py --ORDER='random_2' --PERCENTAGE=30 --SEED=2 > log/eval/random_2/30/log_seed_2.txt &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=14 python -u evaluate.py --ORDER='random_2' --PERCENTAGE=30 --SEED=3 > log/eval/random_2/30/log_seed_3.txt &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=15 python -u evaluate.py --ORDER='random_2' --PERCENTAGE=30 --SEED=42 > log/eval/random_2/30/log_seed_42.txt &
PIDS[4]=$!



trap "kill ${PIDS[*]}" SIGINT

wait
