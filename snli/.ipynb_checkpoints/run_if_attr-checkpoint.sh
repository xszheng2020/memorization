#!/bin/bash -e
mkdir -p log/score



CUDA_VISIBLE_DEVICES=1 python -u compute_if_attr.py --SEED=42 --START=0 > log/score/log_0.txt &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=2 python -u compute_if_attr.py --SEED=42 --START=1000 > log/score/log_1000.txt &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=4 python -u compute_if_attr.py --SEED=42 --START=2000 > log/score/log_2000.txt &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=10 python -u compute_if_attr.py --SEED=42 --START=3000 > log/score/log_3000.txt &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=12 python -u compute_if_attr.py --SEED=42 --START=4000 > log/score/log_4000.txt &
PIDS[4]=$!

CUDA_VISIBLE_DEVICES=1 python -u compute_if_attr.py --SEED=42 --START=5000 > log/score/log_5000.txt &
PIDS[5]=$!

CUDA_VISIBLE_DEVICES=2 python -u compute_if_attr.py --SEED=42 --START=6000 > log/score/log_6000.txt &
PIDS[6]=$!

CUDA_VISIBLE_DEVICES=4 python -u compute_if_attr.py --SEED=42 --START=7000 > log/score/log_7000.txt &
PIDS[7]=$!

CUDA_VISIBLE_DEVICES=10 python -u compute_if_attr.py --SEED=42 --START=8000 > log/score/log_8000.txt &
PIDS[8]=$!

CUDA_VISIBLE_DEVICES=12 python -u compute_if_attr.py --SEED=42 --START=9000 > log/score/log_9000.txt &
PIDS[9]=$!



trap "kill ${PIDS[*]}" SIGINT

wait
