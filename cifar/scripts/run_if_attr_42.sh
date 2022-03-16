#!/bin/bash -e
mkdir -p log/score_42



CUDA_VISIBLE_DEVICES=1 python -u compute_if_attr.py --CHECKPOINT=42 --START=0 > log/score_42/log_0.txt &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=2 python -u compute_if_attr.py --CHECKPOINT=42 --START=1000 > log/score_42/log_1000.txt &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=3 python -u compute_if_attr.py --CHECKPOINT=42 --START=2000 > log/score_42/log_2000.txt &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=4 python -u compute_if_attr.py --CHECKPOINT=42 --START=3000 > log/score_42/log_3000.txt &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=5 python -u compute_if_attr.py --CHECKPOINT=42 --START=4000 > log/score_42/log_4000.txt &
PIDS[4]=$!

CUDA_VISIBLE_DEVICES=6 python -u compute_if_attr.py --CHECKPOINT=42 --START=5000 > log/score_42/log_5000.txt &
PIDS[5]=$!

CUDA_VISIBLE_DEVICES=7 python -u compute_if_attr.py --CHECKPOINT=42 --START=6000 > log/score_42/log_6000.txt &
PIDS[6]=$!

CUDA_VISIBLE_DEVICES=8 python -u compute_if_attr.py --CHECKPOINT=42 --START=7000 > log/score_42/log_7000.txt &
PIDS[7]=$!

CUDA_VISIBLE_DEVICES=9 python -u compute_if_attr.py --CHECKPOINT=42 --START=8000 > log/score_42/log_8000.txt &
PIDS[8]=$!

CUDA_VISIBLE_DEVICES=10 python -u compute_if_attr.py --CHECKPOINT=42 --START=9000 > log/score_42/log_9000.txt &
PIDS[9]=$!



trap "kill ${PIDS[*]}" SIGINT

wait
