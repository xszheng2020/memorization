#!/bin/bash -e
mkdir -p log/score_0



CUDA_VISIBLE_DEVICES=9 python -u compute_if_attr.py --CHECKPOINT=0 --START=0 > log/score_0/log_0.txt &
PIDS[0]=$!

CUDA_VISIBLE_DEVICES=10 python -u compute_if_attr.py --CHECKPOINT=0 --START=1000 > log/score_0/log_1000.txt &
PIDS[1]=$!

CUDA_VISIBLE_DEVICES=11 python -u compute_if_attr.py --CHECKPOINT=0 --START=2000 > log/score_0/log_2000.txt &
PIDS[2]=$!

CUDA_VISIBLE_DEVICES=12 python -u compute_if_attr.py --CHECKPOINT=0 --START=3000 > log/score_0/log_3000.txt &
PIDS[3]=$!

CUDA_VISIBLE_DEVICES=14 python -u compute_if_attr.py --CHECKPOINT=0 --START=4000 > log/score_0/log_4000.txt &
PIDS[4]=$!

CUDA_VISIBLE_DEVICES=15 python -u compute_if_attr.py --CHECKPOINT=0 --START=5000 > log/score_0/log_5000.txt &
PIDS[5]=$!

CUDA_VISIBLE_DEVICES=4 python -u compute_if_attr.py --CHECKPOINT=0 --START=6000 > log/score_0/log_6000.txt &
PIDS[6]=$!



trap "kill ${PIDS[*]}" SIGINT

wait
