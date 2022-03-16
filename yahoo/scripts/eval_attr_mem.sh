#!/bin/bash -e
# +
mkdir -p log/eval_attr/mem/10

CUDA_VISIBLE_DEVICES=1 python -u compute_if.py --ATTR_ORDER='mem' --ATTR_PERCENTAGE=10 --START=0 > log/eval_attr/mem/10/log_0.txt &
PIDS[0]=$!
# +
mkdir -p log/eval_attr/mem/20

CUDA_VISIBLE_DEVICES=2 python -u compute_if.py --ATTR_ORDER='mem' --ATTR_PERCENTAGE=20 --START=0 > log/eval_attr/mem/20/log_0.txt &
PIDS[1]=$!

# +
mkdir -p log/eval_attr/mem/30

CUDA_VISIBLE_DEVICES=10 python -u compute_if.py --ATTR_ORDER='mem' --ATTR_PERCENTAGE=30 --START=0 > log/eval_attr/mem/30/log_0.txt &
PIDS[2]=$!
# +
mkdir -p log/eval_attr/mem/40

CUDA_VISIBLE_DEVICES=4 python -u compute_if.py --ATTR_ORDER='mem' --ATTR_PERCENTAGE=40 --START=0 > log/eval_attr/mem/40/log_0.txt &
PIDS[3]=$!


# +
mkdir -p log/eval_attr/mem/50

CUDA_VISIBLE_DEVICES=5 python -u compute_if.py --ATTR_ORDER='mem' --ATTR_PERCENTAGE=50 --START=0 > log/eval_attr/mem/50/log_0.txt &
PIDS[4]=$!


# +
mkdir -p log/eval_attr/mem/60

CUDA_VISIBLE_DEVICES=6 python -u compute_if.py --ATTR_ORDER='mem' --ATTR_PERCENTAGE=60 --START=0 > log/eval_attr/mem/60/log_0.txt &
PIDS[5]=$!


# +
mkdir -p log/eval_attr/mem/70

CUDA_VISIBLE_DEVICES=7 python -u compute_if.py --ATTR_ORDER='mem' --ATTR_PERCENTAGE=70 --START=0 > log/eval_attr/mem/70/log_0.txt &
PIDS[6]=$!

# +
mkdir -p log/eval_attr/mem/80

CUDA_VISIBLE_DEVICES=8 python -u compute_if.py --ATTR_ORDER='mem' --ATTR_PERCENTAGE=80 --START=0 > log/eval_attr/mem/80/log_0.txt &
PIDS[7]=$!

# +
mkdir -p log/eval_attr/mem/90

CUDA_VISIBLE_DEVICES=9 python -u compute_if.py --ATTR_ORDER='mem' --ATTR_PERCENTAGE=90 --START=0 > log/eval_attr/mem/90/log_0.txt &
PIDS[8]=$!
# -



trap "kill ${PIDS[*]}" SIGINT

wait
