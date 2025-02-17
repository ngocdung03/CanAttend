#!/bin/bash

if [ $# -ne 1 ]; then
	echo "usage: sh $0 <imputation method>"
	exit 1
fi

conda activate torch

CANCERS="
THROI
STOMA
BREAC
CRC
LUNG
"

DATA="dataset/ba8.csv"
FEATURE="dataset/features.txt"
CANCER="dataset/cancers_new.txt"
N_NUM=30
MODEL="pl_mtl_lstm"
IMPU=$1

python -u run_surv.py --dataset $DATA \
		--feature $FEATURE \
		--cancer $CANCER \
		--n_trials $N_NUM \
		--model $MODEL \
		--by_sex true \
		--imputation $IMPU \
        --n_epochs 2 \
		--new True \
		--tasks THROI STOMA >> outputs/batch_mtl_lstm_$IMPU.log

python -u run_surv.py --dataset $DATA \
		--feature $FEATURE \
		--cancer $CANCER \
		--n_trials $N_NUM \
		--model $MODEL \
		--by_sex true \
		--imputation $IMPU \
        --n_epochs 2 \
		--new True \
		--tasks BREAC CRC LUNG >> outputs/batch_mtl_lstm_$IMPU.log

exit 0
