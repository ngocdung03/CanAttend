#!/bin/bash
if [ $# -ne 1 ]; then
	echo "usage: sh $0 <imputation method>"
	exit 1
fi

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
MODEL="deephit"
IMPU=$1

for c in $CANCERS
do
	python -u run_surv.py --dataset $DATA \
		--feature $FEATURE \
		--cancer $CANCER \
		--n_trials $N_NUM \
		--model $MODEL \
		--by_sex true \
		--imputation $IMPU\
		--batch_size 1024 \
		--new True \
		--target_cancer $c >> outputs/batch_deephit.log
done

exit 0
