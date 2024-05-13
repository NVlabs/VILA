#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file ./eval_output/$CKPT/mmbench_cn/$SPLIT.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir ./eval_output/$CKPT/mmbench_cn \
    --upload-dir ./eval_output/$CKPT/mmbench_cn \
    --experiment $SPLIT
