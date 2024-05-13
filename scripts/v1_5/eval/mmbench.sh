#!/bin/bash

SPLIT="mmbench_dev_20230712"
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./eval_output/$CKPT/mmbench/$SPLIT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./eval_output/$CKPT/mmbench \
    --upload-dir ./eval_output/$CKPT/mmbench \
    --experiment $SPLIT
