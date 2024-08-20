#!/bin/bash

MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_loader.py \
    --model-path $MODEL_PATH \
    --generation-config '{"max_new_tokens": 128}' \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file runs/eval/$CKPT/vizwiz/answers.jsonl \
    --conv-mode $CONV_MODE

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file runs/eval/$CKPT/vizwiz/answers.jsonl \
    --result-upload-file runs/eval/$CKPT/vizwiz/answers_upload.json
