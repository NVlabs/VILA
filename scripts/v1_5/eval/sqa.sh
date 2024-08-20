#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

DATA_ROOT="playground/data/eval/scienceqa"
DATA_PATH="$DATA_ROOT/llava_test_CQM-A.json"
IMAGE_DIR="$DATA_ROOT/images/test"
OUTPUT_PATH="runs/eval/$CKPT/scienceqa/outputs.jsonl"
RESULT_PATH="runs/eval/$CKPT/scienceqa/results.json"
GENERATION_CONFIG='{"max_new_tokens": 1024}'

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_science.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --question-file $DATA_PATH \
    --image-folder $IMAGE_DIR \
    --answers-file $OUTPUT_PATH \
    --single-pred-prompt

python llava/eval/eval_science_qa.py \
    --base-dir $DATA_ROOT \
    --result-file $OUTPUT_PATH \
    --output-file $OUTPUT_PATH \
    --output-result $RESULT_PATH
