#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

DATA_PATH="playground/data/eval/ai2d/test.jsonl"
IMAGE_DIR="playground/data/eval/ai2d"
OUTPUT_PATH="runs/eval/$CKPT/ai2d/outputs.jsonl"
GENERATION_CONFIG='{"max_new_tokens": 10}'

torchrun --nproc-per-node=8 \
    llava/eval/evaluate_vqa.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --dataset ai2diagram_test \
    --data-path $DATA_PATH \
    --image-folder $IMAGE_DIR \
    --answers-file $OUTPUT_PATH

python llava/eval/evaluate_vqa_score.py \
    --answers-file $OUTPUT_PATH \
    --metric accuracy
