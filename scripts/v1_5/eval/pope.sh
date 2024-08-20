#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

DATA_PATH="playground/data/eval/pope/llava_pope_test.jsonl"
IMAGE_DIR="playground/data/eval/pope/val2014"
ANNOTATION_DIR="playground/data/eval/pope/coco"
OUTPUT_PATH="runs/eval/$CKPT/pope/outputs.jsonl"
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_loader.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --question-file $DATA_PATH \
    --image-folder $IMAGE_DIR \
    --answers-file $OUTPUT_PATH

python llava/eval/eval_pope.py \
    --annotation-dir $ANNOTATION_DIR \
    --question-file $DATA_PATH \
    --result-file $OUTPUT_PATH
