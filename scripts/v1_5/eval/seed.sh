#!/bin/bash

MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

DATA_PATH="playground/data/eval/seed_bench/llava-seed-bench.jsonl"
IMAGE_DIR="playground/data/eval/seed_bench"
ANNOTATION_PATH="playground/data/eval/seed_bench/SEED-Bench.json"
OUTPUT_PATH="runs/eval/$CKPT/seed/outputs.jsonl"
UPLOAD_PATH="runs/eval/$CKPT/seed/upload.jsonl"
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_loader.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --question-file $DATA_PATH \
    --image-folder $IMAGE_DIR \
    --answers-file $OUTPUT_PATH \

python scripts/convert_seed_for_submission.py \
    --annotation-file $ANNOTATION_PATH \
    --result-file $OUTPUT_PATH \
    --result-upload-file $UPLOAD_PATH
