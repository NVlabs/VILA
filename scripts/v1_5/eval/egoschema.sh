#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

DATA_PATH="playground/data/eval/EgoSchema/questions.json"
VIDEO_DIR="playground/data/eval/EgoSchema/videos"
ANSWER_PATH="playground/data/eval/EgoSchema/subset_answers.json"
OUTPUT_DIR="runs/eval/$CKPT/egoschema/validation"
GENERATION_CONFIG='{"max_new_tokens": 1024}'

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_ego_schema.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --question-file $DATA_PATH \
    --video-folder $VIDEO_DIR \
    --gt-answers-file $ANSWER_PATH \
    --split validation \
    --output_dir $OUTPUT_DIR \
    --output_name outputs
