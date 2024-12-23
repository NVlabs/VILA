#!/bin/bash

MODEL_PATH=$1
CONV_MODE=$2

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/cinepile"}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 1024}'

VIDEO_DIR="/home/xiuli/workspace/cinepile/yt_videos"

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/cinepile.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --video-dir $VIDEO_DIR \
    --output-dir $OUTPUT_DIR
