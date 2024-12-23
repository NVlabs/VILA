#!/bin/bash

SPLIT=$1
MODEL_PATH=$2
CONV_MODE=$3
NUM_VIDEO_FRAMES=$4

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/vnbench_$SPLIT"}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 32}'

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/vnbench.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --num-video-frames $NUM_VIDEO_FRAMES \
    --generation-config "$GENERATION_CONFIG" \
    --output-dir $OUTPUT_DIR
