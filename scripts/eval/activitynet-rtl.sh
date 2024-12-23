#!/bin/bash

MODEL_PATH=$1
OUTPUT_DIR=$2

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/activitynet-rtl"}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 1024}'

torchrun --nproc-per-node=$NPROC_PER_NODE \
    -m llava.eval.rtl \
    --dataset activitynet-rtl/val \
    --model-path $MODEL_PATH \
    --generation-config "$GENERATION_CONFIG" \
    --output-dir $OUTPUT_DIR
