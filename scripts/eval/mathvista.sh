#!/bin/bash

SPLIT=$1
MODEL_PATH=$2
CONV_MODE=$3
MAX_TILES=$4

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/mathvista-$SPLIT"}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/mathvista.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --max-tiles $MAX_TILES \
    --generation-config "$GENERATION_CONFIG" \
    --split $SPLIT \
    --output-dir $OUTPUT_DIR
