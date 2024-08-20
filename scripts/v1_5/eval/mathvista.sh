#!/bin/bash
MODEL_PATH=$1
CKPT=$2
SPLIT=$3
CONV_MODE=vicuna_v1
if [ "$#" -ge 4 ]; then
    CONV_MODE="$4"
fi

OUTPUT_PATH="runs/eval/$CKPT/mathvista/$SPLIT/outputs.json"
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_mathvista.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --split $SPLIT \
    --answers-file $OUTPUT_PATH

if [ "$SPLIT" = "testmini" ]; then
    python llava/eval/eval_mathvista.py \
        --answer_file $OUTPUT_PATH
fi
