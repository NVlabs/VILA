#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

SPLIT="llava_gqa_testdev_balanced"
GQADIR="playground/data/eval/gqa"
OUTPUT_PATH=runs/eval/$CKPT/gqa/$SPLIT/answers/merge.jsonl
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_loader.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
    --image-folder ./playground/data/eval/gqa/images \
    --answers-file $OUTPUT_PATH

python scripts/convert_gqa_for_eval.py \
    --src $OUTPUT_PATH \
    --dst runs/eval/$CKPT/gqa/$SPLIT/testdev_balanced_predictions.json

python $GQADIR/eval.py \
    --tier $GQADIR/testdev_balanced \
    --predictions runs/eval/$CKPT/gqa/$SPLIT/testdev_balanced_predictions.json
