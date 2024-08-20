#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

SPLIT="llava_vqav2_mscoco_test-dev2015"
OUTPUT_DIR=runs/eval/$CKPT/vqav2/$SPLIT/answers/merge.jsonl
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_loader.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
    --image-folder ./playground/data/eval/vqav2/test2015 \
    --answers-file $OUTPUT_DIR

cp ./playground/data/eval/vqav2/llava_vqav2_mscoco_test2015.jsonl runs/eval/$CKPT/vqav2/

python scripts/convert_vqav2_for_submission.py --dir runs/eval/$CKPT/vqav2/ --split $SPLIT
