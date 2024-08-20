#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

DATA_PATH="playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_DIR="playground/data/eval/textvqa/train_images"
ANNOTATION_PATH="playground/data/eval/textvqa/TextVQA_0.5.1_val.json"
OUTPUT_PATH="runs/eval/$CKPT/textvqa/outputs.jsonl"
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_loader.py \
    --model-path $MODEL_PATH \
    --generation-config "$GENERATION_CONFIG" \
    --question-file $DATA_PATH \
    --image-folder $IMAGE_DIR \
    --answers-file $OUTPUT_PATH \
    --conv-mode $CONV_MODE

python llava/eval/eval_textvqa.py \
    --annotation-file $ANNOTATION_PATH \
    --result-file $OUTPUT_PATH
