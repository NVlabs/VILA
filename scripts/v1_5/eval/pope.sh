#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./eval_output/$CKPT/pope/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./eval_output/$CKPT/pope/answers.jsonl
