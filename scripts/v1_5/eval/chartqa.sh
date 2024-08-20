#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

torchrun --nproc-per-node=8 \
    llava/eval/evaluate_vqa.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config '{"max_new_tokens": 100}' \
    --dataset chartqa_test_human \
    --image-folder ./playground/data/eval/chartqa \
    --data-path ./playground/data/eval/chartqa/test_human.jsonl \
    --answers-file runs/eval/$CKPT/chartqa/answers1/merge.jsonl

python -m llava.eval.evaluate_vqa_score --answers-file runs/eval/$CKPT/chartqa/answers1/merge.jsonl --metric relaxed_accuracy

torchrun --nproc-per-node=8 \
    llava/eval/evaluate_vqa.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config '{"max_new_tokens": 100}' \
    --dataset chartqa_test_augmented \
    --data-path ./playground/data/eval/chartqa/test_augmented.jsonl \
    --image-folder ./playground/data/eval/chartqa \
    --answers-file runs/eval/$CKPT/chartqa/answers2/merge.jsonl

python -m llava.eval.evaluate_vqa_score --answers-file runs/eval/$CKPT/chartqa/answers2/merge.jsonl --metric relaxed_accuracy
