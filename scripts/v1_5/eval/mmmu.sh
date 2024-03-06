#!/bin/bash
MODEL_PATH=$1
CKPT=$2
SPLIT=$3

mkdir -p ./playground/data/eval/MMMU/test_results

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_mmmu \
    --model_path $MODEL_PATH \
    --data_path ./playground/data/eval/MMMU \
    --conv-mode vicuna_v1 \
    --config_path llava/eval/mmmu_utils/configs/llava1.5.yaml \
    --output_path ./playground/data/eval/MMMU/test_results/$CKPT.json \
    --split $SPLIT

