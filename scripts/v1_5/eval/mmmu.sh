#!/bin/bash
MODEL_PATH=$1
CKPT=$2
SPLIT=$3
CONV_MODE=vicuna_v1
if [ "$#" -ge 4 ]; then
    CONV_MODE="$4"
fi

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_mmmu \
    --model_path $MODEL_PATH \
    --data_path ./playground/data/eval/MMMU \
    --conv-mode $CONV_MODE \
    --config_path llava/eval/mmmu_utils/configs/llava1.5.yaml \
    --output_path ./eval_output/$CKPT/MMMU/${SPLIT}_answers.json \
    --split $SPLIT

if [ "$SPLIT" = "validation" ]; then
    python llava/eval/eval_mmmu.py --output_path ./eval_output/$CKPT/MMMU/${SPLIT}_answers.json --answer_path ./playground/data/eval/MMMU/answer_dict_val.json
fi