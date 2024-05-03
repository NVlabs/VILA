#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./eval_output/$CKPT/mm-vet/answers.json \
    --temperature 0 \
    --conv-mode $CONV_MODE

python scripts/convert_mmvet_for_eval.py \
    --src ./eval_output/$CKPT/mm-vet/answers.json \
    --dst ./eval_output/$CKPT/mm-vet/results.json

python llava/eval/eval_mmvet.py --results_file ./eval_output/$CKPT/mm-vet/results.json
