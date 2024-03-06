#!/bin/bash
MODEL_PATH=$1
CKPT=$2

mkdir -p ./playground/data/eval/MME/answers/$CKPT

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$CKPT/mme.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment answers/$CKPT/mme.jsonl

cd eval_tool

python calculation.py --results_dir ../answers/$CKPT/mme_results

