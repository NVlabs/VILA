#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

echo "$MODEL_PATH $CKPT"

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./eval_output/$CKPT/llava-bench-in-the-wild/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        ./eval_output/$CKPT/llava-bench-in-the-wild/answers.jsonl \
    --output \
        ./eval_output/$CKPT/llava-bench-in-the-wild/reviews.jsonl

python llava/eval/summarize_gpt_review.py -f ./eval_output/$CKPT/llava-bench-in-the-wild/reviews.jsonl
