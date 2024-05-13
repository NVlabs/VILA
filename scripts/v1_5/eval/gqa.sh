#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa"
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

mkdir ./eval_output/$CKPT/gqa

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/images \
        --answers-file ./eval_output/$CKPT/gqa/$SPLIT/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done

wait

output_file=./eval_output/$CKPT/gqa/$SPLIT/answers/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./eval_output/$CKPT/gqa/$SPLIT/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst ./eval_output/$CKPT/gqa/$SPLIT/testdev_balanced_predictions.json

python $GQADIR/eval.py --tier $GQADIR/testdev_balanced --predictions ./eval_output/$CKPT/gqa/$SPLIT/testdev_balanced_predictions.json
