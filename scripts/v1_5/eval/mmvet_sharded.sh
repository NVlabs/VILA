#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=$(( ${#GPULIST[@]} / 2 )) # Calculate chunks for 2 GPUs per chunk

MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_IDX1=$((IDX * 2))  # First GPU index
    GPU_IDX2=$((GPU_IDX1 + 1))  # Second GPU index

    CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX1]},${GPULIST[$GPU_IDX2]} python -m llava.eval.model_vqa \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
        --image-folder ./playground/data/eval/mm-vet/images \
        --answers-file ./eval_output/$CKPT/mm-vet/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done

wait

output_file=./eval_output/$CKPT/mm-vet/answers.json

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./eval_output/$CKPT/mm-vet/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./eval_output/$CKPT/mm-vet/answers.json \
    --dst ./eval_output/$CKPT/mm-vet/results.json

python llava/eval/eval_mmvet.py --results_file ./eval_output/$CKPT/mm-vet/results.json
