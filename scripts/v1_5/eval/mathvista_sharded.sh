#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=$(( ${#GPULIST[@]} / 2 )) # Calculate chunks for 2 GPUs per chunk

MODEL_PATH=$1
CKPT=$2
SPLIT=$3
CONV_MODE=vicuna_v1
if [ "$#" -ge 4 ]; then
    CONV_MODE="$4"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_IDX1=$((IDX * 2))  # First GPU index
    GPU_IDX2=$((GPU_IDX1 + 1))  # Second GPU index

    CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX1]},${GPULIST[$GPU_IDX2]} python -m llava.eval.model_vqa_mathvista \
        --model-path $MODEL_PATH \
        --generation-config '{"max_new_tokens": 128}' \
        --split $SPLIT \
        --answers-file runs/eval/$CKPT/MathVista/MathVista_$SPLIT/${CHUNKS}_${IDX}.json \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode hermes-2 &
done

wait

output_file=runs/eval/$CKPT/MathVista/MathVista_$SPLIT.json

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat runs/eval/$CKPT/MathVista/MathVista_$SPLIT/${CHUNKS}_${IDX}.json >> "$output_file"
done


if [ "$SPLIT" = "testmini" ]; then
    python llava/eval/eval_mathvista.py --answer_file $output_file
fi
