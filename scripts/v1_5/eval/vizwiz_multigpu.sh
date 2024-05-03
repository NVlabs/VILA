#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa"
MODEL_PATH=$1
CONV_MODE=vicuna_v1
if [ "$#" -ge 2 ]; then
    CONV_MODE="$2"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
        --image-folder ./playground/data/eval/vizwiz/test \
        --answers-file ./playground/data/eval/vizwiz/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done
wait

output_file=./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vizwiz/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-v1.5-13b.json
