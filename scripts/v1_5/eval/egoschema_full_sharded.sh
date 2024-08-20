#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

output_dir="runs/eval/$CKPT/EgoSchema_full"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=$(( ${#GPULIST[@]} / 2 )) # Calculate chunks for 2 GPUs per chunk

echo "Evaluating $CKPT with conv_mode $CONV_MODE..."
for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_IDX1=$((IDX * 2))  # First GPU index
    GPU_IDX2=$((GPU_IDX1 + 1))  # Second GPU index

    CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX1]},${GPULIST[$GPU_IDX2]} python3 llava/eval/model_vqa_ego_schema.py \
    --model-path $MODEL_PATH \
    --generation-config '{"max_new_tokens": 1024}' \
    --video-folder ./playground/data/eval/EgoSchema/videos \
    --question-file ./playground/data/eval/EgoSchema/questions.json \
    --gt-answers-file ./playground/data/eval/EgoSchema/subset_answers.json \
    --conv-mode $CONV_MODE \
    --split test \
    --output_dir ${output_dir} \
    --output_name ${CHUNKS}_${IDX} \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX &

done

wait

output_file=${output_dir}/merge.json

# Clear out the output file if it exists.
if [ -f "$output_file" ]; then
    > "$output_file"
fi


# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done

# convert json to csv for kaggle submission
python scripts/v1_5/eval/convert_pred_to_csv.py --output_dir ${output_dir}
