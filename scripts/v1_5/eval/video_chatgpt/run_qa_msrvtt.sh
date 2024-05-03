#!/bin/bash

model_path=$1
CKPT_NAME=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi
GPT_Zero_Shot_QA="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/videos/all"
gt_file_question="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/test_a.json"
output_dir="./eval_output/${CKPT_NAME}/MSRVTT_Zero_Shot_QA"



gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 llava/eval/model_vqa_video.py \
      --model-path ${model_path} \
      --video_dir ${video_dir} \
      --model_max_length 8192 \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_$((IDX)) \
      --num-chunks $CHUNKS \
      --chunk-idx $IDX \
      --conv-mode $CONV_MODE \
      --temperature 0 &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done