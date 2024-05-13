#!/bin/bash
GPT_Zero_Shot_QA="./eval_output"
output_name=$1
pred_path="${GPT_Zero_Shot_QA}/${output_name}/Activitynet_Zero_Shot_QA/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/${output_name}/Activitynet_Zero_Shot_QA/gpt4-turbo"
output_json="${GPT_Zero_Shot_QA}/${output_name}/Activitynet_Zero_Shot_QA/results.json"
num_tasks=8

python3 llava/eval/video/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --num_tasks ${num_tasks}