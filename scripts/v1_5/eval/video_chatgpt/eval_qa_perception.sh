#!/bin/bash
GPT_Zero_Shot_QA="./eval_output"
output_name=$1
pred_path="${GPT_Zero_Shot_QA}/${output_name}/PerceptionTest_Zero_Shot_QA/merge.jsonl"
output_json="${GPT_Zero_Shot_QA}/${output_name}/PerceptionTest_Zero_Shot_QA/results.json"
DATA_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/perception_test"
num_tasks=8

python3 llava/eval/video/eval_video_perception.py \
    --pred_path ${pred_path} \
    --output_json ${output_json} \