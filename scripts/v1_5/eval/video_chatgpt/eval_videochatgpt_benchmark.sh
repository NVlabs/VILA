#!/bin/bash
set -e
GPT_Zero_Shot_QA="runs/eval"
output_name=$1
model=${2:-"gpt-3.5-turbo-0125"}
RESULT_DIR="${GPT_Zero_Shot_QA}/${output_name}/videochatgpt"
num_tasks=8
api_key=${OPENAI_API_KEY:-"none"}

# 1
task=1_correctness
pred_path="${RESULT_DIR}/generic_qa/merge.jsonl"
output_dir="${RESULT_DIR}/${task}/${model}"
output_json="${RESULT_DIR}/${task}/${model}.json"

python -m llava.eval.video.eval_benchmark_1_correctness \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --num_tasks ${num_tasks} \
    --model ${model}


# 2
task=2_detailed
pred_path="${RESULT_DIR}/generic_qa/merge.jsonl"
output_dir="${RESULT_DIR}/${task}/${model}"
output_json="${RESULT_DIR}/${task}/${model}.json"

python -m llava.eval.video.eval_benchmark_2_detailed_orientation \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --num_tasks ${num_tasks} \
    --model ${model}


# 3
task=3_context
pred_path="${RESULT_DIR}/generic_qa/merge.jsonl"
output_dir="${RESULT_DIR}/${task}/${model}"
output_json="${RESULT_DIR}/${task}/${model}.json"

python -m llava.eval.video.eval_benchmark_3_context \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --num_tasks ${num_tasks} \
    --model ${model}


# 4
task=4_temporal
pred_path="${RESULT_DIR}/temporal_qa/merge.jsonl"
output_dir="${RESULT_DIR}/${task}/${model}"
output_json="${RESULT_DIR}/${task}/${model}.json"

python -m llava.eval.video.eval_benchmark_4_temporal \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --num_tasks ${num_tasks} \
    --model ${model}


# 5
task=5_consistency
pred_path="${RESULT_DIR}/consistency_qa/merge.jsonl"
output_dir="${RESULT_DIR}/${task}/${model}"
output_json="${RESULT_DIR}/${task}/${model}.json"

python -m llava.eval.video.eval_benchmark_5_consistency \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --num_tasks ${num_tasks} \
    --model ${model}
