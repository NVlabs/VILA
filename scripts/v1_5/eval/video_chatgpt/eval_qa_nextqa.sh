#!/bin/bash
GPT_Zero_Shot_QA="./eval_output"
output_name=$1
pred_path="${GPT_Zero_Shot_QA}/${output_name}/NextQA_Zero_Shot_QA/merge.jsonl"
output_json="${GPT_Zero_Shot_QA}/${output_name}/NextQA_Zero_Shot_QA/results.json"
NEXTQA="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/video_datasets_v2/nextqa"
stopwords_file="${NEXTQA}/stopwords.txt"
gt_file="${NEXTQA}/test_data_nextoe/test.csv"
num_tasks=8

python3 llava/eval/video/eval_video_nextqa.py \
    --pred_path ${pred_path} \
    --output_json ${output_json} \
    --gt_file ${gt_file} \
    --stopwords_file ${stopwords_file} \