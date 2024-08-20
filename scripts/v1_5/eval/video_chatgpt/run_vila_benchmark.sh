#!/bin/bash

model_path=$1
model_name=$2
conv_mode=$3

echo vila_benchmark
output_dir="runs/eval/${model_name}/VILA-benchmark/pexels"
python llava/eval/video/model_vqa_videodemo_benchmark.py --model-path ${model_path} --eval_type pexels --output_dir ${output_dir} --conv-mode ${conv_mode} --temperature 0
output_dir="runs/eval/${model_name}/VILA-benchmark/robotics"
python llava/eval/video/model_vqa_videodemo_benchmark.py --model-path ${model_path} --eval_type robotics --output_dir ${output_dir} --conv-mode ${conv_mode} --temperature 0
output_dir="runs/eval/${model_name}/VILA-benchmark/av"
python llava/eval/video/model_vqa_videodemo_benchmark.py --model-path ${model_path} --eval_type av --output_dir ${output_dir} --conv-mode ${conv_mode} --temperature 0
output_dir="runs/eval/${model_name}/VILA-benchmark/long"
python llava/eval/video/model_vqa_videodemo_benchmark.py --model-path ${model_path} --eval_type long --output_dir ${output_dir} --conv-mode ${conv_mode} --temperature 0
