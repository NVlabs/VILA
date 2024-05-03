#!/bin/bash

# Checkpoint path and model name (replace with your actual values)
checkpoint_path=$1
model_name=$2
conv_mode=vicuna_v1
if [ "$#" -ge 3 ]; then
    conv_mode="$3"
fi

# Create output directory if it doesn't exist
mkdir -p eval_output/$model_name

./scripts/v1_5/eval/video_chatgpt/run_qa_activitynet.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/activitynet.txt &&
./scripts/v1_5/eval/video_chatgpt/run_qa_msvd.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/msvd.txt &&
./scripts/v1_5/eval/video_chatgpt/run_qa_msrvtt.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/msrvtt.txt &&
./scripts/v1_5/eval/video_chatgpt/run_qa_tgif.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/tgif.txt &&
./scripts/v1_5/eval/video_chatgpt/run_qa_perception.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/perception.txt &&