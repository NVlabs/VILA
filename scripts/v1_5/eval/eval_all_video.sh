#!/bin/bash

# Checkpoint path and model name (replace with your actual values)
checkpoint_path=$1
model_name=$2

# Create output directory if it doesn't exist
mkdir -p runs/eval/$model_name

# Execute the scripts in sequential order, redirecting output
.scripts/v1_5/eval/video_chatgpt/run_qa_msvd.sh $checkpoint_path $model_name > runs/eval/$model_name/msvd &&
