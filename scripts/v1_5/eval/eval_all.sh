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

# Execute the scripts in sequential order, redirecting output
./scripts/v1_5/eval/vqav2.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/vqav2.txt &&
./scripts/v1_5/eval/gqa.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/gqa.txt &&
./scripts/v1_5/eval/vizwiz.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/vizwiz.txt &&
./scripts/v1_5/eval/sqa.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/scienceqa.txt &&
./scripts/v1_5/eval/textvqa.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/textvqa.txt &&
./scripts/v1_5/eval/pope.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/pope.txt &&
./scripts/v1_5/eval/mme.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/mme.txt &&
./scripts/v1_5/eval/mmbench.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/mmbench.txt &&
./scripts/v1_5/eval/mmbench_cn.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/mmbench_cn.txt &&
./scripts/v1_5/eval/seed.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/seed.txt &&
./scripts/v1_5/eval/mmmu.sh $checkpoint_path $model_name $conv_mode > eval_output/$model_name/mmmu.txt 

