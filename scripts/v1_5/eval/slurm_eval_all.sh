#!/bin/bash

# Prerequisite:
# 1. pip install -e ".[eval]";
# 2. Softlink "/home/yunhaof/workspace/datasets/evaluation" to "YOUR_VILA_PATH/playground/data/eval" before evaluation.

source scripts/setups/base.sh

# Checkpoint path and model name (replace with your actual values)
checkpoint_path=$1
model_name=$2
conv_mode=vicuna_v1
if [ "$#" -ge 3 ]; then
    conv_mode="$3"
fi

## local evaluation benchmarks
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/mmmu_val --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mmmu_val.txt ./scripts/v1_5/eval/mmmu.sh $checkpoint_path $model_name validation $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/mathvista_testmini --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mathvista_testmini.txt ./scripts/v1_5/eval/mathvista.sh $checkpoint_path $model_name testmini $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/mathvista_test --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mathvista_test.txt ./scripts/v1_5/eval/mathvista.sh $checkpoint_path $model_name test $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/llavabench --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.llavabench.txt ./scripts/v1_5/eval/llavabench.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/sqa --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.science.txt ./scripts/v1_5/eval/sqa.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/textvqa --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.textvqa.txt ./scripts/v1_5/eval/textvqa.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/mme --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mme.txt ./scripts/v1_5/eval/mme.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/mmvet --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mmvet.txt ./scripts/v1_5/eval/mmvet.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/pope --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.pope.txt ./scripts/v1_5/eval/pope.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/seed --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.seed.txt ./scripts/v1_5/eval/seed.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/gqa --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.gqa.txt ./scripts/v1_5/eval/gqa.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/chartqa --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.chartqa.txt ./scripts/v1_5/eval/chartqa.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/ai2d --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.ai2d.txt ./scripts/v1_5/eval/ai2d.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/egoschema --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.egoschema.txt ./scripts/v1_5/eval/egoschema.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/egoschema_full --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.egoschema_full.txt ./scripts/v1_5/eval/egoschema_full.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/cinepile --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.cinepile.txt ./scripts/v1_5/eval/cinepile.sh $checkpoint_path $model_name $conv_mode &

## server evaluation benchmarks
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/vqav2 --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.vqav2.txt ./scripts/v1_5/eval/vqav2.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/vizwiz --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.vizwiz.txt ./scripts/v1_5/eval/vizwiz.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/mmbench --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mmbench.txt ./scripts/v1_5/eval/mmbench.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/mmbench_cn --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mmbench_cn.txt ./scripts/v1_5/eval/mmbench_cn.sh $checkpoint_path $model_name $conv_mode &
srun -p $VILA_SLURM_PARTITION -A $VILA_SLURM_ACCOUNT -N 1 -t 4:00:00 -J $VILA_SLURM_ACCOUNT:eval/mmmu_test --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mmmu_test.txt ./scripts/v1_5/eval/mmmu.sh $checkpoint_path $model_name test $conv_mode &
