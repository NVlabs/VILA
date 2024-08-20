#!/bin/bash

#source /lustre/fsw/portfolios/nvr/users/${USER}/anaconda3/bin/activate
#conda init
source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal
# Prerequisite: 1. pip install -e ".[eval]"; 2.Softlink "/home/yunhaof/workspace/datasets/evaluation" to "YOUR_VILA_PATH/playground/data/eval" before evaluation.

# Make sure partitions according to different clusters.
#PARTITIONS="batch_block1,batch_block2,batch_block3,batch_block4"
PARTITIONS="polar,grizzly"
#PARTITIONS="interactive,grizzly,polar,polar2,polar3,polar4"
#ACCOUNT="llmservice_nlp_fm"
ACCOUNT="nvr_elm_llm"
#ACCOUNT="nvr_lpr_aiagent"

# Checkpoint path and model name (replace with your actual values)
checkpoint_path=$1
model_name=$2
conv_mode=hermes-2
if [ "$#" -ge 3 ]; then
    conv_mode="$3"
fi

# Create output directory if it doesn't exist
mkdir -p runs/eval/$model_name

## server evaluation benchmarks
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_vqav2 --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.vqav2.txt ./scripts/v1_5/eval/vqav2_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_vizwiz --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.vizwiz.txt ./scripts/v1_5/eval/vizwiz_sharded.sh $checkpoint_path $model_name $conv_mode &
# srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_mmbench --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mmbench.txt ./scripts/v1_5/eval/mmbench_sharded.sh $checkpoint_path $model_name $conv_mode &
# srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_mmbench_cn --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mmbench_cn.txt ./scripts/v1_5/eval/mmbench_cn_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_mmmu_test --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mmmu_test.txt ./scripts/v1_5/eval/mmmu_sharded.sh $checkpoint_path $model_name test $conv_mode &
## local evaluation benchmarks
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_mmmu_val --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mmmu_val.txt ./scripts/v1_5/eval/mmmu_sharded.sh $checkpoint_path $model_name validation $conv_mode &

# srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_mathvista_testmini --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mathvista_testmini.txt ./scripts/v1_5/eval/mathvista_sharded.sh $checkpoint_path $model_name testmini $conv_mode &
# srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_mathvista_test --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mathvista_test.txt ./scripts/v1_5/eval/mathvista_sharded.sh $checkpoint_path $model_name test $conv_mode &

srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_llavabench --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.llavabench.txt ./scripts/v1_5/eval/llavabench_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_sqa --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.science.txt ./scripts/v1_5/eval/sqa_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_textvqa --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.textvqa.txt ./scripts/v1_5/eval/textvqa_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_mme --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mme.txt ./scripts/v1_5/eval/mme_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_mmvet --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.mmvet.txt ./scripts/v1_5/eval/mmvet_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_pope --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.pope.txt ./scripts/v1_5/eval/pope_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_seed --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.seed.txt ./scripts/v1_5/eval/seed_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_gqa --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.gqa.txt ./scripts/v1_5/eval/gqa_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_chartqa --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.chartqa.txt ./scripts/v1_5/eval/chartqa_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_ai2d --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.ai2d.txt ./scripts/v1_5/eval/ai2d_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_egoschema_full --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.egoschema_full.txt ./scripts/v1_5/eval/egoschema_full_sharded.sh $checkpoint_path $model_name $conv_mode &
