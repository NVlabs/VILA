#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal
# Prerequisite: 1. pip install -e ".[eval]";

# Make sure partitions according to different clusters.
PARTITIONS="batch_block1,batch_block2,batch_block3,batch_block4"
#PARTITIONS="polar,grizzly"
ACCOUNT='llmservice_nlp_fm'
#ACCOUNT='nvr_elm_llm'

# Checkpoint path and model name (replace with your actual values)
checkpoint_path=$1
model_name=$2
conv_mode=hermes-2
if [ "$#" -ge 3 ]; then
    conv_mode="$3"
fi

mkdir -p runs/eval/$model_name

srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_videochatgpt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.videochatgpt_benchmark.txt ./scripts/v1_5/eval/video_chatgpt/run_videochatgpt_benchmark_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_videochatgpt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.videochatgpt_benchmark.txt ./scripts/v1_5/eval/video_chatgpt/run_videochatgpt_benchmark_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_videochatgpt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.videochatgpt_benchmark.txt ./scripts/v1_5/eval/video_chatgpt/run_videochatgpt_benchmark_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_videochatgpt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.videochatgpt_benchmark.txt ./scripts/v1_5/eval/video_chatgpt/run_videochatgpt_benchmark_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_videochatgpt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.videochatgpt_benchmark.txt ./scripts/v1_5/eval/video_chatgpt/run_videochatgpt_benchmark_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_videochatgpt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.videochatgpt_benchmark.txt ./scripts/v1_5/eval/video_chatgpt/run_videochatgpt_benchmark_sharded.sh $checkpoint_path $model_name $conv_mode &

srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_activitynet_sharded --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.activitynet.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_activitynet_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msvd_sharded --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.msvd.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msvd_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msrvtt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.msrvtt.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msrvtt_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msrvtt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.msrvtt.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msrvtt_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msrvtt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.msrvtt.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msrvtt_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msrvtt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.msrvtt.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msrvtt_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msrvtt_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.msrvtt.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msrvtt_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_tgif_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.tgif.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_tgif_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_tgif_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.tgif.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_tgif_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_nextqa_sharded --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.nextqa.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_nextqa_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_perception_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.perception.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_perception_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_perception_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.perception.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_perception_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_perception_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.perception.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_perception_sharded.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_vila_benchmark_sharded --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.vila_benchmark.txt ./scripts/v1_5/eval/video_chatgpt/run_vila_benchmark.sh $checkpoint_path $model_name $conv_mode &

sbatch -p $PARTITIONS llava/eval/video_mme/sbatch_eval.sh $checkpoint_path $model_name $conv_mode
