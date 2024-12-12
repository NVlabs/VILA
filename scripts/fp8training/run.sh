#/bin/bash
set -e

# export VILA_SLURM_ACCOUNT=llmservice_nlp_fm
export DATA_MIXTURE=${DATA_MIXTURE:-"sharegpt4v_sft"}

nodes=${1:-"1"}
# fp16, fp8, fp8_memoryefficient
# qwen_fp8, qwen_fp16
BATCHSIZE=${BATCHSIZE:-"12"}
TASK=${TASK:-"fp16"}


export TRITON_HOME=$HOME/.cache/trion
export COAT_BS=$BATCHSIZE
export WANDB_PROJECT="VILA-fp8-dev"

export ngpus=${ngpus:-"8"}
mname=$DATA_MIXTURE-nodes_$nodes-$TASK-bs$COAT_BS
CONV_MODE="auto"

if [ "$nodes" = "1" ] || [ "$nodes" = "2" ]; then
    VILA_SLURM_PARTITION=$VILA_SLURM_PARTITION
fi

export WANDB_MODE=disabled
export SLURM_JOB_GPUS_PER_NODE=$ngpus
export VILA_SLURM_PARTITION=batch
# export VILA_SLURM_PARTITION=batch,hp,large_runs
if [ ! -f runs/bench/$mname/model/config.json ]; then
    vila-run -t 1:00:00 -m bench -J $mname -N $nodes --uuid \
        --gpus-per-node $ngpus --uuid \
        bash scripts/experimental/te_qlinear/sft_${TASK}.sh
fi
exit 0

export WANDB_PROJECT="VILA-fp8-dev-eval"
# pip install lmms-eval==0.2.1
vila-eval \
    --model-path runs/train/$mname/model \
    --model-name $mname \
    --conv-mode $CONV_MODE \
    --tags-include local \
    --report-to wandb

exit 0
