#!/bin/bash

DEFAULT_RUN_NAME="vila-qwen2-7b-s2-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=1024
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

source scripts/setups/train.sh

STAGE2_PATH=${1:-"Efficient-Large-Model/qwen2-vl-7b-instruct-pretrain"}

DATA_MIXTURE=${DATA_MIXTURE:-"sharegpt4v_sft"}
echo "DATA_MIXTURE = $DATA_MIXTURE"

if [ "$NNODES" = "1" ] || [ "$NNODES" = "2" ]; then
    echo "Detected on single machine. Automatically set batch size to 1 for debugging purpose."
    PER_DEVICE_TRAIN_BATCH_SIZE=16
fi

PER_DEVICE_TRAIN_BATCH_SIZE=${COAT_BS:-"12"}
echo "Final batch size to $PER_DEVICE_TRAIN_BATCH_SIZE for COAT purpose."

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE2_PATH \
        --data_mixture $DATA_MIXTURE \
        --vision_tower google/siglip-so400m-patch14-384 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --max_steps 300 \
        --save_steps 5000 \
        --save_total_limit 1 \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to wandb

# --optim sgd
