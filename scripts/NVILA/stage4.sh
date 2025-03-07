#!/bin/bash

DEFAULT_RUN_NAME="nvila-video-joint-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=2048
DEFAULT_GRADIENT_ACCUMULATION_STEPS=8

source scripts/setups/train.sh

DATA_MIXTURE=(
    "llava-video-sft"
    "llave_onevision_images_sft"
    "llava-instruct-150k"
    "sharegpt4v-instruct-100k"
    "activitynet-dvc*3"
    "youcook2-dvc*3"
    "medical-dvc"
    "warehouse-dvc"
    "activitynet-el*3"
    "youcook2-el*3"
    "didemo-el"
    "charades-el"
    "medical-el"
    "warehouse-el"
    "nextqa"
    "activitynet-rtl"
)
IFS=$'\n' DATA_MIXTURE=($(sort <<<"${DATA_MIXTURE[*]}"))
DATA_MIXTURE=$(IFS=+; echo "${DATA_MIXTURE[*]}")
echo "DATA_MIXTURE = $DATA_MIXTURE"

STAGE2_PATH="runs/train/stage3_9tile/model"

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE2_PATH \
        --data_mixture $DATA_MIXTURE \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_2x2_fix \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --video_encoder '{"_target_": "llava.model.encoders.TSPVideoEncoder", "pool_sizes": [[8, 1, 1]]}' \
        --num_video_frames 256 \
        --num_time_tokens 100 \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 35 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 16384 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to none
