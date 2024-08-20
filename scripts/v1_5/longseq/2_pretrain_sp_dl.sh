#!/bin/bash
bs=4

torchrun --nnodes=1 --nproc_per_node=8 --master_port=25002 \
    llava/train/train_hybrid.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path openlm-research/open_llama_3b_v2 \
    --version v1 \
    --data_mixture panda_longseq_pjlab \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector mlp2x_gelu \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/vila-llama2-7b-pjlab-debug \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 270 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --seq_parallel_size 4

    # --seq_parallel_ring_size 2
