#!/bin/bash
MODEL_PATH=$1
CKPT=$2
SPLIT=$3
CONV_MODE=vicuna_v1
if [ "$#" -ge 4 ]; then
    CONV_MODE="$4"
fi

DATA_PATH="playground/data/eval/MMMU"
ANSWER_PATH="playground/data/eval/MMMU/answer_dict_val.json"
OUTPUT_PATH="runs/${CKPT}/mmmu/${SPLIT}/outputs.json"
GENERATION_CONFIG='{"max_new_tokens": 128, "do_sample": true, "num_beams": 5}'

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_mmmu.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --data-path $DATA_PATH \
    --output-path $OUTPUT_PATH \
    --split $SPLIT

if [ "$SPLIT" = "validation" ]; then
    python llava/eval/eval_mmmu.py --output_path $OUTPUT_PATH --answer_path $ANSWER_PATH
fi
