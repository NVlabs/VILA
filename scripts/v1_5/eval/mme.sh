#!/bin/bash
MODEL_PATH=$1
CKPT=$2
MMEDIR="./playground/data/eval/MME"
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

DATA_PATH="./playground/data/eval/MME/llava_mme.jsonl"
IMAGE_DIR="./playground/data/eval/MME/MME_Benchmark_release_version"
OUTPUT_PATH="runs/eval/$CKPT/mme/mme.jsonl"
RESULT_DIR="runs/eval/$CKPT/mme/mme_results"
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=8 \
    llava/eval/model_vqa_loader.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --question-file $DATA_PATH \
    --image-folder $IMAGE_DIR \
    --answers-file $OUTPUT_PATH

python $MMEDIR/convert_answer_to_mme.py \
    --experiment $OUTPUT_PATH

python $MMEDIR/eval_tool/calculation.py \
    --results_dir $RESULT_DIR
