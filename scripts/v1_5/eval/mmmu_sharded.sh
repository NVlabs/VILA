#!/bin/bash
# Sample usage: ./scripts/v1_5/eval/mmmu_sharded.sh checkpoints/llava-v1.5-7b-sft-vflan+sharegpt4v-sharegpt4v-pretrain-siglip llava-v1.5-7b-sft-vflan+sharegpt4v-sharegpt4v-pretrain-siglip --validation
# The script uses two GPUs (with pipeline parallelism) for each chunk.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

MODEL_PATH=$1
CKPT=$2
SPLIT=$3
CONV_MODE=vicuna_v1
if [ "$#" -ge 4 ]; then
    CONV_MODE="$4"
fi
# Input Validation
if [[ "$SPLIT" != "validation" && "$SPLIT" != "test" ]]; then
    echo "Error: SPLIT must be either 'validation' or 'test'"
    exit 1 
fi

CHUNKS=$(( ${#GPULIST[@]} / 2 )) # Calculate chunks for 2 GPUs per chunk

for IDX in $(seq 0 $((CHUNKS-1))); do
  GPU_IDX1=$((IDX * 2))  # First GPU index
  GPU_IDX2=$((GPU_IDX1 + 1))  # Second GPU index

  CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX1]},${GPULIST[$GPU_IDX2]} python -m llava.eval.model_vqa_mmmu_sharded \
    --model_path $MODEL_PATH \
    --data_path ./playground/data/eval/MMMU \
    --conv-mode $CONV_MODE \
    --config_path llava/eval/mmmu_utils/configs/llava1.5.yaml \
    --output_path ./eval_output/$CKPT/MMMU/$SPLIT.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --split $SPLIT &
done

wait

python llava/eval/mmmu_utils/merge_jsons.py --prediction-path ./eval_output/$CKPT/MMMU/$SPLIT --num-chunks $CHUNKS

if [ "$SPLIT" = "validation" ]; then
  python llava/eval/eval_mmmu.py  --output_path ./eval_output/$CKPT/MMMU/$SPLIT.json --answer_path llava/eval/mmmu_utils/answer_dict_val.json
fi
