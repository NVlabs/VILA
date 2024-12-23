#!/bin/bash

set -e

TASK=$1
MODEL_PATH=$2
CONV_MODE=$3

NUM_VIDEO_FRAMES=$(echo "$TASK" | cut -d'-' -f2-)
IFS='-' read -ra segments <<< "$TASK"
unset segments[${#segments[@]}-1]
TASK=$(IFS=-; echo "${segments[*]}")

MAX_TILES=$4

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/lmms-$TASK"}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}

export LMMS_EVAL_PLUGINS=llava.eval.lmms
export HF_HOME=$HOME/.cache/huggingface
export CACHE_DIR=$OUTPUT_DIR/cache

torchrun --nproc_per_node=$NPROC_PER_NODE \
	-m lmms_eval \
	--model vila_internal \
	--model_args model_path=$MODEL_PATH,conv_mode=$CONV_MODE,num_video_frames=$NUM_VIDEO_FRAMES,max_tiles=$MAX_TILES \
	--tasks $TASK \
	--log_samples \
	--output_path $OUTPUT_DIR

mv $OUTPUT_DIR/*/*_results.json $OUTPUT_DIR/results.json || true
mv $OUTPUT_DIR/*/*_samples_*.jsonl $OUTPUT_DIR/samples.jsonl || true
