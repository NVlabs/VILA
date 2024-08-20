
#!/bin/bash

MODEL_PATH=$1       # Your model path
CKPT=$2             # Checkpoint name
CONV_MODE=$3   # Convolution mode
RANK=0              # Rank of the job (multi-node)
YOUR_ANSWER_PATH=runs/eval/${CKPT}/RefCOCO/

mkdir -p ${YOUR_ANSWER_PATH}

BSIZE=1      # batch size per job
CHUNKS=8     # number of parallel jobs

image_path='/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/coco/train2014/'
ann_path='/home/yunhaof/workspace/datasets/REC/minigpt_eval_annotations/'

# your LLM inference parameters
top_p=0.2
top_k=100
temperature=4.0

export HF_HOME=$(pwd)/hf_cache/
export WANDB_API_KEY='' # if you want to visualize predictions (and set [--use-wandb True] below)

for IDX in 0 1 2 3 4 5 6 7; do
    CIDX=$((RANK*8+IDX))
    echo "global idx: $CIDX rank: $RANK chunks: ${CHUNKS} local idx: ${IDX}"
    CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_refcoco \
        --model-path $MODEL_PATH \
        --data-path ${ann_path} \
        --image-folder ${image_path} \
        --answers-path ${YOUR_ANSWER_PATH} \
        --answers-file-name ${CHUNKS}_${CIDX}.jsonl \
        --num-chunks $CHUNKS \
        --batch-size ${BSIZE} \
        --chunk-idx $CIDX \
        --num-resample 20 \
        --top_p ${top_p} \
        --top_k ${top_k} \
        --temperature ${temperature} \
        --max-new-tokens 32 \
        --use-wandb False \
        --vis_interval 100 \
        --run_name ${CKPT} \
        --box_format vila \
        --conv-mode ${CONV_MODE} &
done

wait

for dataset in refcoco refcoco+; do
    for data_split in val testA testB; do
        output_file=${YOUR_ANSWER_PATH}/${dataset}/${data_split}/merge.jsonl

        # Clear out the output file if it exists.
        > "$output_file"

        # Loop through the indices and concatenate each file.
        for IDX in $(seq 0 $((CHUNKS-1))); do
            cat ${YOUR_ANSWER_PATH}/${dataset}/${data_split}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
        done
    done
done

wait

dataset=refcocog
for data_split in val test; do
    output_file=${YOUR_ANSWER_PATH}/${dataset}/${data_split}/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${YOUR_ANSWER_PATH}/${dataset}/${data_split}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
done

python -m llava.eval.eval_refcoco \
    --pred_path ${YOUR_ANSWER_PATH} \
    --data_path ${ann_path}
