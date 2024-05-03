

CKPT_NAME="checkpoints/Video-LLaVA-7B"
Video_5_Benchmark="eval/Video_5_Benchmark"
pred_path="${Video_5_Benchmark}/${CKPT_NAME}/temporal_qa.json"
output_dir="${Video_5_Benchmark}/${CKPT_NAME}/gpt3/temporal"
output_json="${Video_5_Benchmark}/${CKPT_NAME}/results/temporal_qa.json"
api_key=""
api_base=""
num_tasks=8

python3 llava/eval/video/eval_benchmark_4_temporal.py \
    --pred_path  ${pred_path} \
    --output_dir  ${output_dir} \
    --output_json  ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}
