
CKPT_NAME="checkpoints/Video-LLaVA-7B"
Video_5_Benchmark="eval/Video_5_Benchmark"
pred_path="${Video_5_Benchmark}/${CKPT_NAME}/correctness_qa.json"
output_dir="${Video_5_Benchmark}/${CKPT_NAME}/gpt3/correctness"
output_json="${Video_5_Benchmark}/${CKPT_NAME}/results/correctness_qa.json"
api_key=""
api_base=""
num_tasks=8

python3 llava/eval/video/eval_benchmark_1_correctness.py \
    --pred_path  ${pred_path} \
    --output_dir  ${output_dir} \
    --output_json  ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}
