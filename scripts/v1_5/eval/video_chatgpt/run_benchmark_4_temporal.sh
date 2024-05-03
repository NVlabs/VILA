
CKPT_NAME="Video-LLaVA-7B"
model_path="checkpoints/${CKPT_NAME}"
cache_dir="./cache_dir"
Video_5_Benchmark="eval/Video_5_Benchmark"
video_dir="${Video_5_Benchmark}/Test_Videos"
gt_file="${Video_5_Benchmark}/Benchmarking_QA/temporal_qa.json"
output_dir="${Video_5_Benchmark}/${CKPT_NAME}"
output_name="temporal_qa"

python3 llava/eval/video/run_inference_benchmark_general.py \
    --model_path ${model_path} \
    --cache_dir ${cache_dir} \
    --video_dir ${video_dir} \
    --gt_file ${gt_file} \
    --output_dir ${output_dir} \
    --output_name ${output_name}
