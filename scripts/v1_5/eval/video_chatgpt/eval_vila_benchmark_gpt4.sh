MODEL=$1
DATA_TYPE=$2
EVAL_TYPE=$3
pred_path=$4
Video_5_Benchmark="eval_vila_benchmark"
output_dir="${Video_5_Benchmark}/${MODEL}/${DATA_TYPE}/gpt4/${EVAL_TYPE}"
output_json="${Video_5_Benchmark}/${MODEL}/${DATA_TYPE}/results/${EVAL_TYPE}_qa.json"

mkdir -p "${Video_5_Benchmark}/${MODEL}/${DATA_TYPE}/results/"

python llava/eval/video/eval_benchmark_${EVAL_TYPE}.py \
    --pred_path  ${pred_path} \
    --output_dir  ${output_dir} \
    --output_json  ${output_json} \
    --num_tasks 8
