# huggingface-cli download Efficient-Large-Model/coyo-25m-recap --local-dir coyo-25m-recap --repo-type dataset --local-dir-use-symlinks False --resume-download
JOBS_LIMIT=${1:-32}  # Set your limit here
# model_id=NousResearch/Llama-2-13b-chat-hf
# model_id=google/gemma-7b-it
# task="cap2qa"
task="rephrase"

for f in /home/ligengz/workspace/coyo-25m-recap/*.json; do
  while [ $(jobs -rp | wc -l) -ge $JOBS_LIMIT ]; do
    sleep 1
  done

  model_id="mistralai/Mistral-7B-Instruct-v0.2"
  fname=$(echo $f | rev | cut -d "/" -f 1 | rev)
  model=$(echo $model_id | cut -d "/" -f 2)
  # Replace this with your actual command
  echo "[$model_id] Processing $task on $f and running jobs $(jobs -rp | wc -l)"; \
  srun --label -A $SLURM_ACCOUNT -N 1 \
    -p $SLURM_PARTITION,batch_singlenode \
    -t 4:00:00 \
    -J vila:cap2qa-$fname-$model --gpus-per-node 8 --exclusive \
    -e slurm-logs/dev-$task/$fname-$model.err \
    -o slurm-logs/dev-$task/$fname-$model.out \
    torchrun --nproc-per-node 8 llava/data_aug/caption2qa.py --data_path=$f --task=$task --model_id=$model_id &

  # model_id="mit-han-lab/Meta-Llama-3-70B-Instruct"
  # model_id="/home/ligengz/downloads/Meta-Llama-3-70B-Instruct"
  model_id="deepseek-ai/deepseek-llm-67b-chat"

for model_id in "deepseek-ai/deepseek-llm-67b-chat" "mit-han-lab/Meta-Llama-3-70B-Instruct"; do
  fname=$(echo $f | rev | cut -d "/" -f 1 | rev)
  model=$(echo $model_id | cut -d "/" -f 2)
  # Replace this with your actual command
  echo "[$model_id] Processing $task on $f and running jobs $(jobs -rp | wc -l)"; \
  srun --label -A $SLURM_ACCOUNT -N 1 \
    -p $SLURM_PARTITION,batch_singlenode \
    -t 4:00:00 \
    -J vila:cap2qa-$fname-$model --gpus-per-node 8 --exclusive \
    -e slurm-logs/dev-$task/$fname-$model.err \
    -o slurm-logs/dev-$task/$fname-$model.out \
    torchrun --nproc-per-node 8 llava/data_aug/caption2qa.py --data_path=$f --task=$task --model_id=$model_id --load_in_4bit=True &
done

done
wait
