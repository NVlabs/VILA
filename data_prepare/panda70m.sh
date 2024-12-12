JOBS_LIMIT=${1:-32}  # Set your limit here
workdir=${2:-$HOME/dataset/panda70m/panda70m_training_10m}

wname=$(echo $workdir | rev | cut -d "/" -f 1 | rev)

echo "Parallely checking for all shards in $workdir / $wname"
parallel_size=32
idx_size=$(( parallel_size - 1 ))

mkdir -p slurm-logs/data

for idx in $(seq 0 $idx_size); do
    while [ $(jobs -rp | wc -l) -ge $JOBS_LIMIT ]; do
        sleep 1
    done
    echo "Running jobs $(jobs -rp | wc -l) $wname-$idx-of-$parallel_size";

    srun -A llmservice_nlp_fm \
        -p cpu,cpu_1,cpu_long -t 4:00:00 -J cleanup-$wname-$idx-of-$parallel_size \
        --cpus-per-task 8 \
        --mem-per-cpu 8G \
        -e slurm-logs/data/$idx-of-$parallel_size.err \
        -o slurm-logs/data/$idx-of-$parallel_size.txt \
        python llava/data/dataset_impl/panda70m.py --workdir=$workdir --shards=$idx --total=$parallel_size &

done

# bash data_prepare/panda70m.sh 32 $HOME/dataset/panda70m/panda70m_training_10m;
# bash data_prepare/panda70m.sh 32 $HOME/dataset/panda70m/panda70m_training_2m;
# bash data_prepare/panda70m.sh 32 $HOME/dataset/panda70m/panda70m_testing;

# --exclusive \
# --cpus-per-task 8 \
# --mem-per-cpu 8G \
