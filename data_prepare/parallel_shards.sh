JOBS_LIMIT=${1:-32}  # Set your limit here
workdir=${2:-$HOME/dataset/panda70m/panda70m_training_10m}


workdir=$HOME/dataset/video_datasets_v2/internvid/video_data_tar

parallel_size=32
idx_size=$(( parallel_size - 1 ))

mkdir -p slurm-logs/data

for idx in $(seq 0 $idx_size); do
    while [ $(jobs -rp | wc -l) -ge $JOBS_LIMIT ]; do
        sleep 1
    done
    echo "Running jobs $(jobs -rp | wc -l) $idx-of-$parallel_size";

    srun -A $SLURM_ACCOUNT \
        -p cpu,cpu_1,cpu_long -t 4:00:00 -J creating-WDS-$idx-of-$parallel_size \
        --cpus-per-task 8 \
        --mem-per-cpu 8G \
        --dependency singleton \
        -e slurm-logs/data/$idx-of-$parallel_size.err \
        -o slurm-logs/data/$idx-of-$parallel_size.txt \
        python llava/data/simple_vila_webdataset.py $workdir --shards=$idx --total=$parallel_size &
done
wait

python llava/data/simple_vila_webdataset.py $workdir
