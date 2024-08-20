#!/bin/bash
#SBATCH -A llmservice_nlp_fm            #account
#SBATCH -p interactive,batch_block1,batch_block3,batch_block4,batch_singlenode
#SBATCH -t 04:00:00             		#wall time limit, hr:min:sec
#SBATCH -N 1                    		#number of nodes
#SBATCH -J vila:eval-video-mmev2	#job name
#SBATCH --gpus-per-node 2
#SBATCH --cpus-per-task 12
#SBATCH --mem-per-cpu 16G
#SBATCH --array=0-31
#SBATCH -e slurm-logs/sbatch/dev.err
#SBATCH -o slurm-logs/sbatch/dev.out
#SBATCH --dependency singleton

# cosmos_misc
# llmservice_nlp_fm
# nvr_elm_llm
#### 123 SBATCH --dependency singleton

idx=$SLURM_ARRAY_TASK_ID
total=$SLURM_ARRAY_TASK_COUNT
jname=seval-$idx-of-$total-random


ckpt=${1:-"Efficient-Large-Model/VILA1.5-3b"}
_model_name=$(echo $ckpt | rev | cut -d "/" -f 1 | rev)
# llava_v1
# hermes-2
model_name=${2:-"$_model_name"}
conv_mode=${3:-"hermes-2"}
temperature=${temperature:-"0.0"}
num_beams=${num_beams:-1}

num_video_frames=${num_video_frames:-"-1"}

OUTDIR=slurm-logs/$ckpt
#_$wname
> $OUTDIR/$jname.err
> $OUTDIR/$jname.out


srun \
    -e $OUTDIR/$jname.err -o $OUTDIR/$jname.out \
    python llava/eval/video_mme/video_eval.py \
        --model-path $ckpt --shard $idx --total $total --conv-mode $conv_mode \
        --output_dir=runs/eval/$model_name/video_mme/ --output_name=frames-$num_video_frames \
        --num_video_frames $num_video_frames \
        --temperature $temperature --num-beams $num_beams

exit 0

# usage examples

# debuging usage
python llava/eval/video_mme/video_eval.py --model-path Efficient-Large-Model/VILA1.5-3b

# sbatch launch
export temperature=0
export num_beams=1
# export num_video_frames=12
sbatch -A nvr_elm_llm -p interactive,$SLURM_PARTITION -J videomme:VILA1.5-40b \
    llava/eval/video_mme/sbatch_eval.sh \
    Efficient-Large-Model/VILA1.5-40b \
    VILA1.5-40b \
    hermes-2


sbatch -A nvr_elm_llm -p interactive,$SLURM_PARTITION -J videomme:VILA1.5-3b \
    llava/eval/video_mme/sbatch_eval.sh \
    Efficient-Large-Model/VILA1.5-3b \
    VILA1.5-3b \
    llava_v1


sbatch -A nvr_elm_llm -p interactive,$SLURM_PARTITION -J videomme:VILA1.5-40b \
    llava/eval/video_mme/sbatch_eval.sh \
    Efficient-Large-Model/VILA1.5-40b \
    VILA1.5-40b \
    hermes-2

sbatch -A llmservice_nlp_fm -p interactive,$SLURM_PARTITION -J videomme:vila-yi-34b-intern-6b-stage2_5_r620_sft_more_r2 \
    llava/eval/video_mme/sbatch_eval.sh \
    /home/jasonlu/workspace/VILA-Internal/checkpoints/vila-yi-34b-intern-6b-stage2_5_r620_sft_more_r2 \
    vila-yi-34b-intern-6b-stage2_5_r620_sft_more_r2 \
    hermes-2
