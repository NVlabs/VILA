#!/bin/bash
#SBATCH -A cosmos_misc                  #account
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
# llava_v1
# hermes-2
conv_mode=${2:-"hermes-2"}
temperature=${3:-"0.0"}
num_beams=${4:-1}

OUTDIR=slurm-logs/$ckpt
#_$wname
> $OUTDIR/$jname.err
> $OUTDIR/$jname.out


srun \
    -e $OUTDIR/$jname.err -o $OUTDIR/$jname.out \
    python llava/data_aug/video_eval.py \
        --model-path $ckpt --shard $idx --total $total --conv-mode $conv_mode \
        --temperature $temperature --num-beams $num_beams

'''
python llava/data_aug/video_eval.py --model-path Efficient-Large-Model/VILA1.5-3b -c
python llava/data_aug/video_eval.py --model-path Efficient-Large-Model/VILA1.5-13b -c
python llava/data_aug/video_eval.py --model-path Efficient-Large-Model/VILA1.5-40b -c
python llava/data_aug/video_eval.py --model-path Efficient-Large-Model/VILA1.5-40b --conv-mode hermes-2
python llava/data_aug/video_eval.py --shard 7 --total 10


tmp=0.2
beam=1
python llava/data_aug/video_eval.py --model-path Efficient-Large-Model/VILA1.5-3b -c --temperature $tmp --num-beams $beam
python llava/data_aug/video_eval.py --model-path Efficient-Large-Model/VILA1.5-13b -c --temperature $tmp --num-beams $beam
python llava/data_aug/video_eval.py --model-path Efficient-Large-Model/VILA1.5-40b -c --temperature $tmp --num-beams $beam


tmp=0
beam=1
sbatch -A cosmos_misc -p interactive,interactive_singlenode,$SLURM_PARTITION -J fz-13b-video-mme-eval \
    llava/data_aug/seval.sh /home/jasonlu/workspace/VILA-Internal/checkpoints/fuzhao2 llava_v1 $tmp $beam
python llava/data_aug/video_eval.py --model-path /home/jasonlu/workspace/VILA-Internal/checkpoints/fuzhao2 -c --temperature $tmp --num-beams $beam

sbatch -A nvr_elm_llm -p interactive,interactive_singlenode,$SLURM_PARTITION -J 3b-video-mme-eval \
    llava/data_aug/seval.sh Efficient-Large-Model/VILA1.5-3b llava_v1 $tmp $beam

sbatch -A nvr_elm_llm -p interactive,interactive_singlenode,$SLURM_PARTITION -J 13b-video-mme-eval \
    llava/data_aug/seval.sh Efficient-Large-Model/VILA1.5-13b llava_v1 $tmp $beam

sbatch -A cosmos_misc -p interactive,interactive_singlenode,$SLURM_PARTITION -J 40b-video-mme-eval \
    llava/data_aug/seval.sh Efficient-Large-Model/VILA1.5-40b hermes-2 $tmp $beam
'''
