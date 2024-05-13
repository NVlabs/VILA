#!/bin/bash

source ~/.bashrc
conda activate multi-modal
which python

cd ~/workspace/multi-modality-research/VILA

model_name=$1

echo activity_net
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_activitynet.sh ${model_name}
echo msvd
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_msvd.sh ${model_name}
echo msrvtt
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_msrvtt.sh ${model_name}
echo tgif
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_tgif.sh ${model_name}
echo nextqa
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_nextqa.sh ${model_name}
echo perception_test
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_perception.sh ${model_name}
