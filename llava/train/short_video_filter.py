import os
import json


video_json_path = '/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/activitynet-qa/train-processed-filtered.json'
video_output_json_path = '/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/activitynet-qa/train-processed-filtered-v2.json'
video_dir = '/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets/Video_ChatGPT/activitynet_videos'

video_json = json.load(open(video_json_path, "r"))

output_list = []
processed_files = 0
for video in video_json:
    print(f"Processing {processed_files} files")
    processed_files += 1
    if 'video' in video.keys():
        path = os.path.join(video_dir, video['video'])
    else:
        path = os.path.join(video_dir, video['id'] + '.mp4') 
    if os.path.isfile(path) and os.path.getsize(path) > 100 * 1024:
        output_list.append(video)
print(f"Num original videos: {len(video_json)}")
print(f"Num new videos: {len(output_list)}")

json.dump(output_list, open(video_output_json_path, "w"))
