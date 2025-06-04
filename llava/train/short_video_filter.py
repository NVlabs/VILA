# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

video_json_path = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/activitynet-qa/train-processed-filtered.json"
video_output_json_path = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/activitynet-qa/train-processed-filtered-v2.json"
video_dir = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets/Video_ChatGPT/activitynet_videos"

video_json = json.load(open(video_json_path))

output_list = []
processed_files = 0
for video in video_json:
    print(f"Processing {processed_files} files")
    processed_files += 1
    if "video" in video.keys():
        path = os.path.join(video_dir, video["video"])
    else:
        path = os.path.join(video_dir, video["id"] + ".mp4")
    if os.path.isfile(path) and os.path.getsize(path) > 100 * 1024:
        output_list.append(video)
print(f"Num original videos: {len(video_json)}")
print(f"Num new videos: {len(output_list)}")

json.dump(output_list, open(video_output_json_path, "w"))
