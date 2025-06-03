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

import io
import json
import os
import os.path as osp
import pathlib

from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.encoded_video import EncodedVideo, select_video_class


class VILAEncodedVideo(EncodedVideo):
    @classmethod
    def from_bytesio(cls, file_path: str, decode_audio: bool = True, decoder: str = "pyav"):
        if isinstance(file_path, io.BytesIO):
            video_file = file_path
            file_path = "tmp.mp4"
        elif isinstance(file_path, str):
            # We read the file with PathManager so that we can read from remote uris.
            with g_pathmgr.open(file_path, "rb") as fh:
                video_file = io.BytesIO(fh.read())
        else:
            print(f"unsupported type {type(file_path)}")
        video_cls = select_video_class(decoder)
        return video_cls(video_file, pathlib.Path(file_path).name, decode_audio)


class LabelingFactory(dict):
    def __init__(self, name: str, output_dir: str = "thinking_data"):
        self.name = name
        self.output_dir = output_dir

    def save(self):
        print(f"\033[1mSaving {self.name} to {self.output_dir}/{self.name} \033[0m")
        os.makedirs(osp.join(self.output_dir, self.name), exist_ok=True)
        with open(osp.join(self.output_dir, self.name, f"data.json"), "w") as f:
            json.dump(self, f, indent=2, ensure_ascii=False)

        with open(osp.join(self.output_dir, self.name, f"instances.json"), "w") as f:
            json.dump(list(self.values()), f, indent=2, ensure_ascii=False)

    def load(self, data_path=None):
        if data_path is None:
            data_path = osp.join(self.output_dir, self.name)
        if os.path.exists(osp.join(data_path, f"data.json")):
            with open(osp.join(data_path, f"data.json")) as f:
                data = json.load(f)
                self.update(data)
            print(f"Loaded {self.name} from {data_path}, total {len(list(data.items()))} instances")
        else:
            print(f"No data found for {self.name}, creating empty data")
