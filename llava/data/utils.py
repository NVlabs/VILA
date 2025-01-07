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


