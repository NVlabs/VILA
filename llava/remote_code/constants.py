# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"

SENTINEL_TOKEN = "<vila/sentinel>"
MEDIA_TOKENS = {
    "image": "<image>",
    "video": "<vila/video>",
}
# <image> <vila/video> <vila/sentinel>
# TODO(ligeng): need to discuss with Zhijian for the following tokens for different models.
"""
151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151646: AddedToken("[BOS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151647: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151648: AddedToken("<vila/sentinel>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151649: AddedToken("<image>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
151650: AddedToken("<vila/video>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
"""
NUM_EXTRA_TOKENS = 8
