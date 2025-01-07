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

import asyncio
import base64
import json
import os
import pickle
import shutil
import ssl
import sys
from io import BytesIO

import aiofiles
import aiohttp
from tqdm import tqdm

input_dir = "/dataset/mmc4-test/jsonl"  # path to the MMC4 annotations
output_dir = "/dataset/mmc4-test/pkl"

os.makedirs(output_dir, exist_ok=True)

jsonl_list = sorted(os.listdir(input_dir))

if len(sys.argv) > 1:  # optional: shard the workload distributedly
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    jsonl_list = jsonl_list[start_idx:end_idx]

all_data = []
for fname in tqdm(jsonl_list):
    full_path = os.path.join(input_dir, fname)
    with open(full_path) as json_file:
        json_list = list(json_file)
    data_list = [json.loads(json_str) for json_str in json_list]
    for i_d, d in enumerate(data_list):  # register shard info
        d["shard"] = fname.split(".")[0]
        d["shard_idx"] = i_d
    all_data.extend(data_list)


semaphore = asyncio.Semaphore(512)  # limit number of simultaneous downloads

progress = tqdm(total=len(all_data), desc="Download progress")  # Initialize progress bar


async def download_file(session, data, output_dict):
    async with semaphore:  # limit the number of simultaneous downloads
        base = "/tmp"
        base = os.path.join(base, f"{data['shard']}")
        os.makedirs(base, exist_ok=True)
        base = os.path.join(base, f"{data['shard_idx']:09d}")
        os.makedirs(base, exist_ok=True)
        success = True
        downloaded = []
        for i_image, image_info in enumerate(data["image_info"]):
            try:
                async with session.get(image_info["raw_url"]) as resp:
                    if resp.status == 200:
                        f_name = f"{i_image:03d}." + image_info["raw_url"].split(".")[-1]
                        f_name = os.path.join(base, f_name)
                        f = await aiofiles.open(f_name, mode="wb")
                        await f.write(await resp.read())
                        await f.close()
                        downloaded.append(f_name)
                    else:
                        success = False
                        # print(f"Unable to download image at {url}. HTTP response code: {resp.status}")
            except Exception as e:
                print(e)
                success = False
                break
        if not success:  # only keep samples where all images are valid
            shutil.rmtree(base)
        else:
            success = True
            try:
                from PIL import Image

                for fname in downloaded:
                    img = Image.open(fname).convert("RGB")
                    size_limit = 336  # reduce the resolution to save disk space
                    if min(img.size) > size_limit:
                        w, h = img.size
                        if h < w:
                            new_h = size_limit
                            new_w = int(size_limit * w / h)
                        else:
                            new_w = size_limit
                            new_h = int(size_limit * h / w)
                        img = img.resize((new_w, new_h))

                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()

                    if data["shard"] not in output_dict:
                        output_dict[data["shard"]] = {}
                    if data["shard_idx"] not in output_dict[data["shard"]]:
                        output_dict[data["shard"]][data["shard_idx"]] = {}
                    output_dict[data["shard"]][data["shard_idx"]][fname.split("/")[-1]] = img_b64_str

            except Exception as e:
                print(e)
                success = False
                shutil.rmtree(base)

            if not success:
                if data["shard"] in output_dict and data["shard_idx"] in output_dict[data["shard"]]:
                    output_dict[data["shard"]].pop(data["shard_idx"])

        if os.path.exists(base):
            shutil.rmtree(base)

        progress.update(1)


async def main(data_list):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    conn = aiohttp.TCPConnector(ssl=ssl_context)

    output_dict = {}

    async with aiohttp.ClientSession(connector=conn) as session:

        tasks = []
        for data in data_list:
            tasks.append(download_file(session, data, output_dict))
        await asyncio.gather(*tasks)
    progress.close()  # Close progress bar when done

    for k, v in output_dict.items():  # TODO: @ligeng, please change to tar format?
        with open(os.path.join(output_dir, k + ".pkl"), "wb") as f:
            pickle.dump(v, f)


asyncio.run(main(all_data))


