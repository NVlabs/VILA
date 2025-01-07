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
import os
import pickle
import ssl
import sys
from io import BytesIO

import aiofiles
import aiohttp
import pandas as pd
from tqdm import tqdm

input_dir = "/dataset/coyo-test/coyo-700m/data"  # path to the MMC4 annotations
output_dir = "/dataset/coyo-test/coyo-700m/pkl"  # path to the download file

os.makedirs(output_dir, exist_ok=True)

shard_idx = int(sys.argv[1])
anno_name = sorted(f for f in os.listdir(input_dir) if f.endswith(".parquet"))[shard_idx]

print(anno_name)

df = pd.read_parquet(os.path.join(input_dir, anno_name))
# keep only top 20% similar samples
df["clip_sim"] = df["clip_similarity_vitb32"] + df["clip_similarity_vitl14"]
n_org_samples = df.shape[0]
df = df[df["clip_sim"] > 0.6]
assert df.shape[0] / n_org_samples > 0.2

df.sort_values(by="clip_sim", inplace=True, ascending=False)
df = df.head(int(n_org_samples * 0.2))  # keep top 20%

df = df[["id", "url", "text", "clip_sim"]]
metadata_list = df.to_dict("records")

print(len(metadata_list))

base = "/tmp/coyo-cache"
os.makedirs(base, exist_ok=True)

semaphore = asyncio.Semaphore(512)  # limit number of simultaneous downloads

progress = tqdm(total=len(metadata_list), desc="Download progress")  # Initialize progress bar


async def download_file(session, data, output_dict):
    async with semaphore:  # limit the number of simultaneous downloads
        success = True
        f_name = str(data["id"])
        f_name = os.path.join(base, f_name)
        try:
            async with session.get(data["url"], timeout=10) as resp:
                if resp.status == 200:
                    f = await aiofiles.open(f_name, mode="wb")
                    await f.write(await resp.read())
                    await f.close()
                else:
                    success = False
                    # print(f"Unable to download image at {url}. HTTP response code: {resp.status}")
        except Exception as e:
            print(e)
            success = False

        if success:
            try:
                from PIL import Image

                img = Image.open(f_name).convert("RGB")
                size_limit = 336
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

                data["image"] = img_b64_str
                output_dict[data["id"]] = data

            except Exception as e:
                print(e)
                success = False

        if os.path.exists(f_name):
            os.remove(f_name)

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

    v = list(output_dict.values())

    # TODO: @ligeng, please help change to webdataset format
    with open(os.path.join(output_dir, f"{shard_idx:04d}.pkl"), "wb") as f:
        pickle.dump(v, f)


asyncio.run(main(metadata_list))


