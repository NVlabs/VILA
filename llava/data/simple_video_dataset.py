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

import argparse
import base64
import getpass
import io
import json
import multiprocessing
import os
import os.path as osp
import pickle
import pprint
import shutil
import tarfile
from bisect import bisect
from functools import lru_cache, reduce
from multiprocessing.pool import ThreadPool as Pool

import torch
import torch.distributed
from filelock import FileLock, Timeout
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, get_worker_info

from llava.wids import ShardListDataset


# @lru_cache(maxsize=32)
def load_tarfile(tar_path):
    return tarfile.open(tar_path)


# INTERNVID = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video"
INTERNVID = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v3/ego4d/ego4d_clips_tar/ego4d_1m"
CACHEDIR = "/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v3/ego4d/ego4d_clips_tar/ego4d_1m-webds-meta"


def process_tarfile(tar_abspath, tar_meta_path, cache_dir):
    tar_realpath = osp.realpath(tar_abspath)
    tar_real_meta_path = osp.join(
        osp.expanduser(cache_dir),
        "dev",
        tar_realpath.replace("/", "--") + ".wdsmeta.json",
    )

    print(f"Fetch meta information {tar_abspath} ...")

    if not osp.exists(tar_meta_path) and not osp.exists(tar_real_meta_path):
        print(f"    Generating meta: {tar_meta_path}")
        try:
            tar = load_tarfile(tar_abspath)
            uuids = list({".".join(_.split(".")[:-1]) for _ in tar.getnames()})
        except tarfile.ReadError as e:
            print(f"Skipping {tar_abspath}")
            print(e)
            return
        nsamples = len(uuids)
        url = osp.abspath(tar_abspath)
        tar_meta = {
            "url": url,
            "nsamples": nsamples,
            "filesize": osp.getsize(tar_abspath),
        }
        os.makedirs(osp.dirname(tar_meta_path), exist_ok=True)
        json.dump(tar_meta, open(tar_meta_path, "w+"), indent=2)

    if osp.exists(tar_meta_path):
        print(f"    Generating abs meta: {tar_meta_path}")
        tar_meta = json.load(open(tar_meta_path))
    elif osp.exists(tar_real_meta_path):
        print(f"    Generating abs meta: {tar_real_meta_path}")
        tar_meta = json.load(open(tar_real_meta_path))
    else:
        raise NotImplementedError

    tar_meta["url"] = osp.abspath(tar_abspath)
    os.makedirs(osp.dirname(tar_meta_path), exist_ok=True)
    json.dump(tar_meta, open(tar_meta_path, "w+"), indent=2)
    if tar_meta_path != tar_real_meta_path and not osp.exists(tar_real_meta_path):
        # tar_meta["url"] = osp.realpath(tar_abspath)
        print(f"    [abs2real] Copying {tar_meta_path} => {tar_real_meta_path}")
        os.makedirs(osp.dirname(tar_real_meta_path), exist_ok=True)
        json.dump(tar_meta, open(tar_real_meta_path, "w+"), indent=2)

    return tar_meta


class SimpleVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-1300K-flt",
        # cache_dir="/home/ligengz/.cache/simplecoyo",
        # cache_dir="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/vila-webds-meta",
        cache_dir="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/internvid/video_data_tar/InternVid-1300K-flt-webds-meta",
        meta_path=None,
        # image_load_mode="pil",  # pil / rawbytes / fpath,
        max_shards_to_load=None,
        overwrite=False,
    ):
        self.data_path = data_path
        self.meta_path = meta_path
        if meta_path is None:
            self.meta_path = osp.join(
                cache_dir,
                data_path.replace("/", "--") + f".max_shards:{max_shards_to_load}" + ".wdsmeta.json",
            )
            self.max_shards_to_load = max_shards_to_load

        if not osp.exists(self.meta_path) or overwrite:
            assert (
                not torch.distributed.is_initialized()
            ), "Dataset meta file does not exist and generating may take a long time. \
                Please exit distributed mode and run `python llava/train/simple_coyo_dataset.py <webdataset path>`. \
                or set proper `meta_path=` when initializing."
            print(f"Meta path not found: {self.meta_path}")
            print(f"Walking through dirs {data_path}")
            # tar_list = sorted([f for f in os.listdir(data_path) if f.endswith(".tar")])
            tar_list = []
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    fpath = osp.join(root, file)
                    fpath = osp.relpath(fpath, data_path)
                    if not fpath.endswith(".tar"):
                        continue
                    # fpath = osp.abspath(osp.join(root, file))
                    tar_list.append(fpath)
            tar_list = sorted(tar_list)

            if "internvid" in data_path:
                meta_name = "internvid-dev"
            elif "ego4d" in data_path:
                meta_name = "ego4d-dev"
            else:
                print(f"Unknown dataset: {data_path}. Please include the dataset name in the data path")
                meta_name = "unknown"
                raise NotImplementedError

            meta = {
                "name": meta_name,
                "__kind__": "SimpleVideoDataset",
                "wids_version": 1,
                "shardlist": [],
            }

            max_processes = 16  # Set the maximum number of processes
            pool = multiprocessing.Pool(processes=max_processes)
            results = []
            for idx, tar_relpath in enumerate(tar_list):
                tar_abspath = osp.join(data_path, tar_relpath)
                tar_meta_path = osp.join(
                    osp.expanduser(cache_dir),
                    "dev",
                    tar_abspath.replace("/", "--") + ".wdsmeta.json",
                )
                result = pool.apply_async(process_tarfile, (tar_abspath, tar_meta_path, cache_dir))
                results.append(result)

            pool.close()
            pool.join()

            meta["shardlist"] = [result.get() for result in results if result.get() is not None]
            # sorted by tar names
            meta["shardlist"] = sorted(meta["shardlist"], key=lambda x: x["url"])
            os.makedirs(osp.dirname(self.meta_path), exist_ok=True)
            json.dump(meta, open(self.meta_path, "w+"), indent=2)

        print(f"[SimplyVideo] Loading meta infomation {self.meta_path}", flush=True)

        # uuid = abs(hash(self.meta_path)) % (10 ** 8)
        import hashlib

        uuid = hashlib.sha256(self.meta_path.encode()).hexdigest()[:8]
        # get user name
        user_name = os.getenv("USER")
        print(f"User name: {user_name}")
        self.dataset = ShardListDataset(
            self.meta_path,
            cache_dir=osp.expanduser(
                f"/lustre/fsw/portfolios/nvr/users/{user_name}/cache/_wids_cache/{getpass.getuser()}-{uuid}"
            ),
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def simple_collate(batch):
        batched_data = {}
        for data in batch:
            for k, v in data.items():
                if k not in batched_data:
                    batched_data[k] = []
                batched_data[k].append(v)
        return dict(batched_data)

    @staticmethod
    def custom_collate(batch):
        def transform2list(a: dict):
            # trasnform all leaf nodes to list
            for k, v in a.items():
                if isinstance(v, dict):
                    a[k] = transform2list(v)
                else:
                    a[k] = [
                        v,
                    ]
            return a

        def merge(a: dict, b: dict, path=[], strict=False):
            c = {}
            keys = set(a.keys()).union(b.keys())
            # print(keys)
            for key in keys:
                if key in a and key in b:
                    if isinstance(a[key], dict) and isinstance(b[key], dict):
                        # apply recursively
                        c[key] = merge(a[key], b[key], path + [str(key)], strict=strict)
                    else:
                        c[key] = a[key] + b[key]
                else:
                    if strict:
                        raise Exception("Conflict at " + ".".join(path + [str(key)]))
                    c[key] = a[key] if key in a else b[key]
            return c

        tasks = (transform2list(_) for _ in batch)
        return reduce(merge, tasks)


if __name__ == "__main__":
    import argparse

    import torch
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", type=str, default=INTERNVID)
    parser.add_argument("cache_path", nargs="?", type=str, default=CACHEDIR)
    # replaced by rank and world size
    parser.add_argument("-m", "--max-shards", type=int, default=None)
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()

    print("overwrite:", args.overwrite)
    train_dataset = SimpleVideoDataset(
        data_path=args.data_path,
        max_shards_to_load=args.max_shards,
        cache_dir=args.cache_path,
        overwrite=args.overwrite,
    )

    sampler = None
    # from PIL import Image
    from collections import defaultdict

    from torch.utils.data import default_collate

    dloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        sampler=sampler,
        batch_size=2,
        collate_fn=SimpleVideoDataset.custom_collate,
        num_workers=8,
    )
    # sampler.set_epoch(0)
    print(len(train_dataset), len(dloader))
    for idx, data in enumerate(dloader):
        print(f"{idx}-of-{len(dloader)}", data)
        if idx >= 5:
            break
