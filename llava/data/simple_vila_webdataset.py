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


COYO_25M_VILA = "~/datasets/captioning/coyo-25m-vila"


def save_json(obj, fpath):
    print(f"saving to {fpath}")
    os.makedirs(osp.dirname(fpath), exist_ok=True)
    json.dump(obj, open(fpath, "w+"), indent=2)


def generate_and_load_tar_meta(data_path, tar_path, cache_dir, overwrite=False):
    tar_abspath = osp.abspath(osp.join(data_path, tar_path))
    tar_abs_metapath = osp.join(
        osp.expanduser(cache_dir),
        "dev",
        tar_abspath.replace("/", "--") + ".wdsmeta.json",
    )
    tar_real_metapath = osp.join(
        osp.expanduser(cache_dir),
        "dev",
        osp.realpath(tar_abspath).replace("/", "--") + ".wdsmeta.json",
    )

    if not osp.exists(tar_abs_metapath) and not osp.exists(tar_real_metapath) or overwrite:
        # generate meta information for both abs and real file paths
        print(f"    Generating meta: {tar_abs_metapath}")
        try:
            tar = load_tarfile(tar_abspath)
            uuids = list({osp.splitext(_)[0] for _ in tar.getnames()})
        except tarfile.ReadError as e:
            print(f"Skipping {tar_abspath}")
            print(e)
            return None
        nsamples = len(uuids)
        # print(uuids)
        # print(nsamples)
        # input()

        tar_meta = {
            "url": osp.abspath(tar_abspath),
            "nsamples": nsamples,
            "filesize": osp.getsize(tar_abspath),
        }
        save_json(tar_meta, tar_abs_metapath)

        tar_meta = {
            "url": osp.realpath(tar_abspath),
            "nsamples": nsamples,
            "filesize": osp.getsize(tar_abspath),
        }
        save_json(tar_meta, tar_real_metapath)

    if osp.exists(tar_abs_metapath):
        print(f"    Loading abs meta: {tar_abs_metapath}")
        tar_meta = json.load(open(tar_abs_metapath))
    elif osp.exists(tar_real_metapath):
        print(f"    Loading realpath meta: {tar_real_metapath}")
        tar_meta = json.load(open(tar_real_metapath))
    else:
        return None
    return tar_meta


def generate_wids_meta(tar_list, data_path, cache_dir, idx=0, total=0):
    # TODO(ligeng): add return value
    meta_path_of_tar_abs = osp.join(
        osp.expanduser(cache_dir),
        data_path.replace("/", "--") + ".wdsmeta.json",
    )

    meta_path_of_tar_rel = osp.join(osp.expanduser(data_path), "wids-meta.json")
    ####################################################################################
    meta = {
        "name": "coyo-dev",
        "__kind__": "VILA-WebDataset",
        "wids_version": 1,
        "shardlist": [],
    }

    for idx, tar_path in enumerate(tar_list):
        print(f"{idx}-of-{len(tar_list)}")
        tar_meta = generate_and_load_tar_meta(data_path, tar_path, cache_dir)
        # tar_meta["url"] = tar_path
        tar_meta["url"] = osp.abspath(osp.join(data_path, tar_path))
        meta["shardlist"].append(tar_meta)

    # sorted by tar names
    meta["shardlist"] = sorted(meta["shardlist"], key=lambda x: x["url"])
    if total == 0:
        # only save for all information
        save_json(meta, meta_path_of_tar_abs)

    ####################################################################################
    meta = {
        "name": "coyo-dev",
        "__kind__": "VILA-WebDataset",
        "wids_version": 1,
        "shardlist": [],
    }
    for idx, tar_path in enumerate(tar_list):
        print(f"{idx}-of-{len(tar_list)}")
        tar_meta = generate_and_load_tar_meta(data_path, tar_path, cache_dir)
        if tar_meta is None:
            continue
        tar_meta["url"] = tar_path
        meta["shardlist"].append(tar_meta)

    # sorted by tar names
    meta["shardlist"] = sorted(meta["shardlist"], key=lambda x: x["url"])
    if total == 0:
        # only save for all information
        save_json(meta, meta_path_of_tar_rel)


def prepare_wids_meta(data_path, cache_dir="~/datasets/vila-webds-meta-2", idx=0, total=0):
    cache_dir = osp.expanduser(cache_dir)
    # TODO(ligeng): speedup the generation
    #   1. parallelize the meta file generation
    #   2. add options for meta file
    tar_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            fpath = osp.join(root, file)
            fpath = osp.relpath(fpath, data_path)
            # print(fpath)
            if not fpath.endswith(".tar"):
                continue
            # fpath = osp.abspath(osp.join(root, file))
            tar_list.append(fpath)
    tar_list = sorted(tar_list)

    if total > 0:
        chunk = len(tar_list) // total
        begin_idx = chunk * idx
        end_idx = chunk * (idx + 1)
        if idx == total - 1:
            end_idx = len(tar_list)
        tar_list = tar_list[begin_idx:end_idx]
        print(f"{chunk}, {begin_idx} -> {end_idx}")

    assert len(tar_list) > 0, f"no tar was found in the repository {data_path} !"
    print(f"generating meta for total {len(tar_list)} files.")
    generate_wids_meta(tar_list, data_path, cache_dir, idx=idx, total=total)


class VILAWebDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path=COYO_25M_VILA,
        meta_path=None,
        cache_dir=osp.join(osp.expanduser("~/.cache/vila/webds-meta")),
        max_shards_to_load=None,
    ):
        self.data_path = osp.expanduser(data_path)
        self.meta_path = osp.expanduser(meta_path) if meta_path is not None else None
        # self.max_shards_to_load = max_shards_to_load

        _local_meta_path = osp.join(self.data_path, "wids-meta.json")
        if meta_path is None and osp.exists(_local_meta_path):
            print(f"loading from {_local_meta_path}")
            self.meta_path = meta_path = _local_meta_path

        if meta_path is None:
            self.meta_path = osp.join(
                osp.expanduser(cache_dir),
                self.data_path.replace("/", "--") + f".max_shards:{max_shards_to_load}" + ".wdsmeta.json",
            )

        assert osp.exists(self.meta_path), f"meta path not found in [{self.meta_path}] or [{_local_meta_path}]"
        print(f"[VILA-forked-Webdataset] Loading meta infomation {self.meta_path}", flush=True)

        # uuid = abs(hash(self.meta_path)) % (10 ** 8)
        import hashlib

        uuid = hashlib.sha256(self.meta_path.encode()).hexdigest()[:8]
        self.dataset = ShardListDataset(
            self.meta_path,
            cache_dir=osp.expanduser(f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"),
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
    parser.add_argument("data_path", nargs="?", type=str)  # , default=COYO_25M_VILA)
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--shards", type=int, default=0)
    parser.add_argument("--total", type=int, default=0)
    parser.add_argument("--test-all", action="store_true")
    args = parser.parse_args()

    print("Data path: ", args.data_path)
    prepare_wids_meta(args.data_path, idx=args.shards, total=args.total)

    if args.total > 0:
        print("building meta information only")
        exit(0)

    train_dataset = VILAWebDataset(
        data_path=args.data_path,
    )
    # print("overwrite:", args.overwrite)
    print("dataset size: ", len(train_dataset))
    print(train_dataset[0])

    if args.test_all:
        print("iterating all dataset for data integrity.")
        train_dataset = VILAWebDataset(
            data_path=args.data_path,
            # cache_dir="~/.cache/simplecoyo",
            # overwrite=args.overwrite,
        )

        sampler = None
        from collections import defaultdict

        from PIL import Image

        dloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            sampler=sampler,
            batch_size=8,
            collate_fn=VILAWebDataset.custom_collate,
            num_workers=8,
        )
        # dloader = train_dataset
        # sampler.set_epoch(0)
        print(len(train_dataset), len(dloader))
        count = 0
        for idx, data in enumerate(dloader):
            if ".json" in data and ".mp4" in data:
                print(f"{idx}-of-{len(dloader)}", type(data), count)
            else:
                count += 1

            # if idx >= 5:
            #     break


