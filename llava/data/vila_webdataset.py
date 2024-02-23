import shutil
import os, os.path as osp, io
import argparse
import pprint
import pickle
from bisect import bisect
import base64
from PIL import Image
import json
from filelock import Timeout, FileLock
from functools import lru_cache, reduce
import tarfile
from multiprocessing.pool import ThreadPool as Pool

import torch
import torch.distributed
from torch.utils.data import Dataset, get_worker_info, ConcatDataset

import getpass
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
            uuids = list(
                set([".".join(_.split(".")[:-1]) for _ in tar.getnames()])
            )
        except tarfile.ReadError as e:
            print(f"Skipping {tar_abspath}")
            print(e)
            return None
        nsamples = len(uuids)
        
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
        tar_meta = json.load(open(tar_abs_metapath, "r"))
    elif osp.exists(tar_real_metapath):
        print(f"    Loading realpath meta: {tar_real_metapath}")
        tar_meta = json.load(open(tar_real_metapath, "r"))
    else:
        return None
    return tar_meta


def prepare_wids_meta(data_path, cache_dir="~/datasets/vila-webds-meta", overwrite=False):
    # TODO(ligeng): speedup the generation
    #   1. parallelize the meta file generation 
    #   2. add options for meta file 
    
    meta_path_of_tar_abs = osp.join(
        osp.expanduser(cache_dir),
        data_path.replace("/", "--")
        + ".wdsmeta.json",
    )
    
    meta_path_of_tar_rel = osp.join(osp.expanduser(data_path), "wids-meta.json")
    if osp.exists(meta_path_of_tar_rel) and osp.exists(meta_path_of_tar_abs) and not overwrite:
        return 
    
    tar_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            fpath = osp.join(root, file)
            fpath = osp.relpath(fpath, data_path)
            print(fpath)
            if not fpath.endswith(".tar"):
                continue
            # fpath = osp.abspath(osp.join(root, file))
            tar_list.append(fpath)
    tar_list = sorted(tar_list)
    assert len(tar_list) > 0, f"no tar was found in the repository {data_path} !"

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
    save_json(meta, meta_path_of_tar_rel)
    


class VILAWebDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path=COYO_25M_VILA,
        meta_path=None,
        cache_dir="~/datasets/vila-webds-meta",
        max_shards_to_load=None,
    ):
        self.data_path = data_path
        self.meta_path = meta_path
        self.max_shards_to_load = max_shards_to_load

        _local_meta_path = osp.join(data_path, "wids-meta.json")
        if meta_path is None and osp.exists(_local_meta_path):
            print(f"loading from {_local_meta_path}")
            self.meta_path = meta_path = _local_meta_path
            
        if meta_path is None:
            self.meta_path = osp.join(
                osp.expanduser(cache_dir),
                data_path.replace("/", "--")
                + f".max_shards:{max_shards_to_load}"
                + ".wdsmeta.json",
            )

        assert osp.exists(self.meta_path), f"meta path not found in {self.meta_path} {_local_meta_path}:{osp.exists(_local_meta_path)}"
        print(f"[SimplyCoyo] Loading meta infomation {self.meta_path}", flush=True)

        # uuid = abs(hash(self.meta_path)) % (10 ** 8)
        import hashlib

        uuid = hashlib.sha256(self.meta_path.encode()).hexdigest()[:8]
        self.dataset = ShardListDataset(
            self.meta_path,
            cache_dir=osp.expanduser(
                f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"
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
    import torch
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", type=str, default=COYO_25M_VILA)
    # replaced by rank and world size
    parser.add_argument("-m", "--max-shards", type=int, default=None)
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()
    
    prepare_wids_meta(args.data_path)

    train_dataset = VILAWebDataset(
        data_path=args.data_path,
        max_shards_to_load=args.max_shards,
    )
    
    print(train_dataset[0])
    exit(0)
    print("overwrite:", args.overwrite)
    train_dataset = VILAWebDataset(
        data_path=args.data_path,
        max_shards_to_load=args.max_shards,
        # cache_dir="~/.cache/simplecoyo",
        overwrite=args.overwrite,
    )

    sampler = None
    from PIL import Image
    from torch.utils.data import default_collate
    from collections import defaultdict

    dloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        sampler=sampler,
        batch_size=1,
        collate_fn=VILAWebDataset.custom_collate,
        # num_workers=8,
    )
    # sampler.set_epoch(0)
    print(len(train_dataset), len(dloader))
    for idx, data in enumerate(dloader):
        print(f"{idx}-of-{len(dloader)}", data)
        if idx >= 5:
            break
