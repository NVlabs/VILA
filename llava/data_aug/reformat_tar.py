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

import os
import os.path as osp
import shutil
import tarfile

from fire import Fire
from tqdm import tqdm


def main(
    src_tar="~/datasets/sam-raw/sa_000000.tar",
    src_folder="~/datasets/sam-raw",
    tgt_folder="~/datasets/sam-reformat",
    overwrite=False,
):
    src_tar_path = osp.expanduser(src_tar)
    src_folder_path = osp.expanduser(src_folder)
    tgt_folder_path = osp.expanduser(tgt_folder)
    rpath = osp.relpath(src_tar_path, src_folder_path)
    fpath = osp.join(tgt_folder_path, rpath)
    fpath_tmp = osp.join(tgt_folder_path, rpath + ".tmp")

    if osp.exists(fpath) and not overwrite:
        print("Skipping")
        return

    t = tarfile.open(src_tar_path)

    os.makedirs(osp.dirname(fpath_tmp), exist_ok=True)
    tdev = tarfile.open(fpath_tmp, "w")

    for idx, member in tqdm(enumerate(t.getmembers())):
        print(idx, member, flush=True)
        tdev.addfile(member, t.extractfile(member.name))

    t.close()
    tdev.close()

    shutil.move(fpath_tmp, fpath)
    print("Finish")


if __name__ == "__main__":
    Fire(main)
