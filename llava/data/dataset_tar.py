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

import glob
import hashlib
import json
import os
import os.path as osp
import tarfile
from io import BytesIO
from multiprocessing.pool import ThreadPool as Pool

from PIL import Image, ImageFile
from torch.utils.data import ConcatDataset, Dataset, get_worker_info

try:  # make torchvision optional
    from torchvision.transforms.functional import to_tensor
except:
    to_tensor = None

ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnexpectedEOFTarFile(tarfile.TarFile):
    def _load(self):
        """Read through the entire archive file and look for readable
        members.
        """
        try:
            while True:
                tarinfo = self.next()
                if tarinfo is None:
                    break
        except tarfile.ReadError as e:
            assert e.args[0] == "unexpected end of data"
        self._loaded = True


class TarDataset(Dataset):
    """Dataset that supports Tar archives (uncompressed)."""

    def __init__(
        self,
        archive,
        transform=None,
        is_valid_file=lambda m: m.isfile() and m.name.lower().endswith((".png", ".jpg", ".jpeg")),
        ignore_unexpected_eof=False,
        cache_dir="~/.cache/tardataset",
    ):
        self.transform = transform
        self.archive = archive
        self.default_label = archive

        # open tar file. in a multiprocessing setting (e.g. DataLoader workers), we
        # have to open one file handle per worker (stored as the tar_obj dict), since
        # when the multiprocessing method is 'fork', the workers share this TarDataset.
        # we want one file handle per worker because TarFile is not thread-safe.
        worker = get_worker_info()
        worker = worker.id if worker else None
        self.tar_obj = {}  # lazy init

        # TODO: add a function hash
        # create a hash to cache results
        mtime = osp.getmtime(archive)
        # fn_hash = hash(is_valid_file)
        m = hashlib.sha256()
        m.update(str(mtime).encode("utf-8"))
        # m.update(str(fn_hash).encode("utf-8"))
        uuid = m.hexdigest()[:7]
        # print(mtime, uuid)

        tar_info = osp.realpath(archive).replace("/", "-") + f"-{uuid}.json"
        fpath = osp.join(osp.expanduser(cache_dir), tar_info)
        if not osp.exists(fpath):
            self.tar_obj = {
                worker: tarfile.open(archive) if ignore_unexpected_eof is False else UnexpectedEOFTarFile.open(archive)
            }
            print(f"{osp.basename(archive)} preparing tar.getnames() ...")
            self.all_members = self.tar_obj[worker].getmembers()
            # also store references to the iterated samples (a subset of the above)
            self.samples = [m.name for m in self.all_members if is_valid_file(m)]
            os.makedirs(osp.dirname(fpath), exist_ok=True)
            json.dump(self.samples, open(fpath, "w"), indent=2)
        else:
            print(f"loading cached tarinfo from {fpath}")
            self.samples = json.load(open(fpath))

    def __getitem__(self, index):
        image = self.get_image(self.samples[index], pil=True)
        image = image.convert("RGB")  # if it's grayscale, convert to RGB
        if self.transform:  # apply any custom transforms
            image = self.transform(image)
        return image, self.default_label

    def __len__(self):
        return len(self.samples)

    def get_image(self, name, pil=False):
        image = Image.open(BytesIO(self.get_file(name).read()))
        if pil:
            return image
        return to_tensor(image)

    def get_text_file(self, name, encoding="utf-8"):
        """Read a text file from the Tar archive, returned as a string.

        Args:
          name (str): File name to retrieve.
          encoding (str): Encoding of file, default is utf-8.

        Returns:
          str: Content of text file.
        """
        return self.get_file(name).read().decode(encoding)

    def get_file(self, name):
        """Read an arbitrary file from the Tar archive.

        Args:
          name (str): File name to retrieve.

        Returns:
          io.BufferedReader: Object used to read the file's content.
        """
        # ensure a unique file handle per worker, in multiprocessing settings
        worker = get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.archive)

        return self.tar_obj[worker].extractfile(name)

    def __del__(self):
        """Close the TarFile file handles on exit."""
        for o in self.tar_obj.values():
            o.close()

    def __getstate__(self):
        """Serialize without the TarFile references, for multiprocessing compatibility."""
        state = dict(self.__dict__)
        state["tar_obj"] = {}
        return state


class TarImageFolder(Dataset):
    """Dataset that supports Tar archives (uncompressed), with a folder per class.

    Similarly to torchvision.datasets.ImageFolder, assumes that the images inside
    the Tar archive are arranged in this way by default:
      root/
        dog.tar
        cat.tar
        ...
        bird.tar

      where

      dog.tar/
        xxx.png
        xxy.png
        [...]/xxz.png

      cat.tar/
        123.png
        nsdf3.png
        [...]/asd932_.png
    """

    def __init__(
        self,
        root,
        transform=None,
        max_loads=None,
        is_valid_file=lambda m: m.isfile() and m.name.lower().endswith((".png", ".jpg", ".jpeg")),
        pool_size=16,
    ):
        # load the archive meta information, and filter the samples
        super().__init__()

        self.transform = transform
        # assign a label to each image, based on its top-level folder name
        self.class_to_idx = {}
        self.targets = []

        tarfs = sorted(glob.glob(osp.join(root, "*.tar")))

        if max_loads is not None:
            tarfs = tarfs[: min(max_loads, len(tarfs))]
        print(tarfs)
        # tar_dst_list = [TarDataset(tar_path) for tar_path in tarfs]

        self.class2idx = {}

        for tar_fpath in tarfs:
            self.class2idx[tar_fpath] = len(self.class2idx.keys())

        print(self.class2idx)

        # parallel loading
        def worker(tar_path):
            return TarDataset(tar_path)

        pool = Pool(pool_size)
        jobs = []
        for tar_path in tarfs:
            jobs.append(pool.apply_async(worker, (tar_path,)))
        pool.close()
        pool.join()
        tar_dst_list = [_.get() for _ in jobs]
        self.dataset = ConcatDataset(tar_dst_list)
        print("TarImageFolder dataset init finish")

        if len(self.class2idx) == 0:
            raise OSError(
                "No classes (top-level folders) were found with the given criteria. The given\n"
                "extensions, is_valid_file is too strict, or the archive is empty."
            )

        # the inverse mapping is often useful
        self.idx2class = {v: k for k, v in self.class2idx.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _label = self.dataset[index]
        if self.transform:  # apply any custom transforms
            image = self.transform(image)

        label = self.class2idx[_label]
        return (image, label)


if __name__ == "__main__":
    # img = TarDataset(
    #     "/lustre/fs4/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/segmentation/sam/stage0/sa_000999.tar"
    # )
    # print("init finish, try to fetch images")
    # print(img[0])

    dst = TarImageFolder(
        "/lustre/fs4/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/segmentation/sam/stage0",
        max_loads=16,
    )
    print("init finish, try to fetch images")

    for idx, (image, label) in enumerate(dst):
        print(image, label)
        if idx > 100:
            break
