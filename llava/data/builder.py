import os
import os.path as osp
from itertools import chain
from typing import Any, List, Optional

import torch
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from llava.data.datasets_mixture import DATASETS_LEGACY
from llava.train.args import DataArguments, TrainingArguments
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger

__all__ = ["DATASETS", "MIXTURES", "register_datasets", "register_mixtures", "parse_mixture", "build_dataset"]


def load_dataset_yaml(name):
    fname = f"{name}.yaml" if not name.endswith(".yaml") else name

    # yaml under llava/data/registry/datasets
    repo_path = osp.join(osp.dirname(__file__), "registry", "datasets", fname)
    if osp.exists(repo_path):
        return repo_path

    # # yaml under <fs yaml path>
    abs_path = osp.expanduser(fname)
    if osp.exists(abs_path):
        return abs_path

    raise FileNotFoundError(f"Dataset '{name}' is not found in the {repo_path} or {abs_path}.")


def register_datasets(name: Optional[str] = None):
    if name is None:
        name = os.environ.get("VILA_DATASETS", "default")
        logger.info(f"Registering datasets from environment: '{name}'.")
    # return io.load(osp.join(osp.dirname(__file__), "registry", "datasets", f"{name}.yaml"))
    dataset_meta = {}
    for _name in name.split(","):
        yamlpath = load_dataset_yaml(_name)
        logger.info(f"Registering datasets from: '{yamlpath}'.")
        meta = io.load(yamlpath)
        dataset_meta.update(meta)
    return dataset_meta


def register_mixtures():
    return io.load(os.path.join(os.path.dirname(__file__), "registry", "mixtures.yaml"))


DATASETS = register_datasets()
MIXTURES = register_mixtures()


def parse_mixture(mixture: str) -> List[str]:
    names = mixture.split("+") if "+" in mixture else [mixture]
    while any(name in MIXTURES for name in names):
        names = list(chain(*[MIXTURES.get(name, [name]) for name in names]))
    return sorted(names)


class RepeatedDataset(Dataset):
    def __init__(self, dataset: Dataset, times: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.times = times

    def __len__(self) -> int:
        return len(self.dataset) * self.times

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]


def get_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def build_dataset(
    mixture: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    logger.warning(f"Using mixture '{mixture}'.")
    datasets = []
    for name in parse_mixture(mixture):
        slice_subset = False
        if "@" in name:
            try:
                name, subset_choice = name.split("@")
                slice_subset = True
                slice_folder = os.environ.get(
                    "VILA_SLICE_FOLDER", "/home/ligengz/workspace/dataset-curation/filter_index"
                )
            except ValueError as e:
                logger.warning(f"failed on {name}")
                raise e
            # logger.warning(f"Using subset '{subset_choice}' for dataset '{name}'.")

        if "*" in name:
            name, times = name.split("*")
            times = int(times)
        else:
            times = 1

        if DATASETS is not None and name in DATASETS:
            if name in DATASETS_LEGACY:
                logger.warning(f"Dataset '{name}' exists in both new and legacy registries. Using the new one.")
            dataset = instantiate(DATASETS[name], _partial_=True)(
                tokenizer=tokenizer,
                data_args=data_args,
                global_batch_size=(
                    training_args.per_device_train_batch_size
                    # * torch.distributed.get_world_size()
                    * get_world_size()
                    * training_args.gradient_accumulation_steps
                ),
            )
        elif name in DATASETS_LEGACY:
            logger.warning(f"Dataset '{name}' is from the legacy registry. Please consider migrating it.")
            dataset = build_dataset_legacy(
                name,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Dataset '{name}' is not found in the registries.")

        if slice_subset:
            slice_json = osp.join(slice_folder, subset_choice, f"{name}.json")
            ignore_indices = io.load(slice_json)
            total_indices = range(len(dataset))
            indices = sorted(list(set(total_indices) - set(ignore_indices)))
            logger.info(f"[{name}] Slicing subset indices {slice_json}, len(dataset): {len(dataset)} => {len(indices)}")
            dataset = torch.utils.data.Subset(dataset, indices)

        if times > 1:
            dataset = RepeatedDataset(dataset, times)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def build_dataset_legacy(
    name: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    from llava.data.dataset import (
        LazyCCSWebDataset,
        LazyCoyoDataset,
        LazyCoyoWebDataset,
        LazyMMC4Dataset,
        LazySupervisedDataset,
        LazyVideoWebDataset,
        LazyWDSDataset,
    )
    from llava.data.dataset_impl.coyo_recap import LazyCoyoWebRecapDataset
    from llava.data.dataset_impl.panda70m import VILAPanda70m
    from llava.data.dataset_impl.sam import LazySAMWebDataset
    from llava.data.dataset_impl.textocr import VILATextOCR

    dataset = DATASETS_LEGACY[name]
    dataset_type = dataset.dataset_type
    if dataset_type == "torch":
        dataset_cls = LazySupervisedDataset
    elif dataset_type == "wds":
        dataset_cls = LazyWDSDataset
    elif dataset_type == "mmc4":
        dataset_cls = LazyMMC4Dataset
    elif dataset_type == "coyo":
        dataset_cls = LazyCoyoDataset
    elif dataset_type == "sam-wds":
        dataset_cls = LazySAMWebDataset
    elif dataset_type == "coyo-wds":
        dataset_cls = LazyCoyoWebDataset
    elif dataset_type == "coyo-wds-qas":
        print("dataset.py: Loading coyo-wds-qas class")
        from llava.data.dataset_impl.coyo_qa import LazyCoyoWebQADataset

        dataset_cls = LazyCoyoWebQADataset
    elif dataset_type == "coyo-wds-recap":
        dataset_cls = LazyCoyoWebRecapDataset
    elif dataset_type == "textocr":
        dataset_cls = VILATextOCR
    elif dataset_type == "panda70m":
        dataset_cls = VILAPanda70m
    elif dataset_type == "ccs-wds":
        dataset_cls = LazyCCSWebDataset
    elif dataset_type == "video-wds":
        dataset_cls = LazyVideoWebDataset
    else:
        raise NotImplementedError(f"{dataset_type} is not supported.")

    data_args.meta_path = getattr(dataset, "meta_path", None)
    data_args.caption_choice = getattr(dataset, "caption_choice", None)
    data_args.caption_choice_2 = getattr(dataset, "caption_choice_2", None)
    data_args.start_idx = getattr(dataset, "start_idx", None)
    data_args.end_idx = getattr(dataset, "end_idx", None)

    return dataset_cls(
        tokenizer=tokenizer,
        data_path=dataset.data_path,
        image_folder=getattr(dataset, "image_path"),
        data_args=data_args,
        training_args=training_args,
    )


