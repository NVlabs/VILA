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

from dataclasses import dataclass, field

@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    image_path: str = field(
        default=None, metadata={"help": "Path to the training image data."}
    )

DATASETS_MIXTURES = {}
DATASETS = {}

def add_dataset(dataset):
    DATASETS.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():
    coyo_25m = Dataset(
        dataset_name='coyo',
        dataset_type='coyo',
        data_path='./playground/data/coyo-700m/pkl02-split')
    add_dataset(coyo_25m)
    coyo_25m_wds = Dataset(
        dataset_name='coyo-webdataset',
        dataset_type='coyo-wds',
        data_path='./playground/data/coyo-25m-vila')
    add_dataset(coyo_25m_wds)
    mmc4core = Dataset(
        dataset_name='mmc4core',
        dataset_type='mmc4',
        data_path='./playground/data/mmc4-core/pkl-core')
    add_dataset(mmc4core)
    llava_1_5_mm_align = Dataset(
        dataset_name='llava_1_5_align',
        dataset_type='torch',
        data_path='./playground/data/LLaVA-Pretrain/LLaVA-CC3M-Pretrain-595K.json',
        image_path='./playground/data/LLaVA-Pretrain/images'
    )
    add_dataset(llava_1_5_mm_align)
    llava_1_5_sft = Dataset(
        dataset_name='llava_1_5_sft',
        dataset_type='torch',
        data_path='./playground/data/llava_v1_5_mix665k.json',
        image_path='./playground/data'
    )
    add_dataset(llava_1_5_sft)
    sharegpt4v_sft = Dataset(
        dataset_name='sharegpt4v_sft',
        dataset_type='torch',
        data_path='./playground/data/sharegpt4v/sharegpt4v_mix738k_remove_sa.json',
        image_path='./playground/data'
    )
    add_dataset(sharegpt4v_sft)
    vflan = Dataset(
        dataset_name='vflan',
        dataset_type='vflan',
        data_path='./playground/data/vlm-flan-clean-text1m-nosqa-sharded'
    )
    add_dataset(vflan)
    
    
    DATASETS_MIXTURES.update({'coyo_25m_wds': [coyo_25m_wds]})
    DATASETS_MIXTURES.update({'llava_1_5_mm_align': [llava_1_5_mm_align]})
    DATASETS_MIXTURES.update({'llava_1_5_sft': [llava_1_5_sft]})
    DATASETS_MIXTURES.update({'sharegpt4v_sft': [sharegpt4v_sft]})
    DATASETS_MIXTURES.update({'coyo_25m': [coyo_25m]})
    DATASETS_MIXTURES.update({'mmc4core': [mmc4core]})
    DATASETS_MIXTURES.update({'coyo_25m_mmc4core': [coyo_25m, mmc4core]})
    DATASETS_MIXTURES.update({'vflan': [vflan]})
    DATASETS_MIXTURES.update({'vflan_sharegpt4v_sft': [vflan, sharegpt4v_sft]})
