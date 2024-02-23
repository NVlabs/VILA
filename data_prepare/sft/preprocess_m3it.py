from datasets import load_dataset, concatenate_datasets
import os
import pickle
import torch
import json
from tqdm import tqdm

# download M3IT to the dataset_path directory
dataset_path = "/dataset/llava-data/instruction-tuning/M3IT"
save_path = "/dataset/llava-data/instruction-tuning/new-vflan"
os.makedirs(save_path, exist_ok=True)

dataset_types = [
    "captioning",
    "captioning",
    "generation",
    "generation",
    "reasoning",
    "reasoning",
    "reasoning",
    "vqa",
    "vqa",
    "vqa",
    "vqa",
    "vqa",
    "vqa",
    "vqa"
]
dataset_names = [
    "image-paragraph-captioning",
    "textcap",
    "multi30k",
    "visual-dialog",
    "clevr",
    "nlvr",
    "visual-mrc",
    "docvqa",
    "gqa",
    "ivqa",
    "ocr-vqa",
    "st-vqa",
    "viquae",
    "vqa-v2",
]



assert len(dataset_types) == len(dataset_names)

for dataset_type, dataset_name in zip(dataset_types, dataset_names):
    print("Processing", dataset_name, "...")
    dataset = list(load_dataset(dataset_path, dataset_name)["train"])
    for item in dataset:
        question = item["instruction"] + item["inputs"]
        answer = item["outputs"]
        image = item["image_base64_str"]
        item.pop("instruction")
        item.pop("inputs")
        item.pop("outputs")
        item.pop("image_base64_str")
        item["question"] = question
        item["answer"] = answer
        item["image"] = image
    print(len(dataset), dataset[-1].keys())
    save_filename = f"{dataset_type}_{dataset_name}_train.pkl"
    save_filename = os.path.join(save_path, save_filename)
    with open(save_filename, "wb") as f:
        pickle.dump(dataset, f)
    
