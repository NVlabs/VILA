import io
import json
import os
from multiprocessing import Pool, cpu_count

import pandas as pd
from PIL import Image
from tqdm import tqdm


def general_conversation_preprocessor(save_path, item, dataset_name, id):
    # process the conversation item to llava format.
    conversations = []
    ret_item = dict(id=id)
    # ret_item["images"] = item["images"]
    img_paths = []
    for img_idx, img in enumerate(item["images"]):
        img = Image.open(io.BytesIO(img["bytes"]))
        save_path_to_append = os.path.join("images", dataset_name, f"{id}_{img_idx}.png")
        img_path = os.path.join(save_path, save_path_to_append)
        if img.mode == "CMYK":
            img = img.convert("RGB")
        img.save(img_path)
        img_paths.append(save_path_to_append)
    ret_item["images"] = img_paths
    old_conversations = item["texts"]
    for idx, conv in enumerate(old_conversations):
        if "user" in conv:
            if idx > 0:
                cur_conv = conv["user"]
                new_conv = {"from": "human", "value": cur_conv}
            else:
                cur_conv = conv["user"]
                new_conv = {"from": "human", "value": "<image>\n" * len(item["images"]) + cur_conv}
            conversations.append(new_conv)
        if "assistant" in conv:
            cur_conv = conv["assistant"]
            if cur_conv.startswith("Answer: "):
                cur_conv = cur_conv.replace("Answer: ", "")
            new_conv = {"from": "gpt", "value": cur_conv}
            conversations.append(new_conv)
    ret_item["conversations"] = conversations
    return ret_item


# def general_conversation_preprocessor(save_path, item, dataset_name, id):
#     # process the conversation item to llava format
#     ret_item = dict(id=id)
#     if item["image"] is not None:
#         img = Image.open(io.BytesIO(item["image"]["bytes"]))
#         save_path_to_append = os.path.join("images", dataset_name, f"{id}.png")
#         img_path = os.path.join(save_path, save_path_to_append)
#         if img.mode == "CMYK":
#             img = img.convert("RGB")
#         img.save(img_path)  # Save as optimized JPEG
#         ret_item["image"] = img_path
#     ret_item["conversations"] = item["conversations"].tolist()
#     return ret_item


def process_parquet_file(parquet_file_info):
    parquet_file, dataset_name, save_path, start_index = parquet_file_info
    df = pd.read_parquet(parquet_file, engine="pyarrow")  # Efficient Parquet reading
    llava_format_data = []

    # Process each row in the Parquet file
    for i, row in df.iterrows():  # Row-wise processing to save memory
        item = {key: row[key] for key in df.columns}
        processed_item = general_conversation_preprocessor(save_path, item, dataset_name, start_index + i)
        llava_format_data.append(processed_item)

    return llava_format_data, len(df)  # Return the processed data and the count of rows


def process_dataset(args):
    dataset_name, dataset_path, metadata_path, save_path = args
    output_file = os.path.join(metadata_path, f"{dataset_name}_train.jsonl")

    if os.path.exists(output_file):
        return

    print(f"Processing {dataset_name}...")

    dataset_files = sorted(
        filter(lambda x: x.endswith(".parquet"), os.listdir(os.path.join(dataset_path, dataset_name)))
    )

    parquet_file_info = []
    start_index = 0
    for fn in dataset_files:
        parquet_file_path = os.path.join(dataset_path, dataset_name, fn)
        parquet_file_info.append((parquet_file_path, dataset_name, save_path, start_index))
        # Instead of assuming a fixed number, update `start_index` after processing each file
        start_index += pd.read_parquet(parquet_file_path, engine="pyarrow").shape[0]

    num_processes = min(cpu_count(), len(parquet_file_info))
    print(f"Processing {len(parquet_file_info)} Parquet files in parallel using {num_processes} processes...")

    # Parallelize Parquet file processing within a dataset
    with Pool(processes=num_processes) as pool:
        llava_format_datasets = list(
            tqdm(pool.imap_unordered(process_parquet_file, parquet_file_info), total=len(parquet_file_info))
        )

    # Flatten the results and save them to a single JSONL file
    cur_llava_format_dataset = [item for sublist, _ in llava_format_datasets for item in sublist]

    with open(output_file, "w") as f:
        for item in cur_llava_format_dataset:
            json.dump(item, f)
            f.write("\n")


def main(
    dataset_path="./the_cauldron",
    save_path="./data_processed",
):
    metadata_path = os.path.join(save_path, "metadata")
    os.makedirs(metadata_path, exist_ok=True)

    keep_datasets = [
        "ai2d",
        "aokvqa",
        "diagram_image_to_text",
        "geomverse",
        "iam",
        "mapqa",
        "ocrvqa",
        "raven",
        "st_vqa",
        "vistext",
        "visual7w",
        "visualmrc",
        "websight",
        "intergps",
        "rendered_text",
        "iconqa",
        "vqarad",
        "cocoqa",
        "multihiertt",
        "nlvr2",
        "mimic_cgd",
        "robut_sqa",
        "hateful_memes",
        "robut_wikisql",
        "robut_wtq",
        "spot_the_diff",
        # InternVL split
        "infographic_vqa",
        "plotqa",
        "vsr",
        "vqarad",
    ]

    dataset_names = [
        name
        for name in sorted(os.listdir(dataset_path))
        if not name.startswith(".") and name in keep_datasets and os.path.isdir(os.path.join(dataset_path, name))
    ]

    os.makedirs(os.path.join(save_path, "images"), exist_ok=True)

    # Sequentially process datasets
    for dataset_name in tqdm(dataset_names):
        os.makedirs(os.path.join(save_path, "images", dataset_name), exist_ok=True)
        process_dataset((dataset_name, dataset_path, metadata_path, save_path))


if __name__ == "__main__":
    import fire

    fire.Fire(main)


