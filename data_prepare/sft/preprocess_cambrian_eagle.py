import json
import os

from tqdm import tqdm

data_path = "./Cambrian7M_withsystemprompt.json"

all_sources = dict()

with open(data_path) as f:
    records = json.load(f)

for record in records:
    all_sources[record["source"]] = all_sources.get(record["source"], 0) + 1

print(all_sources)


cambrian_eagle_keys = [
    "lvis_instruct4v_220k.json",
    "clean_llava_instruct_150k_llavar_20k.json",
    "chartqa_28k.json",
    "docvqa_39k.json",
    "random_3rd_dvqa_2325k.json",
    "tallyqa_250k.json",
    "clevr_700k.json",
    "sketchyvqa_8k.json",
    "oodvqa_8k.json",
    "allava-laion-500k.json",
    "allava-vflan-200k.json",
    "idk_11k.json",
    "laion_gpt4v_11k.json",
    "orca_math_200k.json",
    "design2code_0k.json",
    "wizardlm_143k.json",
    "mathvision_3k.json",
    "geo170k.json",
    "arxivqa_100k.json",
    "screenqa_79k.json",
    "synthdog_500k_modified.json",
    "alfworldgpt_45k.json",
    "lnqa_302k.json",
    "pathvqa_32k.json",
]


def check_sample(sample):
    conversations = sample.get("conversations", [])

    for turn in conversations:
        value = turn.get("value", "")
        image_count = value.count("<image>")

        if image_count > 1:
            return False

    return True


cambrian_eagle = []
for record in tqdm(records):
    source = record["source"]
    check_result = check_sample(record)
    if not check_result:
        print(record)
        continue
    if source in cambrian_eagle_keys:
        cambrian_eagle.append(record)
    elif "image" in record and record["image"].startswith("gqa"):
        cambrian_eagle.append(record)

print(len(cambrian_eagle))

base_path = "./cambrian"

with open(os.path.join(base_path, "cambrian_adlr_train.json"), "w") as f:
    json.dump(cambrian_eagle, f)


