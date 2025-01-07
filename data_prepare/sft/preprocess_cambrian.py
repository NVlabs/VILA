import json

from tqdm import tqdm

data_path = "./Cambrian7M_withsystemprompt.json"

all_sources = dict()

with open(data_path) as f:
    records = json.load(f)

for record in records:
    all_sources[record["source"]] = all_sources.get(record["source"], 0) + 1

print(all_sources)


cambrian_1375k_keys = [
    "lvis_instruct4v_220k.json",
    "sketchyvqa_8k.json",
    "oodvqa_8k.json",
    "idk_11k.json",
    "q-instruct_200k.json",
    "qalign_200k.json",
    "arxivqa_100k.json",
    "screenqa_79k.json",
    "scienceqa_12k.json",
    "alfworldgpt_45k.json",
    "filtered_data_engine_161k.json",
    "lnqa_302k.json",
    "pathvqa_32k.json",
]

cambrian_doc_1275k_keys = [
    "random_3rd_dvqa_2325k.json",
    "synthdog_500k_modified.json",
]


def check_sample(sample):
    conversations = sample.get("conversations", [])

    for turn in conversations:
        value = turn.get("value", "")
        image_count = value.count("<image>")

        if image_count > 1:
            return False

    return True


cambrian_1375k = []
cambrian_doc_1275k = []
for record in tqdm(records):
    source = record["source"]
    check_result = check_sample(record)
    if not check_result:
        print(record)
        continue
    if source in cambrian_1375k_keys:
        cambrian_1375k.append(record)
    elif source in cambrian_doc_1275k_keys:
        cambrian_doc_1275k.append(record)

print(len(cambrian_1375k), len(cambrian_doc_1275k))

with open("cambrian_1375k.json", "w") as f:
    json.dump(cambrian_1375k, f)

with open("cambrian_doc_1275k.json", "w") as f:
    json.dump(cambrian_doc_1275k, f)


