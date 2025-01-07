import json

from tqdm import tqdm

json_file = "./DocReason25K/detailed_explanation.jsonl"
json_file_processed = "./DocReason25K/docreason25k_processed.jsonl"

with open(json_file) as f:
    lines = f.readlines()

records = []
for id, line in tqdm(enumerate(lines)):
    record = json.loads(line)
    new_record = dict(id=id, image=record["image"][0])
    msgs = record["messages"]
    new_msgs = []
    for msg in msgs:
        if msg["role"] == "user":
            current_message = {"from": "human", "value": msg["content"].replace("<|image|>", "<image>\n")}
        else:
            current_message = {"from": "gpt", "value": msg["content"].replace("<|image|>", "<image>\n")}
        new_msgs.append(current_message)
    new_record["conversations"] = new_msgs
    new_record["dataset_name"] = "docreason"
    records.append(new_record)
    if id % 100 == 0:
        print(records[-1])


with open(json_file_processed, "w") as f:
    json.dump(records, f)


