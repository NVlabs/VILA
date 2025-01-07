import json

json_file = "./MetaMathQA/MetaMathQA-395K.json"
json_file_processed = "./MetaMathQA/MetaMathQA-395K_processed.json"

with open(json_file) as f:
    data = json.load(f)

records = []
for id, record in enumerate(data):
    new_record = dict(id=id, dataset_name="metamathqa", question_type=record["type"])
    conversations = []
    conversations.append(
        {
            "from": "human",
            "value": record["query"],
        }
    )
    conversations.append({"from": "gpt", "value": record["response"]})
    new_record["conversations"] = conversations
    records.append(new_record)

    if id % 100 == 0:
        print(records[-1])

with open(json_file_processed, "w") as f:
    json.dump(records, f)


