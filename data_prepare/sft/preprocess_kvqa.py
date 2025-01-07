import json
import os

base = "./kvqa/raw"
info_file = os.path.join(base, "dataset.json")
processed_info_file = os.path.join(base, "kvqa_processed.json")

with open(info_file) as f:
    records = json.load(f)


id = 0
new_records = []
for key in records:
    record = records[key]
    questions = record["Questions"]
    answers = record["Answers"]
    questions[0] = "<image>\n" + questions[0]
    new_record = dict(id=id, dataset_name="kvqa", image=record["imgPath"])
    conversations = []
    assert len(questions) == len(answers)
    for ques, ans in zip(questions, answers):
        if type(ans) is not str:
            ans = str(ans)
        assert type(ques) == str, type(ans) == str
        conversations.append(
            {
                "from": "human",
                "value": ques,
            }
        )
        conversations.append(
            {
                "from": "gpt",
                "value": ans,
            }
        )
    new_record["conversations"] = conversations
    new_records.append(new_record)
    id += 1
    if id % 100 == 0:
        print(new_records[-1])

with open(processed_info_file, "w") as f:
    json.dump(new_records, f)

print(len(new_records))


