import json

from langdetect import detect


def is_english(text):
    try:
        language = detect(text)
        return language == "en"
    except:
        return False


input_json_path = "./estvqa/estvqa.json"
jsonl_path = "./estvqa/ESTVQA_processed.jsonl"

with open(input_json_path) as f:
    data = json.load(f)

with open(jsonl_path, "w") as jsonl_file:
    for item in data:
        image_name = item["image"]
        convs = []
        if is_english(item["annotation"][0]["question"]) and is_english(item["annotation"][0]["answer"]):
            for annotation in item["annotation"]:
                convs.extend(
                    [{"from": "human", "value": annotation["question"]}, {"from": "gpt", "value": annotation["answer"]}]
                )

            convs[0]["value"] = "<image>\n" + convs[0]["value"]
            output = {"id": item["id"], "image": image_name, "conversations": convs}
            json.dump(output, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")

print("Processing complete.")


