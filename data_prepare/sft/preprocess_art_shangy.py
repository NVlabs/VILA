import json
import os

from PIL import Image


def rm_binary_code(text):
    # remove binary code after ##

    if "##" in text:
        out = text.split("##")[0]
    else:
        out = text

    if out == "NoName":
        return ""
    else:
        return out


def is_image_corrupted(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Verify if it's a valid image
        return False  # If no exception, the image is not corrupted
    except OSError:
        return True  # If an exception is raised, the image is corrupted


def convert_txt_to_jsonl(input_file, output_file):

    with open(input_file, encoding="utf-8") as txt_file, open(output_file, "w", encoding="utf-8") as jsonl_file:
        id = 0
        for line in txt_file:
            line = line.strip()
            parts = line.split("|")

            image_info = {
                "author": parts[0] if len(parts) > 0 else "",
                "painting_name": rm_binary_code(parts[1]) if len(parts) > 1 else "",
                "Genre": parts[3] if len(parts) > 3 else "",
                "Style": parts[4] if len(parts) > 4 else "",
                "Path": parts[-1],
            }

            if is_image_corrupted(os.path.join(base_path, image_info["Path"])):
                print(image_info["Path"] + " is corrupted, skip.")
                continue

            record = dict(id=id, image=image_info["Path"])
            conversations = []
            if not len(image_info["painting_name"]) > 0:
                continue
            conversations.append(
                {
                    "from": "human",
                    "value": "<image>\nYou are given an image of an artwork. Please answer the following questions briefly. What is the name of the painting?",
                }
            )
            conversations.append({"from": "gpt", "value": image_info["painting_name"] + "."})
            if len(image_info["author"]) > 0:
                conversations.append({"from": "human", "value": "Who is the author of the painting?"})
                conversations.append({"from": "gpt", "value": image_info["author"] + "."})
            if len(image_info["Genre"]) > 0:
                conversations.append({"from": "human", "value": "What is the genre of the painting?"})
                conversations.append({"from": "gpt", "value": image_info["Genre"] + "."})
            if len(image_info["Style"]) > 0:
                conversations.append({"from": "human", "value": "What is the style of the painting?"})
                conversations.append({"from": "gpt", "value": image_info["Style"] + "."})

            record["conversations"] = conversations

            if os.path.exists(os.path.join(base_path, image_info["Path"])):
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                continue
            # else:
            #     print(f"File not found: {image_info['Path']}. Skipping...")

            id += 1
            if id % 100 == 0:
                print(record)


if __name__ == "__main__":
    input_file = "./label_list.csv"
    output_file = "./art500k_processed.jsonl"
    base_path = "./"
    convert_txt_to_jsonl(input_file, output_file)


