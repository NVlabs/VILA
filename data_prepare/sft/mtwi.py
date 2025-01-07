import json
import os
import random

from PIL import Image
from tqdm import tqdm

folder_path = "./mtwi_train/image_train"
output_base_path = "./mtwi_train/"

choice_0_prompts = [
    "<image>\n Identify the text in the image with the bounding box and the text content. The bounding box needs to be in the format of [x,y,x,y] where x and y both range from 0 to 999.",
    "<image>\n Analyze the provided image and extract the text content along with its corresponding bounding box coordinates. The bounding box should be represented as a four-element list [x1,y1,x2,y2], where x1 and y1 denote the top-left corner coordinates, and x2 and y2 represent the bottom-right corner coordinates. All coordinates should range from 0 to 999.",
    "<image>\n Identify and extract the text elements present within the image. For each text element, provide the corresponding bounding box coordinates in the format [x1,y1,x2,y2], where x1 and y1 represent the top-left corner, and x2 and y2 represent the bottom-right corner. All coordinates should be within the range 0 to 999.",
    "<image>\n Analyze the visual content of the image to identify and extract the text. Each extracted text element should be accompanied by its bounding box coordinates in the format [x1,y1,x2,y2], where x1,y1,x2,and y2 are integers between 0 and 999 representing the top-left and bottom-right corners of the bounding box.",
    "<image>\n Examine the image and extract the textual information within it. For each text segment, provide the bounding box coordinates in the format [x1,y1,x2,y2], where (x1,y1) and (x2,y2) represent the top-left and bottom-right corners, respectively. Coordinates should fall within the range 0 to 999.",
]
choice_1_prompts = [
    "Inside bounding box: [{},{},{},{}], What is the text in the bounding box?",
    "What is the textual content enclosed by the coordinates [{},{},{},{}]?",
    "Inside the defined region [{},{},{},{}], what text is present?",
    "What is the text contained within the rectangular area defined by the points [{},{},{},{}]?",
    "Can you identify the text that falls within the bounding box [{},{},{},{}]?",
]

choice_2_prompts = [
    "<image>\n Identify the texts in the image.",
    "<image>\n Extract the texts from the image.",
    "<image>\n What are the texts in the image?",
    "<image>\n Analyze the image and extract the textual content.",
    "<image>\n Identify and extract all text from the image.",
    "<image>\n Identify and list the text elements present in the image.",
]


def clip(x):
    if x < 0:
        return 0
    elif x > 999:
        return 999
    else:
        return x


jsonl_path = os.path.join(output_base_path, "mtwi_processed.jsonl")

# Get the total number of files to process
total_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# Open the jsonl file for writing
with open(jsonl_path, "w") as jsonl_file:
    # Wrap the file processing loop with tqdm
    for filename in tqdm(os.listdir(folder_path), total=total_files, desc="Processing files"):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            img = Image.open(file_path)
        file_path = file_path.replace("image_train", "txt_train")
        file_path = file_path[:-3] + "txt"

        answer = ""
        answers = []
        bbox = []
        with open(file_path) as file:
            for content in file:
                if "###" in content:
                    continue
                try:
                    content_list = content.split(",")
                    text = content_list[-1].strip()
                    content_list = content_list[:-1]
                    x = content_list[0::2]
                    x = [clip(int(float(i) * 1000 / img.width)) for i in x]
                    x_min = min(x)
                    x_max = max(x)
                    y = content_list[1::2]
                    y = [clip(int(float(i) * 1000 / img.height)) for i in y]
                    y_min = min(y)
                    y_max = max(y)
                    answer += (
                        f"Inside bounding box: [{x_min:03d},{y_min:03d},{x_max:03d},{y_max:03d}], The text is: {text}\n"
                    )
                    bbox.append([f"{x_min:03d}", f"{y_min:03d}", f"{x_max:03d}", f"{y_max:03d}"])
                    answers.append(text)
                except:
                    continue

        conversation_choice = random.choice([0, 1, 2])
        if conversation_choice == 0:
            prompt = random.choice(choice_0_prompts)

            outputs = {
                "id": filename,
                "image": filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt,
                    },
                    {
                        "from": "gpt",
                        "value": answer,
                    },
                ],
            }
        elif conversation_choice == 1:
            conversations = []
            for bbox_item, text_item in zip(bbox, answers):
                conversations.extend(
                    [
                        {
                            "from": "human",
                            "value": random.choice(choice_1_prompts).format(
                                bbox_item[0], bbox_item[1], bbox_item[2], bbox_item[3]
                            ),
                        },
                        {"from": "gpt", "value": text_item},
                    ]
                )
            try:
                conversations[0]["value"] = "<image>\n" + conversations[0]["value"]
            except:
                continue

            outputs = {
                "id": filename,
                "image": filename,
                "conversations": conversations,
            }
        elif conversation_choice == 2:
            prompt = random.choice(choice_2_prompts)

            answer_test = " ".join(answers)
            outputs = {
                "id": filename,
                "image": filename,
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt,
                    },
                    {
                        "from": "gpt",
                        "value": answer_test,
                    },
                ],
            }
        json.dump(outputs, jsonl_file)
        jsonl_file.write("\n")  # Add a newline after each JSON object

print("Processing complete.")


