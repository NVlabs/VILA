import json
import os
import random
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm  # Add this import

choice_1_prompts = [
    "Inside bounding box: {}, What is the text in the bounding box?",
    "What is the textual content enclosed by the coordinates {}?",
    "Inside the defined region {}, what text is present?",
    "What is the text contained within the rectangular area defined by the points {}?",
    "Can you identify the text that falls within the bounding box {}?",
    "What is written in the image inside the box {}?",
]

choice_2_prompts = [
    "Locate the position of : '{}' in the image with a bounding box.",
    "Identify the coordinates of the text '{}' within the image and enclose it in a bounding box.",
    "Determine the spatial location of '{}' in the image and mark it with a rectangular boundary.",
    "Find the exact position of the phrase '{}' inside the image and outline it with a box.",
    "Pinpoint the location of '{}' within the image and highlight it with a bounding rectangle.",
    "Locate the exact position of the text string '{}' in the image and define a bounding box around it.",
]

# PLEASE REPLACE YOUR IMAGE FOLDER HERE.
image_root = "./LSVT/train_full_images"
# PLEASE REPLACE YOUR ANNOTATION FILE HERE.
with open("./LSVT/train_full_labels.json") as f:
    anns = json.load(f)

return_list = []

jsonl_path = os.path.join("./LSVT/", "LSVT_processed.jsonl")


def coords_list2bbox(coords_list: List[List[int]], width: int, height: int) -> str:
    left = np.clip(int(min(coords_list[0][0], coords_list[2][0]) / width * 1000), 0, 999)
    upper = np.clip(int(min(coords_list[0][1], coords_list[1][1]) / height * 1000), 0, 999)
    right = np.clip(int(max(coords_list[1][0], coords_list[3][0]) / width * 1000), 0, 999)
    bottom = np.clip(int(max(coords_list[2][1], coords_list[3][1]) / height * 1000), 0, 999)

    return f"[{left:03d},{upper:03d},{right:03d},{bottom:03d}]"


# Add a progress bar
with open(jsonl_path, "w") as jsonl_file:
    for k, v in tqdm(anns.items(), desc="Processing images", total=len(anns)):
        image_name = k + ".jpg"
        image = Image.open(os.path.join(image_root, image_name))
        w, h = image.size
        convs = []
        for v_i in v:
            if v_i["illegibility"]:
                continue
            coords = coords_list2bbox(v_i["points"], w, h)
            caption = v_i["transcription"]
            if random.random() > 0.5:
                convs.extend(
                    [
                        {
                            "from": "human",
                            "value": choice_1_prompts[random.randint(0, 4)].format(coords),
                        },
                        {"from": "gpt", "value": caption},
                    ]
                )
            else:
                convs.extend(
                    [
                        {
                            "from": "human",
                            "value": choice_2_prompts[random.randint(0, 4)].format(caption),
                        },
                        {"from": "gpt", "value": coords},
                    ]
                )
        if len(convs) > 0:
            convs[0]["value"] = "<image>\n" + convs[0]["value"]
            outputs = {"id": int(k.split("_")[-1]), "image": k + ".jpg", "conversations": convs}
            json.dump(outputs, jsonl_file)
            jsonl_file.write("\n")  # Add a newline after each JSON object

print("Processing complete.")


