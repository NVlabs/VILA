{
    "chars": [
        {"ignore": 0, "transcription": "H", "points": [25, 175, 112, 175, 112, 286, 25, 286]},
        {"ignore": 0, "transcription": "O", "points": [102, 182, 210, 182, 210, 273, 102, 273]},
        {"ignore": 0, "transcription": "K", "points": [201, 185, 293, 185, 293, 285, 201, 285]},
        {"ignore": 0, "transcription": "I", "points": [283, 180, 328, 180, 328, 288, 283, 288]},
        {"ignore": 0, "transcription": "T", "points": [367, 182, 468, 182, 468, 295, 367, 295]},
        {"ignore": 0, "transcription": "E", "points": [452, 201, 515, 201, 515, 294, 452, 294]},
        {"ignore": 0, "transcription": "A", "points": [509, 191, 619, 191, 619, 293, 509, 293]},
    ],
    "lines": [
        {"ignore": 0, "transcription": "HOKI", "points": [23, 173, 327, 180, 327, 290, 23, 283]},
        {"ignore": 0, "transcription": "TEA", "points": [368, 180, 621, 180, 621, 294, 368, 294]},
    ],
}


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
image_root = "/home/jasonlu/vlm_datasets2/ReCTS/img"

return_list = []

jsonl_path = os.path.join("/home/jasonlu/vlm_datasets2/ReCTS/", "ReCTS_processed.jsonl")


def coords_list2bbox(coords_list: List[List[int]], width: int, height: int) -> str:
    x = coords_list[0::2]
    y = coords_list[1::2]

    left = np.clip(int(min(x) / width * 1000), 0, 999)
    upper = np.clip(int(min(y) / height * 1000), 0, 999)
    right = np.clip(int(max(x) / width * 1000), 0, 999)
    bottom = np.clip(int(max(y) / height * 1000), 0, 999)

    return f"[{left:03d},{upper:03d},{right:03d},{bottom:03d}]"


# Add a progress bar
with open(jsonl_path, "w") as jsonl_file:
    image_files = [f for f in os.listdir(image_root) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

    for image_name in tqdm(image_files, desc="Processing images"):
        annotation_file = image_name.replace(".jpg", ".json")
        annotation = json.loads(open(os.path.join("./ReCTS/gt", annotation_file)).read())
        image = Image.open(os.path.join(image_root, image_name))
        w, h = image.size
        convs = []
        for v_i in annotation["chars"]:
            if v_i["ignore"]:
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
            outputs = {"id": int(image_name.split("_")[-1][:-4]), "image": image_name, "conversations": convs}
            json.dump(outputs, jsonl_file)
            jsonl_file.write("\n")  # Add a newline after each JSON object

print("Processing complete.")


