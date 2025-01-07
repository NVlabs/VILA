import json
import os
import random
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm  # Add this import

entity_dict = {
    "SS": "Serving Size",
    "CE-PS": "Calories/Energy of per serving",
    "CE-P1": "Calories/Energy of per 100g/ml",
    "CE-D": "Calories/Energy of % Daily Value",
    "CE-PP": "Calories/Energy of per package",
    "TF-PS": "Total Fat of per serving",
    "TF-P1": "Total Fat of per 100g/ml",
    "TF-D": "Total Fat of % Daily Value",
    "TF-PP": "Total Fat of per package",
    "SO-PS": "Sodium of per serving",
    "SO-P1": "Sodium of per 100g/ml",
    "SO-D": "Sodium of % Daily Value",
    "SO-PP": "Sodium of per package",
    "CAR-PS": "Total Carbohydrate of per serving",
    "CAR-P1": "Total Carbohydrate of per 100g/ml",
    "CAR-D": "Total Carbohydrate of % Daily Value",
    "CAR-PP": "Total Carbohydrate of per package",
    "PRO-PS": "Protein of per serving",
    "PRO-P1": "Protein of per 100g/ml",
    "PRO-D": "Protein of % Daily Value",
    "PRO-PP": "Protein of per package",
}

# PLEASE REPLACE YOUR IMAGE FOLDER HERE.
image_root = "./poie/nfv5/nfv5_3125/image_files"
# PLEASE REPLACE YOUR ANNOTATION FILE HERE.
anns = []
with open("./poie/nfv5/nfv5_3125/train.txt") as f:
    for line in f:
        anns.append(json.loads(line.strip()))

return_list = []

jsonl_path = os.path.join("./poie/", "POIE_processed.jsonl")


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
    for data in tqdm(anns, desc="Processing images", total=len(anns)):
        w = data["width"]
        h = data["height"]
        convs = []

        if "entity_dict" in data:
            for k, v in data["entity_dict"].items():
                convs.extend(
                    [
                        {
                            "from": "human",
                            "value": f"what is the value for {entity_dict[k]}? Answer this question using the text in the image directly.",
                        },
                        {"from": "gpt", "value": v},
                    ]
                )
            if len(convs) > 0:
                convs[0]["value"] = "<image>\n" + convs[0]["value"]
                outputs = {
                    "id": data["file_name"].split("/")[-1].split(".")[0],
                    "image": data["file_name"],
                    "conversations": convs,
                }
                json.dump(outputs, jsonl_file)
                jsonl_file.write("\n")  # Add a newline after each JSON object

print("Processing complete.")


