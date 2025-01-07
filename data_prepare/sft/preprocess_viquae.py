import base64
import json
import os
import pickle
from io import BytesIO

from PIL import Image


def base64_to_pil_image(base64_string):
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)

    # Create a BytesIO object from the decoded bytes
    image_bytes = BytesIO(image_data)

    # Open the image with PIL
    image = Image.open(image_bytes)

    return image


base_path = "./viquae"
source_path = "./vflan_no_video/vqa_viquae_train.pkl"
os.makedirs(base_path, exist_ok=True)
os.makedirs(os.path.join(base_path, "images"), exist_ok=True)

with open(source_path, "rb") as f:
    records = pickle.load(f)


new_records = []
for id, record in enumerate(records):
    img = base64_to_pil_image(record["image:"][0])
    img_path = os.path.join(base_path, "images", "%d.png" % id)
    img.save(img_path)
    img_path = os.path.join("images", "%d.png" % id)
    question = record["question"]
    answer = record["answer:"]
    conversations = []
    conversations.append({"from": "human", "value": question})
    conversations.append({"from": "gpt", "value": answer})
    new_record = dict(id=id, image=img_path, conversations=conversations, dataset_name="viquae")
    new_records.append(new_record)

with open(os.path.join(base_path, "viquae_processed.json"), "w") as f:
    json.dump(new_records, f)


