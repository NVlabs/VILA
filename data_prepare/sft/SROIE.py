import json
import os

from tqdm import tqdm

# PLEASE REPLACE YOUR IMAGE FOLDER HERE.
image_root = "./sroie/train"
# PLEASE REPLACE YOUR ANNOTATION FILE HERE.

question_dict = {
    "company": "what is the name of the company that issued this receipt? Answer this question using the text in the image directly.",
    "address": "where was this receipt issued? Answer this question using the text in the image directly.",
    "date": "when was this receipt issued? Answer this question using the text in the image directly.",
    "total": "what is the total amount of this receipt? Answer this question using the text in the image directly.",
}
return_list = []

jsonl_path = os.path.join("./sroie", "SROIE_processed.jsonl")

# Get list of image files
image_files = [
    f for f in os.listdir(image_root) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
]
images = []

# Open all images in image_root with progress bar
with open(jsonl_path, "w") as jsonl_file:
    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        image_path = os.path.join(image_root, image_file)
        annotation_path = os.path.join(
            image_root, image_file.replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt")
        )

        try:
            with open(annotation_path) as f:
                try:
                    annotation_data = json.load(f)
                except json.JSONDecodeError as json_error:
                    print(f"JSON decoding error in {annotation_path}: {json_error}")
                    continue  # Skip to the next image file
        except FileNotFoundError:
            print(f"Annotation file not found for {image_file}")
            continue  # Skip to the next image file
        except Exception as e:
            print(f"Error opening {annotation_path}: {e}")
            continue  # Skip to the next image file

        for k, v in annotation_data.items():
            convs = [
                (
                    [
                        {"from": "human", "value": "<image>\n" + question_dict[k]},
                        {"from": "gpt", "value": v},
                    ]
                )
            ]

            outputs = {
                "id": image_file[:-3] + k,
                "image": image_file,
                "conversations": convs,
            }
            json.dump(outputs, jsonl_file)
            jsonl_file.write("\n")  # Add a newline after each JSON object


# Now 'images' contains all the opened images from the image_root directory
print(f"Successfully opened {len(images)} out of {len(image_files)} images.")


