import glob
import io
import json
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Path to the directory containing the Parquet files
base_path = "./unichart-qa-data/data/"
output_base_path = "./unichart-qa-data/"  # Change this to your desired output path

# Create output directories if they don't exist
os.makedirs(os.path.join(output_base_path, "images"), exist_ok=True)

# Use glob to get all matching Parquet files
parquet_files = glob.glob(base_path + "train-*.parquet")

# Open a single JSONL file for writing
jsonl_path = os.path.join(output_base_path, "unichart-qa_processed.jsonl")
with open(jsonl_path, "w") as jsonl_file:
    # Get total number of rows across all files
    total_rows = sum(len(pd.read_parquet(file)) for file in parquet_files)

    # Create a progress bar
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        # Process each Parquet file
        for file in parquet_files:
            df = pd.read_parquet(file)

            for _, row in df.iterrows():
                # Save image
                img = Image.open(io.BytesIO(row["image"]))
                img_path = os.path.join(output_base_path, "images", f"{row['imgname']}")
                img.save(img_path)

                # Prepare JSON data
                json_data = {
                    "id": row["sample_id"],
                    "image": row["imgname"],
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n {row['query']}",
                        },
                        {
                            "from": "gpt",
                            "value": row["label"],
                        },
                    ],
                }

                # Write JSON data to JSONL file
                json.dump(json_data, jsonl_file)
                jsonl_file.write("\n")  # Add a newline after each JSON object

                # Update progress bar
                pbar.update(1)

print("Processing complete.")


