# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import os


def main(args):
    with open(os.path.join(args.output_dir, "merge.json")) as file:
        ans_file = file.readlines()

    data = [json.loads(line) for line in ans_file]
    extracted_data = []
    for item in data:
        q_uid, answer = item["id"], item["pred"]
        # kaggle eval doesn't accept null values
        if str(answer) not in ["0", "1", "2", "3", "4"]:
            answer = 0
        extracted_data.append({"q_uid": q_uid, "answer": answer})

    output_csv = os.path.join(args.output_dir, "merge.csv")
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["q_uid", "answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(extracted_data)

    print(f"Data has been written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
