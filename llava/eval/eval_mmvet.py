# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import openai
import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--results_file", type=str)

args = parser.parse_args()

try:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    gpt_model = "gpt-4"
except:
    client = openai
    gpt_model = "gpt-4-0613"

prompt = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
"""

# load metadata
# Download mm-vet.zip and `unzip mm-vet.zip` and change the path below
# mmvet_path = "/tmp/mmvet/mm-vet"
mmvet_path = "playground/data/eval/mm-vet"
use_sub_set = False
decimal_places = 1  # number of decimal places to round to


if use_sub_set:
    bard_set_file = os.path.join(mmvet_path, "bard_set.json")
    with open(bard_set_file) as f:
        sub_set = json.load(f)
    sub_set_name = "bardset"
    sub_set_name = sub_set_name + "_"
else:
    sub_set = None
    sub_set_name = ""

mmvet_metadata = os.path.join(mmvet_path, "mm-vet.json")
with open(mmvet_metadata) as f:
    data = json.load(f)


counter = Counter()
cap_set_list = []
cap_set_counter = []
len_data = 0
for id, value in data.items():
    if sub_set is not None and id not in sub_set:
        continue
    question = value["question"]
    answer = value["answer"]
    cap = value["capability"]
    cap = set(cap)
    counter.update(cap)
    if cap not in cap_set_list:
        cap_set_list.append(cap)
        cap_set_counter.append(1)
    else:
        cap_set_counter[cap_set_list.index(cap)] += 1

    len_data += 1

sorted_list = counter.most_common()
columns = [k for k, v in sorted_list]
columns.append("total")
columns.append("std")
columns.append("runs")
df = pd.DataFrame(columns=columns)


cap_set_sorted_indices = np.argsort(-np.array(cap_set_counter))
new_cap_set_list = []
new_cap_set_counter = []
for index in cap_set_sorted_indices:
    new_cap_set_list.append(cap_set_list[index])
    new_cap_set_counter.append(cap_set_counter[index])

cap_set_list = new_cap_set_list
cap_set_counter = new_cap_set_counter
cap_set_names = ["_".join(list(cap_set)) for cap_set in cap_set_list]

columns2 = cap_set_names
columns2.append("total")
columns2.append("std")
columns2.append("runs")
df2 = pd.DataFrame(columns=columns2)


###### change your model name ######
# model = "llava_llama2_13b_chat"
num_run = 1  # we set it as 5 in the paper
model_results_file = args.results_file
model = args.results_file.split("/")[-1].replace(".json", "")

# grade results for each sample to svae
grade_file = args.results_file.replace(".json", "-grade.json")

# score results regarding capabilities/capability integration to save
cap_score_file = args.results_file.replace(".json", f"-cap-score-{num_run}runs.csv")
cap_int_score_file = f"{model}_{sub_set_name}{gpt_model}-cap-int-score-{num_run}runs.csv"
cap_int_score_file = args.results_file.replace(".json", f"-cap-int-score-{num_run}runs.csv")


with open(model_results_file) as f:
    results = json.load(f)
if os.path.exists(grade_file):
    with open(grade_file) as f:
        grade_results = json.load(f)
else:
    grade_results = {}


def need_more_runs():
    need_more_runs = False
    if len(grade_results) > 0:
        for k, v in grade_results.items():
            if len(v["score"]) < num_run:
                need_more_runs = True
                break
    return need_more_runs or len(grade_results) < len_data


while need_more_runs():
    for j in range(num_run):
        print(f"eval run {j}")
        for id, line in tqdm(data.items()):
            if sub_set is not None and id not in sub_set:
                continue
            if id in grade_results and len(grade_results[id]["score"]) >= (j + 1):
                continue

            model_pred = results[id]

            question = (
                prompt
                + "\n"
                + " | ".join(
                    [
                        line["question"],
                        line["answer"].replace("<AND>", " <AND> ").replace("<OR>", " <OR> "),
                        model_pred,
                        "",
                    ]
                )
            )
            messages = [
                {"role": "user", "content": question},
            ]

            if id not in grade_results:
                sample_grade = {"model": [], "content": [], "score": []}
            else:
                sample_grade = grade_results[id]

            grade_sample_run_complete = False
            temperature = 0.0

            while not grade_sample_run_complete:
                try:
                    response = client.chat.completions.create(
                        model=gpt_model, max_tokens=3, temperature=temperature, messages=messages
                    )
                    content = response.choices[0].message.content
                    flag = True
                    try_time = 1
                    while flag:
                        try:
                            content = content.split(" ")[0].strip()
                            score = float(content)
                            if score > 1.0 or score < 0.0:
                                assert False
                            flag = False
                        except:
                            question = (
                                prompt
                                + "\n"
                                + " | ".join(
                                    [
                                        line["question"],
                                        line["answer"].replace("<AND>", " <AND> ").replace("<OR>", " <OR> "),
                                        model_pred,
                                        "",
                                    ]
                                )
                                + "\nPredict the correctness of the answer (digit): "
                            )
                            messages = [
                                {"role": "user", "content": question},
                            ]
                            response = client.chat.completions.create(
                                model=gpt_model, max_tokens=3, temperature=temperature, messages=messages
                            )
                            content = response.choices[0].message.content
                            try_time += 1
                            temperature += 0.5
                            print(f"{id} try {try_time} times")
                            print(content)
                            if try_time > 5:
                                score = 0.0
                                flag = False
                    grade_sample_run_complete = True
                except:
                    # gpt4 may have token rate limit
                    print("sleep 30s")
                    time.sleep(30)

            if len(sample_grade["model"]) >= j + 1:
                sample_grade["model"][j] = response.model
                sample_grade["content"][j] = content
                sample_grade["score"][j] = score
            else:
                sample_grade["model"].append(response.model)
                sample_grade["content"].append(content)
                sample_grade["score"].append(score)
            grade_results[id] = sample_grade

            with open(grade_file, "w") as f:
                json.dump(grade_results, f, indent=4)


assert not need_more_runs()
cap_socres = {k: [0.0] * num_run for k in columns[:-2]}
counter["total"] = len_data

cap_socres2 = {k: [0.0] * num_run for k in columns2[:-2]}
counter2 = {columns2[i]: cap_set_counter[i] for i in range(len(cap_set_counter))}
counter2["total"] = len_data

for k, v in grade_results.items():
    if sub_set is not None and k not in sub_set:
        continue
    for i in range(num_run):
        score = v["score"][i]
        caps = set(data[k]["capability"])
        for c in caps:
            cap_socres[c][i] += score

        cap_socres["total"][i] += score

        index = cap_set_list.index(caps)
        cap_socres2[cap_set_names[index]][i] += score
        cap_socres2["total"][i] += score

for k, v in cap_socres.items():
    cap_socres[k] = np.array(v) / counter[k] * 100


std = round(cap_socres["total"].std(), decimal_places)
total_copy = cap_socres["total"].copy()
runs = str(list(np.round(total_copy, decimal_places)))

for k, v in cap_socres.items():
    cap_socres[k] = round(v.mean(), decimal_places)

cap_socres["std"] = std
cap_socres["runs"] = runs
df.loc[model] = cap_socres


for k, v in cap_socres2.items():
    cap_socres2[k] = round(np.mean(np.array(v) / counter2[k] * 100), decimal_places)
cap_socres2["std"] = std
cap_socres2["runs"] = runs
df2.loc[model] = cap_socres2

df.to_csv(cap_score_file)
df2.to_csv(cap_int_score_file)

print(df)
print(df2)
