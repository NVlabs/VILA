import argparse
import json
import os
import re

_LARGEST_COUNT = 15


def load_jsonl(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _number_word_to_numeral(text: str) -> str:
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }

    # Check if text is a number word
    if text.lower() in number_words:
        return number_words[text.lower()]

    # Check if text is already a numeral
    if text.isdigit():
        return text

    # Extract numeric part from the text
    match = re.search(r"\d+", text)
    if match:
        return match.group(0)

    # Extract number word from the text
    words = text.lower().split()
    for word in words:
        if word in number_words:
            return number_words[word]

    # If no numeric part is found, return the original text (can be handled later as needed)
    return text


def eval(predictions, annotations):
    accuracies_by_type = {"all": [], "simple": [], "complex": []}
    accuracies_by_type.update({f"count_{i}": [] for i in range(_LARGEST_COUNT + 1)})

    annotation_dict = {entry["question_id"]: entry for entry in annotations}

    for prediction in predictions:
        question_id = prediction["question_id"]
        answer = prediction["text"].strip()
        answer = _number_word_to_numeral(answer)

        if question_id in annotation_dict:
            gt_entry = annotation_dict[question_id]
            gt = _number_word_to_numeral(str(gt_entry["answer"]))

            accuracies_by_type["all"].append(float(answer == gt))

            if "issimple" in gt_entry:
                if gt_entry["issimple"] == 1:
                    accuracies_by_type["simple"].append(float(answer == gt))
                elif gt_entry["issimple"] == 0:
                    accuracies_by_type["complex"].append(float(answer == gt))

            if f"count_{gt}" in accuracies_by_type:
                accuracies_by_type[f"count_{gt}"].append(float(answer == gt))

    sum_accs = {k: sum(v) for k, v in accuracies_by_type.items()}
    num_accs = {k: len(v) for k, v in accuracies_by_type.items()}

    metrics = {}
    if num_accs["all"]:
        metrics["acc"] = sum_accs["all"] / num_accs["all"]
        metrics["num"] = num_accs["all"]

    for key in sum_accs.keys():
        if key != "all" and num_accs[key]:
            metrics[f"acc/{key}"] = sum_accs[key] / num_accs[key]
            metrics[f"num/{key}"] = num_accs[key]

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_file", help="prediction path")
    parser.add_argument("--pred_path", help="prediction path")
    args = parser.parse_args()

    predictions = load_jsonl(args.pred_path)
    annotations = load_jsonl(args.question_file)
    metrics = eval(predictions, annotations)
    for key, value in metrics.items():
        print(f"{key}: {value}")
