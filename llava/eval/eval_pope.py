# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import json
import os


def eval_pope(answers, label_file):
    outputs = []
    for answer in answers:
        text = answer["text"]

        # Only keep the first sentence
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "")
        words = text.split(" ")
        if "No" in words or "not" in words or "no" in words:
            outputs.append((answer["question_id"], "no"))
        else:
            outputs.append((answer["question_id"], "yes"))
    outputs = sorted(outputs)

    targets = [json.loads(line)["label"] for line in open(label_file)]

    pos = "yes"
    neg = "no"

    TP, TN, FP, FN = 0, 0, 0, 0
    for (_, pred), label in zip(outputs, targets):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t")
    print(f"{TP}\t{FP}\t{TN}\t{FN}")

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")
    print(f"{f1:.3f}, {acc:.3f}, {precision:.3f}, {recall:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question["question_id"]: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    for file in os.listdir(args.annotation_dir):
        assert file.startswith("coco_pope_")
        assert file.endswith(".json")
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x["question_id"]]["category"] == category]
        print(f"Category: {category}, # samples: {len(cur_answers)}")
        eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
        print("====================================")
