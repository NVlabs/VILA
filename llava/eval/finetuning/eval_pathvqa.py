import argparse
import json
import math
import re
from collections import defaultdict

import num2words
from pycocoevalcap.bleu.bleu import Bleu


def split_sentence(sentence, n):
    words = defaultdict(int)
    #  e.g., convert 6 to six
    sentence = re.sub(r"\b(\d+)\b", lambda x: num2words.num2words(int(x.group(0))), sentence)
    tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i : i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words


# BLEU
def calculate_bleu(weights, pn, n, bp):
    sum_wlogp = 0
    for i in range(n):
        if pn[i] != 0:
            sum_wlogp += float(weights[i]) * math.log(pn[i])
    bleu_result = bp * math.exp(sum_wlogp)
    return bleu_result


# Exact match
def calculate_exactmatch(candidate, reference):
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]

    if total == 0:
        return "0 (warning: length of candidate's words is 0)"
    else:
        return count / total


# F1
def calculate_f1score(candidate, reference):
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)

    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if len(candidate_words) == 0:
        return "0 (warning: length of candidate's words is 0)"
    elif len(reference_words) == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0
        else:
            return 2 * precision * recall / (precision + recall)


def compute_bleu(gts, res):
    scorer = Bleu(4)
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def load_predictions(predictions_file):
    yes_no_predictions = []
    predictions = []

    with open(predictions_file) as f:
        for line in f:
            prediction = json.loads(line)
            if prediction["gt_caption"] == "yes" or prediction["gt_caption"] == "no":
                yes_no_predictions.append(prediction)
            else:
                predictions.append(json.loads(line))
    return yes_no_predictions, predictions


def compute_acc(predictions):

    gts = {}
    res = {}
    f1 = 0
    exact_match = 0
    for pred in predictions:
        unique_id = f"{pred['question_id']}"
        text = pred["text"]
        if text == "":
            text = "dummy text"
            print(f"Empty prediction for unique ID: {unique_id}")
        gt_captions = pred["gt_caption"]

        gts[unique_id] = [gt_captions]
        res[unique_id] = [text]

        # f1
        if isinstance(calculate_f1score(text, gt_captions), str):
            print(f"Warning: {unique_id}: {text}, {gt_captions}")
        f1 += calculate_f1score(text, gt_captions)
        # exact match
        exact_match += calculate_exactmatch(text, gt_captions)

    f1 /= len(predictions)
    exact_match /= len(predictions)

    return gts, res, f1, exact_match


def main(predictions_file):
    yes_no_predictions, predictions = load_predictions(predictions_file)
    gts, res, f1, exact_match = compute_acc(yes_no_predictions)
    print(f"Yes or No ({len(yes_no_predictions)}), F1: {f1}, Exact Match: {exact_match}")

    gts, res, f1, exact_match = compute_acc(predictions)
    print(f"Free-form ({len(predictions)}), F1: {f1}, Exact Match: {exact_match}")

    # BLEU
    score, scores = compute_bleu(gts, res)
    print(f"Free-form BLEU-1 score: {score[0]}")  # BLEU-1 score
    print(f"Free-form BLEU-2 score: {score[1]}")  # BLEU-2 score
    print(f"Free-form BLEU-3 score: {score[2]}")  # BLEU-3 score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", help="prediction path")
    args = parser.parse_args()
    main(args.pred_path)
