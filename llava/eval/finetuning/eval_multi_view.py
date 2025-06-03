import argparse
import json
import os

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from rouge import Rouge
from tqdm import tqdm


def load_predictions(predictions_file):
    predictions = []
    with open(predictions_file) as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def prepare_references(predictions):
    gts = {}
    res = {}

    for pred in predictions:
        unique_id = f"{pred['question_id']}"
        text = pred["text"]
        gt_captions = pred["gt_caption"]

        if unique_id not in gts:
            gts[unique_id] = [gt_captions]
        else:
            gts[unique_id].extend(gt_captions)

        res[unique_id] = [text]

    return gts, res


def compute_cider(gts, res):
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def compute_bleu(gts, res):
    scorer = Bleu(4)
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def compute_rouge(gts, res):
    rouge = Rouge()
    rouge_scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

    for unique_id in gts:
        gt_text = " ".join(gts[unique_id])  # Join ground truth captions
        res_text = " ".join(res[unique_id])  # Join predicted captions
        if res_text == "":
            print(f"Empty prediction for unique ID: {unique_id}")
            res_text = "dummy text"
        score = rouge.get_scores(res_text, gt_text)[0]
        rouge_scores["rouge-1"].append(score["rouge-1"]["f"])
        rouge_scores["rouge-2"].append(score["rouge-2"]["f"])
        rouge_scores["rouge-l"].append(score["rouge-l"]["f"])

    return rouge_scores


def main(predictions_file):
    predictions = load_predictions(predictions_file)

    print("Preparing references and predictions...")
    gts, res = prepare_references(predictions)

    print("Computing CIDEr score...")
    overall_cider_score, cider_scores = compute_cider(gts, res)

    print("Computing BLEU score...")
    overall_bleu_score, bleu_scores = compute_bleu(gts, res)

    print("Computing ROUGE score...")
    rouge_scores = compute_rouge(gts, res)

    # for unique_id, score in zip(gts.keys(), scores):
    #     print(f"Unique ID: {unique_id}, CIDEr score: {score}")
    print(f"Overall CIDEr score: {overall_cider_score}")
    print(f"Overall BLEU-4 score: {overall_bleu_score[0]}")  # BLEU-1 score
    print(f"Overall BLEU-4 score: {overall_bleu_score[1]}")  # BLEU-2 score
    print(f"Overall BLEU-4 score: {overall_bleu_score[2]}")  # BLEU-3 score
    print(f"Overall BLEU-4 score: {overall_bleu_score[3]}")  # BLEU-4 score
    print(f"Overall ROUGE-1 score (F1): {sum(rouge_scores['rouge-1']) / len(rouge_scores['rouge-1'])}")
    print(f"Overall ROUGE-2 score (F1): {sum(rouge_scores['rouge-2']) / len(rouge_scores['rouge-2'])}")
    print(f"Overall ROUGE-L score (F1): {sum(rouge_scores['rouge-l']) / len(rouge_scores['rouge-l'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", help="prediction path")
    args = parser.parse_args()
    main(args.pred_path)
