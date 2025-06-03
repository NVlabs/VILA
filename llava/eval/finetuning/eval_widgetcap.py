import argparse
import json
import os

from pycocoevalcap.cider.cider import Cider
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
        unique_id = f"{pred['img_id']}_{pred['node_id']}"
        text = pred["text"]
        gt_captions = pred["gt_caption"]

        if unique_id not in gts:
            gts[unique_id] = gt_captions
        else:
            gts[unique_id].extend(gt_captions)

        res[unique_id] = [text]

    return gts, res


def compute_cider(gts, res):
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def main(predictions_file):
    predictions = load_predictions(predictions_file)

    print("Preparing references and predictions...")
    gts, res = prepare_references(predictions)

    print("Computing CIDEr score...")
    overall_score, scores = compute_cider(gts, res)

    for unique_id, score in zip(gts.keys(), scores):
        print(f"Unique ID: {unique_id}, CIDEr score: {score}")
    print(f"Overall CIDEr score: {overall_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", help="prediction path")
    args = parser.parse_args()
    main(args.pred_path)
