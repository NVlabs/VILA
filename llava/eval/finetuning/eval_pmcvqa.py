import argparse
import json


def load_predictions(predictions_file):
    predictions = []
    with open(predictions_file) as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def compute_acc(predictions):
    correct = 0
    incorrect = 0

    for pred in predictions:
        unique_id = f"{pred['question_id']}"
        text = pred["text"] if len(pred["text"]) == 1 else "FAILED"
        gt_captions = pred["gt_caption"]

        # text in A, B, C, D
        if text == gt_captions:
            correct += 1
        else:
            incorrect += 1

    acc = correct / (correct + incorrect)
    return acc


def main(predictions_file):
    predictions = load_predictions(predictions_file)
    acc = compute_acc(predictions)
    print(f"Accuracy: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", help="prediction path")
    args = parser.parse_args()
    main(args.pred_path)
