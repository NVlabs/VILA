import argparse
import json
from typing import Optional


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem["annotation"], str):
            elem["annotation"] = [elem["annotation"]]
        score = max([relaxed_correctness(elem["answer"].strip(), ann) for ann in elem["annotation"]])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem["annotation"], str):
            elem["annotation"] = [elem["annotation"]]
        score = max(
            [(1.0 if (elem["answer"].strip().lower() == ann.strip().lower()) else 0.0) for ann in elem["annotation"]]
        )
        scores.append(score)
    return sum(scores) / len(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    args = parser.parse_args()

    outputs = [json.loads(q) for q in open(args.answers_file)]

    if args.metric == "relaxed_accuracy":
        print("Relaxed accuracy:", evaluate_relaxed_accuracy(outputs) * 100)
    elif args.metric == "accuracy":
        print("Exact match accuracy:", evaluate_exact_match_accuracy(outputs) * 100)
    else:
        raise ValueError(f"Unknown metric: {args.metric}")
