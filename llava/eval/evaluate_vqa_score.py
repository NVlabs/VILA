import argparse
import json
import time
from typing import Optional

ds_collections = {
    'docvqa_test': {
        'test': './playground/data/eval/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'test': './playground/data/eval/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'test': './playground/data/eval/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_val': {
        'test': './playground/data/eval/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'test': './playground/data/eval/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'test': './playground/data/eval/ai2d/test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    }
}

# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
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
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()

    print(f"Evaluating {args.dataset} ...")
    outputs = [json.loads(q) for q in open(args.answers_file, "r")]

    if ds_collections[args.dataset]['metric'] == 'relaxed_accuracy':
        print({
            'relaxed_accuracy': evaluate_relaxed_accuracy(outputs)
        })
    elif ds_collections[args.dataset]['metric'] == 'accuracy':
        if 'gqa' in args.dataset:
            for entry in outputs:
                response = entry['answer']
                response = response.strip().split('.')[0].split(
                    ',')[0].split('!')[0].lower()
                if 'is ' in response:
                    response = response.split('is ')[1]
                if 'are ' in response:
                    response = response.split('are ')[1]
                if 'a ' in response:
                    response = response.split('a ')[1]
                if 'an ' in response:
                    response = response.split('an ')[1]
                if 'the ' in response:
                    response = response.split('the ')[1]
                if ' of' in response:
                    response = response.split(' of')[0]
                response = response.strip()
                entry['answer'] = response
        print({'accuracy': evaluate_exact_match_accuracy(outputs)})