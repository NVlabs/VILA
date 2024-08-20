import re
import sys

import pandas as pd

# !pip install python-Levenshtein
from Levenshtein import distance

sys.path.append("../")
from .utilities import *


def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]
    # return min(choices, key=lambda choice: distance(prediction, choice))


def normalize_extracted_answer(extraction, choices, question_type, answer_type, precision):
    """
    Normalize the extracted answer to match the answer type
    """
    if question_type == "multi_choice":
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except:
                extraction = ""

        # extract "A" from "(A) text"
        letter = re.findall(r"\(([a-zA-Z])\)", extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        options = [chr(ord("A") + i) for i in range(len(choices))]

        if extraction in options:
            # convert option letter to text, e.g. "A" -> "text"
            ind = options.index(extraction)
            extraction = choices[ind]
        else:
            # select the most similar option
            extraction = get_most_similar(extraction, choices)
        assert extraction in choices

    elif answer_type == "integer":
        try:
            extraction = str(int(float(extraction)))
        except:
            extraction = None

    elif answer_type == "float":
        try:
            extraction = str(round(float(extraction), precision))
        except:
            extraction = None

    elif answer_type == "list":
        try:
            extraction = str(extraction)
        except:
            extraction = None

    return extraction


def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


def get_acc_with_contion(res_pd, key, value):
    if key == "skills":
        # if value in res_pd[key]:
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
    else:
        total_pd = res_pd[res_pd[key] == value]

    correct_pd = total_pd[total_pd["true_false"] == True]
    acc = f"{len(correct_pd) / len(total_pd) * 100:.2f}"
    return len(correct_pd), len(total_pd), acc


def simple_calculate_score(results: dict, scores_file: str):
    full_pids = list(results.keys())
    print("Number of testing problems:", len(full_pids))

    ## [1] Evaluate if the prediction is true or false
    print("\nEvaluating the predictions...")
    for pid in full_pids:
        problem = results[pid]
        # print(problem)

        choices = problem["choices"]
        question_type = problem["question_type"]
        answer_type = problem["answer_type"]
        precision = problem["precision"]
        extraction = problem["extraction"]

        answer = problem["answer"]

        # normalize the extracted answer to match the answer type
        prediction = normalize_extracted_answer(extraction, choices, question_type, answer_type, precision)

        # verify the prediction is true or false
        true_false = safe_equal(prediction, answer)

        problem["prediction"] = prediction
        problem["true_false"] = true_false

    ## [2] Calculate the average accuracy
    total = len(full_pids)
    correct = 0
    for pid in full_pids:
        if results[pid]["true_false"]:
            correct += 1
    accuracy = str(round(correct / total * 100, 2))
    print(f"\nCorrect: {correct}, Total: {total}, Accuracy: {accuracy}%")

    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    ## [3] Calculate the fine-grained accuracy scores

    # merge the 'metadata' attribute into the data
    for pid in results:
        results[pid].update(results[pid].pop("metadata"))

    # convert the data to a pandas DataFrame
    df = pd.DataFrame(results).T

    print(len(df))
    print("Number of test problems:", len(df))
    # assert len(df) == 1000 # Important!!!

    # asign the target keys for evaluation
    target_keys = [
        "question_type",
        "answer_type",
        "language",
        "source",
        "category",
        "task",
        "context",
        "grade",
        "skills",
    ]

    for key in target_keys:
        print(f"\nType: [{key}]")
        # get the unique values of the key
        if key == "skills":
            # the value is a list
            values = []
            for i in range(len(df)):
                values += df[key][i]
            values = list(set(values))
        else:
            values = df[key].unique()
        # print(values)

        # calculate the accuracy for each value
        scores[key] = {}
        for value in values:
            correct, total, acc = get_acc_with_contion(df, key, value)
            if total > 0:
                print(f"[{value}]: {acc}% ({correct}/{total})")
                scores[key][value] = {
                    "accuracy": acc,
                    "correct": correct,
                    "total": total,
                }

        # sort the scores by accuracy
        scores[key] = dict(
            sorted(
                scores[key].items(),
                key=lambda item: float(item[1]["accuracy"]),
                reverse=True,
            )
        )

    # save the scores
    print(f"\nSaving {scores_file}...")
    save_json(scores, scores_file)
    print("\nDone!")
