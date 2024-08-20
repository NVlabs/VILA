import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", default=r"", help="The path to file containing prediction.")
    parser.add_argument("--output_json", default=r"", help="The path to save annotation final combined json file.")
    args = parser.parse_args()
    return args


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    # Read the prediction file
    with open(args.pred_path) as f:
        pred = f.readlines()
    # Convert each line of pred to a list of json objects
    new_pred_contents = [json.loads(i.strip()) for i in pred]
    total = 0
    correct = 0
    for pred in new_pred_contents:
        if pred["correct"]:
            correct += 1
        total += 1
    print("Total: ", total)
    print("Correct: ", correct)
    print("Accuracy: ", correct / total)


if __name__ == "__main__":
    main()
