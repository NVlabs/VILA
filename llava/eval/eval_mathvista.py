import json
from argparse import ArgumentParser

from llava.eval.mathvista_utils.calculate_score import simple_calculate_score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--answer_file", type=str, default="", help="The path to model answer file.")
    args = parser.parse_args()

    answer_dict = json.load(open(args.answer_file))
    scores_file = args.answer_file.split(".json")[0] + "_scores.json"
    simple_calculate_score(answer_dict, scores_file)
