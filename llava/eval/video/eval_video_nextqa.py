import argparse
import json
import os

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download(
    "wordnet"
)  # if you see error complaining "Resource wordnet not found.", even if you have already downloaded it, try unzip it by yourself. ref: https://github.com/nltk/nltk/issues/3028
nltk.download("punkt")
import numpy as np
from pywsd.utils import lemmatize_sentence
from tqdm import tqdm


def load_stopwords_file(filename):
    with open(filename) as fp:
        cont = fp.readlines()
        cont = [c.rstrip("\n") for c in cont]
    return cont


def remove_stop(sentence, stopwords):

    words = lemmatize_sentence(sentence)
    words = [w for w in words if not w in stopwords]
    return " ".join(words)


def wup(word1, word2, alpha):
    """
    calculate the wup similarity
    :param word1:
    :param word2:
    :param alpha:
    :return:
    """
    # print(word1, word2)
    if word1 == word2:
        return 1.0

    w1 = wordnet.synsets(word1)
    w1_len = len(w1)
    if w1_len == 0:
        return 0.0
    w2 = wordnet.synsets(word2)
    w2_len = len(w2)
    if w2_len == 0:
        return 0.0

    # match the first
    word_sim = w1[0].wup_similarity(w2[0])
    if word_sim is None:
        word_sim = 0.0

    if word_sim < alpha:
        word_sim = 0.1 * word_sim
    return word_sim


def wups(words1, words2, alpha):
    """

    :param pred:
    :param truth:
    :param alpha:
    :return:
    """
    sim = 1.0
    flag = False
    for w1 in words1:
        max_sim = 0
        for w2 in words2:
            word_sim = wup(w1, w2, alpha)
            if word_sim > max_sim:
                max_sim = word_sim
        if max_sim == 0:
            continue
        sim *= max_sim
        flag = True
    if not flag:
        sim = 0.0
    return sim


def get_wups(pred, truth, alpha):
    """
    calculate the wups score
    :param pred:
    :param truth:
    :return:
    """
    pred = word_tokenize(pred)
    truth = word_tokenize(truth)
    item1 = wups(pred, truth, alpha)
    item2 = wups(truth, pred, alpha)
    value = min(item1, item2)
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", default=r"", help="The path to file containing prediction.")
    parser.add_argument("--output_json", default=r"", help="The path to save annotation final combined json file.")
    parser.add_argument("--gt_file", default=r"", help="The path to save the ground truth file.")
    parser.add_argument("--stopwords_file", default="stopwords.txt", help="The path to save the stop words file.")
    args = parser.parse_args()
    return args


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()
    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]
    file.close()
    stopwords = load_stopwords_file(args.stopwords_file)

    # Read the ground truth csv file
    gt_file = os.path.expanduser(args.gt_file)
    with open(gt_file) as f:
        gt_qa_data = f.readlines()
    qtype_dict = {}
    for line in gt_qa_data[1:]:
        line = line.strip()
        line = line.split(",")
        video_name = line[1]
        question_id = line[7]
        question_type = line[8]
        if video_name not in qtype_dict:
            qtype_dict[video_name] = {}
            qtype_dict[video_name][question_id] = question_type
        else:
            qtype_dict[video_name][question_id] = question_type

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    sum_score = 0.0
    for sample in new_pred_contents:
        video_name = sample["video_name"]
        id = sample["id"]
        question = sample["question"]
        answer = sample["answer"]
        pred = sample["pred"]
        answer_processed = remove_stop(answer, stopwords)
        pred_processed = remove_stop(pred, stopwords)
        # if id in qtype_dict[video_name]:
        qtype = qtype_dict[video_name][id]
        if qtype == "DC" or qtype == "DB":
            score = 1.0 if answer_processed == pred_processed else 0.0
        else:
            score = get_wups(pred_processed, answer_processed, 0.0)
        sum_score += score
        qa_set = {"q": question, "a": answer, "pred": pred, "score": score}
        prediction_set[id] = qa_set

    sum_score = sum_score / len(new_pred_contents)
    print(f"Average WUPS score: {sum_score}")

    with open(args.output_json, "w") as f:
        json.dump(prediction_set, f, indent=4)
    print(f"Saved the prediction file at {args.output_json}")


if __name__ == "__main__":
    main()
