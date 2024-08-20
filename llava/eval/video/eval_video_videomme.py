import argparse
import json
import os
import re
from typing import Dict, List, Optional, Union

CATEGORIES = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual",
]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual",
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def eval_your_results(
    your_results_path: str,
    groundtruth_path: str,
    video_types: Optional[Union[List[str], str]] = None,
    skip_missing: Optional[bool] = False,
    return_categories_accuracy: Optional[bool] = True,
    return_sub_categories_accuracy: Optional[bool] = False,
    return_task_types_accuracy: Optional[bool] = False,
    gt_answer_key: Optional[str] = "answer",
    your_answer_key: Optional[str] = "response",
):
    """
    Evaluate your results against the ground truth

    Args:
    - your_results_path (str): Path to your results file
    - video_types (Optional[List[str], str]): List of video types to evaluate.
    - skip_missing (Optional[bool]): If True, missing files will be skipped. If False, an error will be raised if there are missing files.
    - return_categories_accuracy (Optional[bool]): If True, the accuracy for each video category will be returned.
    - return_sub_categories_accuracy (Optional[bool]): If True, the accuracy for each video sub category will be returned.
    - return_task_types_accuracy (Optional[bool]): If True, the accuracy for each task category will be returned.
    - gt_answer_key (Optional[str]): Key to access the ground truth answer in the results file.
    - your_answer_key (Optional[str]): Key to access your answer in the results file.
    """

    # Load your results
    with open(your_results_path) as f:
        your_results_dict = json.load(f)
    groundtruth_list = json.load(open(groundtruth_path))
    class_mapping = {}
    # for video_item in groundtruth_list:
    #     class_mapping[video_item["video_id"]] = {
    #         "video_category": video_item["video_category"],
    #         "video_subcategory": video_item["video_subcategory"],
    #         "duration_category": video_item["duration_category"],
    #         "questions": video_item["questions"]
    #     }

    # Add the video category, sub category and duration_category to your results
    your_results = []
    for item in groundtruth_list:
        for question in item["questions"]:
            q_id = question["question_id"]
            if q_id in your_results_dict:
                question["response"] = your_results_dict[q_id]["response"]
                item["missing"] = False
            else:
                question["response"] = "C"
                item["missing"] = True
                break
        your_results.append(item)

        # q_id = item["question_id"]
        # video_id = q_id.split("-")[0]
        # if video_id in class_mapping:
        #     item["video_category"] = class_mapping[video_id]["video_category"]
        #     item["video_subcategory"] = class_mapping[video_id]["video_subcategory"]
        #     item["duration_category"] = class_mapping[video_id]["duration_category"]
        #     item["questions"] = class_mapping[video_id]["questions"]
        #     item["missing"] = False
        # else:
        #     item["missing"] = True
        # your_results.append(item)

    if isinstance(video_types, str):
        video_types = video_types.split(",")

    q_type_dict = {}
    v_type_dict = {}
    v_sub_type_dict = {}
    # print("Keys:", your_results.keys())
    # print(asd)

    for video_type in video_types:

        # Filter your results based on video types
        your_results_video_type = [item for item in your_results if item["duration_category"] == video_type]

        # Task Categories
        q_type_dict[video_type] = {}
        for q_type in TASK_CATEGORIES:
            q_type_dict[video_type][q_type] = {"correct": 0, "answered": 0}

        # Video categories
        v_type_dict[video_type] = {}
        for v_type in CATEGORIES:
            v_type_dict[video_type][v_type] = {"correct": 0, "answered": 0}

        v_sub_type_dict[video_type] = {}
        for v_sub_type in SUB_CATEGORIES:
            v_sub_type_dict[video_type][v_sub_type] = {"correct": 0, "answered": 0}

        if not skip_missing:
            # Check if the number of files in your results and ground truth are the same
            assert (
                len(your_results_video_type) == 300
            ), f"Number of files in {video_type} is not 300. Check if there are missing files."

        for item in your_results_video_type:

            if skip_missing and item["missing"]:
                continue

            # Get the video category, sub category and question category
            video_category = item["video_category"]
            video_sub_category = item["video_subcategory"]

            questions = item["questions"]

            for question in questions:
                q_type = question["task_type"]

                # Get the ground truth and your response
                gt_answer = question[gt_answer_key]
                response = question[your_answer_key]

                # Extract the answer from the response
                extration = extract_characters_regex(response)

                if extration != "":
                    q_type_dict[video_type][q_type]["answered"] += 1
                    q_type_dict[video_type][q_type]["correct"] += extration == gt_answer

                    v_type_dict[video_type][video_category]["answered"] += 1
                    v_type_dict[video_type][video_category]["correct"] += extration == gt_answer

                    v_sub_type_dict[video_type][video_sub_category]["answered"] += 1
                    v_sub_type_dict[video_type][video_sub_category]["correct"] += extration == gt_answer

    # Print the results for each video type
    for video_type in video_types:

        print("=====================================")
        print(f"Evaluation on video Type: {video_type}")
        print("=====================================")
        if return_categories_accuracy:
            print("-------------------------------------")
            print("Video Categories")
            print("-------------------------------------")
            for v_type in v_type_dict[video_type]:
                print(
                    f"{v_type}: {100 * v_type_dict[video_type][v_type]['correct'] / v_type_dict[video_type][v_type]['answered'] if v_type_dict[video_type][v_type]['answered'] > 0 else 0 : .1f}%"
                )
        if return_sub_categories_accuracy:
            print("-------------------------------------")
            print("Video Sub Categories")
            print("-------------------------------------")
            for v_sub_type in v_sub_type_dict[video_type]:
                print(
                    f"{v_sub_type}: {100 * v_sub_type_dict[video_type][v_sub_type]['correct'] / v_sub_type_dict[video_type][v_sub_type]['answered'] if v_sub_type_dict[video_type][v_sub_type]['answered'] > 0 else 0 : .1f}%"
                )
        if return_task_types_accuracy:
            print("-------------------------------------")
            print("Task Categories")
            print("-------------------------------------")
            for q_type in q_type_dict[video_type]:
                print(
                    f"{q_type}: {100 * q_type_dict[video_type][q_type]['correct'] / q_type_dict[video_type][q_type]['answered'] if q_type_dict[video_type][q_type]['answered'] > 0 else 0 : .1f}%"
                )

        print("-------------------------------------")
        print("Overall Performance")
        print("-------------------------------------")
        total_correct = sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES])
        total_answered = sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES])
        print(f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

        print("\n")

    # Print the results for the entire dataset
    print("=====================================")
    print("Evaluation on the entire dataset")
    print("=====================================")

    if return_categories_accuracy:
        print("-------------------------------------")
        print("Video Categories")
        print("-------------------------------------")
        for v_type in CATEGORIES:
            total_correct = sum([v_type_dict[video_type][v_type]["correct"] for video_type in video_types])
            total_answered = sum([v_type_dict[video_type][v_type]["answered"] for video_type in video_types])
            print(f"{v_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    if return_sub_categories_accuracy:
        print("-------------------------------------")
        print("Video Sub Categories")
        print("-------------------------------------")

        for v_sub_type in SUB_CATEGORIES:
            total_correct = sum([v_sub_type_dict[video_type][v_sub_type]["correct"] for video_type in video_types])
            total_answered = sum([v_sub_type_dict[video_type][v_sub_type]["answered"] for video_type in video_types])
            print(f"{v_sub_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    if return_task_types_accuracy:
        print("-------------------------------------")
        print("Task Categories")
        print("-------------------------------------")
        for q_type in TASK_CATEGORIES:

            total_correct = sum([q_type_dict[video_type][q_type]["correct"] for video_type in video_types])
            total_answered = sum([q_type_dict[video_type][q_type]["answered"] for video_type in video_types])
            print(f"{q_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    print("-------------------------------------")
    print("Overall Performance")
    print("-------------------------------------")
    total_correct = sum(
        [sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES]) for video_type in video_types]
    )
    total_answered = sum(
        [sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES]) for video_type in video_types]
    )
    print(f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--groundtruth_file", type=str, required=True)
    parser.add_argument("--video_duration_type", type=str, required=True)
    parser.add_argument("--return_categories_accuracy", action="store_true")
    parser.add_argument("--return_sub_categories_accuracy", action="store_true")
    parser.add_argument("--return_task_types_accuracy", action="store_true")
    parser.add_argument("--skip_missing", action="store_true")

    args = parser.parse_args()

    eval_your_results(
        args.results_file,
        args.groundtruth_file,
        video_types=args.video_duration_type,
        return_categories_accuracy=args.return_categories_accuracy,
        return_sub_categories_accuracy=args.return_sub_categories_accuracy,
        return_task_types_accuracy=args.return_task_types_accuracy,
        skip_missing=args.skip_missing,
    )
