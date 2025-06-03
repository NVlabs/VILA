import argparse
import json
import os
import re

import imagesize
import Levenshtein
import numpy as np

base_dir = "/lustre/fs2/portfolios/nvr/users/shiyic/datasets/train/aitz/android_in_the_zoo/test"


def intersect_iou(can_bbox, ref_bboxes):
    inter_xmin = np.maximum(can_bbox[0], ref_bboxes[0])
    inter_ymin = np.maximum(can_bbox[1], ref_bboxes[1])
    inter_xmax = np.minimum(can_bbox[2], ref_bboxes[2])
    inter_ymax = np.minimum(can_bbox[3], ref_bboxes[3])

    inter_area = np.maximum(0, inter_xmax - inter_xmin) * np.maximum(0, inter_ymax - inter_ymin)

    can_bbox_area = np.maximum((can_bbox[2] - can_bbox[0]) * (can_bbox[3] - can_bbox[1]), 1)
    ref_bboxes_area = np.maximum((ref_bboxes[2] - ref_bboxes[0]) * (ref_bboxes[3] - ref_bboxes[1]), 1)

    union_area = can_bbox_area + ref_bboxes_area - inter_area

    ious = inter_area / union_area
    return ious


def load_predictions(predictions_file):
    predictions = []
    with open(predictions_file) as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def compute_episode_metrics(episode_results):
    success, progress = [], []
    for __, eplist in episode_results.items():
        ep_success, ep_progress = True, 0
        for ex in eplist:
            if ex["exact_match"] is True:
                ep_progress += 1
            else:
                ep_success = False
            if not ep_success:
                break
        success.append(ep_success)
        progress.append(ep_progress / len(eplist) * 1.0)

    return {
        "success_rate": round(sum(success) / len(success), 4),
        "goal_progress": round(sum(progress) / len(progress), 4),
    }


def compute_atomic_metrics(step_results):
    recorder = {
        "total": {"count": 0, "type_match": 0, "exact_match": 0, "hit": 0},
        # -------------------------------------------
        "CLICK_ELEMENT": {"count": 0, "type_match": 0, "exact_match": 0},
        "INPUT": {"count": 0, "type_match": 0, "exact_match": 0, "text_dist": []},
        "SCROLL": {"count": 0, "type_match": 0, "exact_match": 0},
        "PRESS_HOME": {"count": 0, "type_match": 0, "exact_match": 0},
        "PRESS_BACK": {"count": 0, "type_match": 0, "exact_match": 0},
        "PRESS_ENTER": {"count": 0, "type_match": 0, "exact_match": 0},
        "STOP": {"count": 0, "type_match": 0, "exact_match": 0},
    }

    for step in step_results:
        recorder["total"]["count"] += 1
        recorder["total"]["hit"] += step["format_hit"]

        action_type = step["answer"]["action_type"].upper()
        recorder[action_type]["count"] += 1
        recorder[action_type]["type_match"] += step["type_match"]
        recorder["total"]["type_match"] += step["type_match"]
        recorder[action_type]["exact_match"] += step["exact_match"]
        recorder["total"]["exact_match"] += step["exact_match"]
        if "text_dist" in recorder[action_type] and step["text_dist"] is not None:
            recorder[action_type]["text_dist"].append(step["text_dist"])

    scores = {
        metric_key: {}
        for metric_key in [
            "total",
            "CLICK_ELEMENT",
            "SCROLL",
            "PRESS_HOME",
            "PRESS_BACK",
            "PRESS_ENTER",
            "STOP",
            "INPUT",
        ]
    }
    scores["total"]["hit_rate"] = round(recorder["total"]["hit"] / recorder["total"]["count"], 4)
    for metric_key in ["total", "CLICK_ELEMENT", "SCROLL", "PRESS_HOME", "PRESS_BACK", "PRESS_ENTER", "STOP", "INPUT"]:
        if recorder[metric_key]["count"] == 0:
            continue
        scores[metric_key]["type_acc"] = round(recorder[metric_key]["type_match"] / recorder[metric_key]["count"], 4)
        scores[metric_key]["exact_acc"] = round(recorder[metric_key]["exact_match"] / recorder[metric_key]["count"], 4)
    if recorder["INPUT"]["text_dist"]:
        scores["INPUT"]["text_dist"] = round(
            sum(recorder["INPUT"]["text_dist"]) / len(recorder["INPUT"]["text_dist"]), 4
        )
    return scores


def _check_click_(pred_bbox, gt_bbox):
    ious = intersect_iou(pred_bbox, gt_bbox)
    if np.any(ious > 0.5):
        return True

    return False


def extract_all_bounding_boxes(text):
    pattern = r"\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]"
    matches = re.findall(pattern, text)
    bboxes = [[float(b) for b in box] for box in matches]
    return bboxes


def hard_parse(pred):
    # try to parse the json part first
    try:
        parse_pred = pred.split("{")
        if len(parse_pred) < 2:
            print("failed parse:", pred)
            raise json.JSONDecodeError("Invalid JSON format", pred, 0)
        pred = json.loads("{" + pred.split("{")[1])
        return pred
    except json.JSONDecodeError:
        pass

    _all_action_types_ = ["click_element", "scroll", "input", "press_home", "press_back", "press_enter", "stop"]
    action_type = None
    action_args = None
    for act_type in _all_action_types_:
        if act_type in pred or act_type.upper() in pred:
            action_type = act_type
            break
    if action_type is None:
        return {"ACTION": None, "ARGS": None}
    if action_type == "click_element":
        # parse the bbox
        bboxes = extract_all_bounding_boxes(pred)
        if len(bboxes) < 1:
            return {"ACTION": action_type, "ARGS": None}
        bbox = bboxes[0]
        return {"ACTION": action_type, "ARGS": {"bbox": bbox}}
    if action_type == "scroll":
        # parse the direction
        for direction in ["up", "down", "left", "right"]:
            parsed_direction = pred.split("scroll")
            if len(parsed_direction) > 1:
                if direction in pred.split("scroll")[1]:
                    return {"ACTION": action_type, "ARGS": {"direction": direction}}
        return {"ACTION": action_type, "ARGS": None}
    if action_type == "input":
        # parse the text
        parsed_text = pred.split("text")
        if len(parsed_text) > 1:
            text = pred.split("text")[1]
        else:
            return {"ACTION": action_type, "ARGS": None}
        return {"ACTION": action_type, "ARGS": {"text": text}}
    if action_type == "press_home" or action_type == "press_back" or action_type == "press_enter":
        return {"ACTION": action_type, "ARGS": {}}
    if action_type == "stop":
        return {"ACTION": action_type, "ARGS": {"status": "success"}}


def evaluate_single(gt, pred, episode_id, step_id):
    gt = json.loads(gt)
    # try json loads first
    try:
        pred = json.loads(pred)
    except json.JSONDecodeError:
        pred = hard_parse(pred)
    # print("pred", pred)

    # get ground truth information
    gt_action_type = gt["ACTION"]
    gt_action_args = gt["ARGS"]

    # get predict action information
    if pred is not None and isinstance(pred, dict) and ("ACTION" in pred or "action" in pred):
        pd_action_type = pred["ACTION"] if "ACTION" in pred else pred["action"]
        if "ARGS" not in pred or "args" not in pred:
            pd_action_args = None
        else:
            pd_action_args = pred["ARGS"] if "ARGS" in pred else pred["args"]
    else:
        pd_action_type = None
        pd_action_args = None

    # compute metrics
    hit_format = True if pd_action_type is not None else False
    type_match = pd_action_type is not None and gt_action_type == pd_action_type

    exact_match = False
    text_dist = None
    if type_match and pd_action_type == "click_element":
        if pd_action_args is None or "bbox" not in pd_action_args:
            exact_match = False
        else:
            exact_match = _check_click_(np.array(pd_action_args["bbox"]), np.array(gt_action_args["bbox"]))

    if type_match and pd_action_type == "scroll":
        if pd_action_args is None or "direction" not in pd_action_args:
            exact_match = False
        else:
            exact_match = pd_action_args["direction"] == gt_action_args["direction"]

    if type_match and pd_action_type == "input":
        if pd_action_args is None or "text" not in pd_action_args:
            exact_match = False
        else:
            pd_action_text = pd_action_args["text"]
            gt_action_text = gt_action_args["text"]
            text_dist = Levenshtein.ratio(pd_action_text, gt_action_text)
            exact_match = pd_action_text in gt_action_text or gt_action_text in pd_action_text or text_dist > 0.8

    if (
        type_match
        and pd_action_type == "press_home"
        or pd_action_type == "press_back"
        or pd_action_type == "press_enter"
    ):
        exact_match = type_match

    if type_match and pd_action_type == "stop":
        exact_match = True

    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "answer": {"action_type": gt_action_type, "action_detail": gt_action_args},
        "pred": {"action_type": pd_action_type, "action_detail": pd_action_args},
        "type_match": type_match,
        "exact_match": exact_match,
        "text_dist": text_dist,
        "format_hit": hit_format,
    }


def eval(predictions):
    step_results = []
    for pred in predictions:
        pd = pred["text"]
        gt = pred["gt_caption"]
        episode_id, step_id = pred["question_id"].split("_")
        results = evaluate_single(gt, pd, episode_id, step_id)
        step_results.append(results)
    return compute_atomic_metrics(step_results)


def main(predictions_file):
    predictions = load_predictions(predictions_file)
    scores = eval(predictions)
    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", help="prediction path")
    args = parser.parse_args()
    main(args.pred_path)
