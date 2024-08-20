import argparse
import json
import os
from collections import defaultdict

eval_dict = {"refcoco": ["val", "testA", "testB"], "refcoco+": ["val", "testA", "testB"], "refcocog": ["val", "test"]}


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", help="prediction path")
    parser.add_argument("--data_path", help="data path")
    args = parser.parse_args()

    for dataset in ["refcoco", "refcoco+", "refcocog"]:
        for split in eval_dict[dataset]:
            try:
                prediction_file = os.path.join(args.pred_path, dataset, split, "merge.jsonl")
                pred_data = [json.loads(x) for x in open(prediction_file)]
                pred_dict = defaultdict(list)
                for pred in pred_data:
                    img_id = pred["img_id"][0]
                    pred_dict[img_id].append(pred)

                with open(os.path.join(args.data_path, f"{dataset}_{split}.json")) as f:
                    ann_data = json.load(f)

                count = 0
                total = len(ann_data)
                print(f"total ann: {total}\ntotal pred: {len(pred_data)}")

                refcoco_dict = defaultdict()
                for item in ann_data:
                    refcoco_dict[item["img_id"]] = item
                for img_id in refcoco_dict:
                    item = refcoco_dict[img_id]
                    bbox = item["bbox"]
                    outputs = pred_dict[img_id]
                    for output in outputs:
                        try:
                            pred_bbox = output["bbox"]

                            gt_bbox = [0, 0, 0, 0]
                            gt_bbox[0] = bbox[0]
                            gt_bbox[1] = bbox[1]
                            gt_bbox[2] = bbox[0] + bbox[2]
                            gt_bbox[3] = bbox[1] + bbox[3]
                            iou_score = computeIoU(pred_bbox, gt_bbox)
                            if iou_score >= 0.5:
                                count += 1
                        except Exception as e:
                            print(e, pred_bbox, gt_bbox, flush=True)
                            continue

                print(f"{dataset} {split}: {count / total * 100:.2f}\n", flush=True)
            except Exception as e:
                print(e, flush=True)
                continue
