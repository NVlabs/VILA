# python llava/eval/mmmu_utils/merge_jsons.py --prediction-path ./playground/data/eval/MMMU/validation_results/$CKPT --num-chunks $CHUNKS

import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--num-chunks", type=int, default=1)
    args = parser.parse_args()
    prediction_paths = [f"{args.prediction_path}-{chunk}.json" for chunk in range(args.num_chunks)]
    combined_results = dict()
    for path in prediction_paths:
        with open(path) as f:
            partial_results = json.load(f)
        combined_results.update(**partial_results)
    print(len(combined_results))
    result_str = json.dumps(combined_results, indent=2)
    with open(f"{args.prediction_path}.json", "w") as f:
        f.write(result_str)
