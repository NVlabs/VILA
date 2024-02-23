
import io
import sys
import os
import pickle
import json


start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])
print(start_idx, end_idx)


pkl_path = "/dataset/mmc4-core/pkl"
jsonl_path = "/dataset/mmc4-core/jsonl"
output_path = "/dataset/mmc4-core/jsonl-filtered"


pkl_list = sorted(os.listdir(pkl_path))[start_idx: end_idx]

for pkl in pkl_list:
    pickle_path = os.path.join(pkl_path, pkl)
    with open(pickle_path, 'rb') as f:
        image_dict = pickle.load(f)
    with open(os.path.join(jsonl_path, pkl.replace(".pkl", ".jsonl")), 'r') as json_file:
        json_list = list(json_file)
    annotation = [json.loads(json_str) for json_str in json_list]
    
    print(len(annotation), len(image_dict))
    filtered_annotation = []
    for i, anno in enumerate(annotation):
        if i in image_dict:
            assert len(image_dict[i]) == len(anno["image_info"])
            anno["org_idx"] = i
            filtered_annotation.append(anno)
    assert len(filtered_annotation) == len(image_dict)
    
    with open(os.path.join(output_path, pkl.replace(".pkl", ".jsonl")), 'w') as outfile:
        for record in filtered_annotation:
            json.dump(record, outfile)
            outfile.write('\n')

    with open(os.path.join(output_path, pkl.replace(".pkl", ".count")), 'w') as f:
        f.write(str(len(filtered_annotation)))
