import os
import pickle
from tqdm import tqdm
import random
random.seed(1)

input_dirs = ["/dataset/llava-data/instruction-tuning/new-vflan"]
out_dir = "/dataset/llava-data/instruction-tuning/new-vflan-sharded"
os.makedirs(out_dir, exist_ok=True)

pkl_list = []

for inp_dir in input_dirs:
    for d in sorted(os.listdir(inp_dir)):
        if d.endswith(".pkl"):
            pkl_list.append(os.path.join(inp_dir, d))


all_samples = []
for pkl in tqdm(pkl_list):
    with open(pkl, "rb") as f:
        data_list = pickle.load(f)
    all_samples += data_list
random.shuffle(all_samples)
print("The vflan dataset has ", len(all_samples), "samples.")
per_shard_samples = len(all_samples) // 128

for idx, sample in enumerate(all_samples[:100]):
    print(idx, sample["question"])
counter = 0
# 11592 means that we will have exactly 128 shards
while len(all_samples) >= per_shard_samples:
    samples2write, all_samples = all_samples[:per_shard_samples], all_samples[per_shard_samples:]
    with open(os.path.join(out_dir, f"part-{counter:05d}.pkl"), "wb") as f:
        pickle.dump(samples2write, f)
    
    with open(os.path.join(out_dir, f"part-{counter:05d}.count"), "w") as f:
        f.write(str(len(samples2write)))    
    print(f"Finished writing part-{counter:05d}.pkl!")
    
    counter += 1
