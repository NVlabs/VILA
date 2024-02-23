import os
import pickle
from tqdm import tqdm

inp_dirs = ["/dataset/coyo/coyo-700m/pkl"]
out_dir = "/dataset/coyo/coyo-700m/pkl02-split"
os.makedirs(out_dir, exist_ok=True)

pkl_list = []

for inp_dir in inp_dirs:
    for d in sorted(os.listdir(inp_dir)):
        if d.endswith(".pkl"):
            pkl_list.append(os.path.join(inp_dir, d))

cur_samples = []

counter = 0
for pkl in tqdm(pkl_list):
    with open(pkl, "rb") as f:
        data_list = pickle.load(f)
    cur_samples += data_list
    while len(cur_samples) >= 12440:
        samples2write, cur_samples = cur_samples[:12440], cur_samples[12440:]
        with open(os.path.join(out_dir, f"part-{counter:05d}.pkl"), "wb") as f:
            pickle.dump(samples2write, f)
        
        with open(os.path.join(out_dir, f"part-{counter:05d}.count"), "w") as f:
            f.write(str(len(samples2write)))    
        
        counter += 1
