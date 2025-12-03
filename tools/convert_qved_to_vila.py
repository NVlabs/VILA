# tools/convert_qved_to_vila.py
import os, json, argparse
from pathlib import Path
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--hf_name", required=True)  # e.g. EdgeVLM-Labs/QVED-Test-Dataset
parser.add_argument("--out_dir", required=True)
parser.add_argument("--media_key", default="image")  # or "video"
args = parser.parse_args()

ds = load_dataset(args.hf_name, split="train")
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
media_dir = out_dir / ("images" if args.media_key=="image" else "videos")
media_dir.mkdir(exist_ok=True)

vila_items = []
for i, ex in enumerate(ds):
    # adjust fields depending on the HF dataset schema
    media_url = ex.get(args.media_key) or ex.get("image") or ex.get("video")
    # naive download if media_url is a URL
    fname = media_url.split("/")[-1]
    local_path = os.path.join(media_dir, fname)
    if media_url.startswith("http"):
        import requests
        r = requests.get(media_url, stream=True)
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
    else:
        # assume it's a local path already present in repo clone; copy it
        from shutil import copyfile
        copyfile(media_url, local_path)
    conv = ex.get("conversations") or ex.get("dialog") or [{"from":"human","value":ex.get("question","")},{"from":"gpt","value":ex.get("answer","")}]
    vila_items.append({args.media_key: os.path.join(media_dir.name, fname), "conversations": conv})

with open(out_dir / "dataset.json", "w") as f:
    json.dump(vila_items, f, indent=2)
print("Wrote:", out_dir)