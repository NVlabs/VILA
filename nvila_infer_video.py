import os
import csv
import subprocess

# -------------------------------
# CONFIG
# -------------------------------
VIDEO_FOLDER = "inference_videos"          # folder with your videos
OUTPUT_CSV = "video_inference_output.csv"  # Excel-readable output
PROMPT = "Please watch this video and describe the main events, actions, and scene in detail."

MODELS = {
    "NVILA-8B-Video": "Efficient-Large-Model/NVILA-8B-Video",
    "NVILA-15B": "Efficient-Large-Model/NVILA-15B",
}
# -------------------------------


def run_vila_infer(model_name, model_path, video_path):
    """
    Call the CLI `vila-infer` and return its stdout as the model's answer.
    """
    cmd = [
        "vila-infer",
        "--model-path", model_path,
        "--conv-mode", "auto",
        "--text", PROMPT,
        "--media", video_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Full stdout (including any logging) – you can later post-process if needed
    return result.stdout.strip()


def main():
    # Collect videos
    videos = [
        f for f in os.listdir(VIDEO_FOLDER)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"))
    ]

    if not videos:
        print(f"No videos found in folder: {VIDEO_FOLDER}")
        return

    # Prepare CSV
    fieldnames = ["video_name", "model", "description"]
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Loop over models and videos
        for model_name, model_path in MODELS.items():
            print(f"\n=== Running model: {model_name} ===")
            for vid in videos:
                vid_path = os.path.join(VIDEO_FOLDER, vid)
                print(f" -> Video: {vid_path}")

                description = run_vila_infer(model_name, model_path, vid_path)

                writer.writerow({
                    "video_name": vid,
                    "model": model_name,
                    "description": description,
                })

    print(f"\n✅ Done! Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
