import os
import re
import sys
from typing import Any, Dict, List

import numpy as np
from lmms_eval.tasks.videomme.utils import base_cache_dir, cache_name, extract_subtitles

__all__ = ["videomme_doc_to_visual", "videomme_doc_to_text", "videomme_doc_to_text_subtitle"]


def videomme_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "data", doc["videoID"] + ".mp4")
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        raise FileNotFoundError(f"Video file not found: {video_path}")
    return [video_path]


def videomme_doc_to_text(doc: Dict[str, Any]) -> str:
    prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
    prompt += doc["question"] + "\n"
    prompt += "\n".join(doc["options"]) + "\n"
    prompt += "The best answer is:"
    return prompt


def videomme_doc_to_text_subtitle(doc: Dict[str, Any], num_frames: int) -> str:
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "data", doc["videoID"] + ".mp4")
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")

    if not os.path.exists(subtitle_path):
        return videomme_doc_to_text(doc)

    subtitles, frame_count = extract_subtitles(video_path, subtitle_path)

    indices = []
    for timestamp in np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int):
        for index, interval in enumerate(subtitles):
            if timestamp < interval[1] and timestamp >= interval[0]:
                indices.append(index)
    indices = list(set(indices))

    texts = []
    for index in indices:
        pattern = r'<font color="white" size=".72c">(.*?)</font>'
        try:
            texts.append(re.findall(pattern, subtitles[index][2])[0])
        except Exception:
            continue
    subtitle = "\n".join(texts)

    prompt = "This video's subtitles are listed below:\n"
    prompt += subtitle + "\n"
    prompt += "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n"
    prompt += doc["question"] + "\n"
    prompt += "\n".join(doc["options"]) + "\n"
    prompt += "The best answer is:"
    return prompt


