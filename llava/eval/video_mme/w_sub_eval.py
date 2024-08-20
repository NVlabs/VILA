import os
import shutil

import cv2
import numpy as np
import pysubs2
from PIL import Image


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq


def create_frame_output_dir(output_dir):
    """
    Create the output directory for storing the extracted frames.

    Parameters:
    output_dir (str): Path to the output directory.

    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)


def slice_frames(video_path, srt_path, num_frames=8):
    """
    Extract frames from a video and save them to the output directory.

    Parameters:
    video_file_name (str): Path to the video file.
    num_frames (int): Number of frames to extract.
    res_path (str): Path to the output directory.
    subtitles_file_name (str): Path to the subtitles file.

    """
    # print(f"Extracting video: {video_path}")
    # create_frame_output_dir(os.path.join(output_path, "frames"))

    cv2_vr = cv2.VideoCapture(video_path)
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cv2_vr.get(cv2.CAP_PROP_FPS)
    selected_frame_ids = get_seq_frames(duration, num_frames)

    output_file_prefix = os.path.basename(video_path).replace(".", "_")

    count = 0

    pil_frames = []
    # Note(ligeng): frames are loaded using VILA's function, here we only load subtitles
    # while cv2_vr.isOpened():
    #     success, frame = cv2_vr.read()
    #     if not success:
    #         break
    #     if count in selected_frame_ids:
    #         min = int(count / fps) // 60
    #         sec = int(count / fps) % 60
    #         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         im_pil = Image.fromarray(img)
    #         pil_frames.append(im_pil)
    #         # time_string = f"{min:02d}:{sec:02d}"
    #         # image_name = f"{output_file_prefix}_frame_{time_string}.jpg"
    #         # cv2.imwrite(f"{output_path}/frames/{image_name}", frame)
    #     count += 1

    subtitles = ""
    if srt_path and os.path.exists(srt_path):
        subs = pysubs2.load(srt_path, encoding="utf-8")
        subtitles = []

        for seleced_frame_id in selected_frame_ids:
            sub_text = ""
            cur_time = pysubs2.make_time(fps=fps, frames=seleced_frame_id)
            for sub in subs:
                if sub.start < cur_time and sub.end > cur_time:
                    sub_text = sub.text.replace("\\N", " ")
                    break
            if sub_text.strip():
                subtitles.append(sub_text)
        subtitles = "\n".join(subtitles)

    return pil_frames, subtitles


if __name__ == "__main__":
    print(
        slice_frames(
            "/home/ligengz/nvr_elm_llm/dataset/Video-MME/videos/_8lBR0E_Tx8.mp4",
            "/home/ligengz/nvr_elm_llm/dataset/Video-MME/subtitle/_8lBR0E_Tx8.srt",
            num_frames=12,
        )
    )
