import argparse
import base64
import json
import time

import cv2
import openai
from openai import AzureOpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="evaluation using GPT-4o")
    parser.add_argument(
        "--pred_path",
        type=str,
        default="/lustre/fs2/portfolios/nvr/users/yukangc/VILA-Internal-main-3/runs/eval/Gemini/pred2.json",
        help="The path to file containing prediction.",
    )
    parser.add_argument("--save_dir", type=str, default="Gemini_golden", help="The path to save annotation json files.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/lustre/fs2/portfolios/nvr/users/yukangc/download_videos/short_videos_eval",
        help="The path to save annotation final combined json file.",
    )
    parser.add_argument("--eval_type", type=str, default="correctness", help="{correctness, detail, contextual}")
    args = parser.parse_args()
    return args


def process_video(video_path, seconds_per_frame=1, max_num_frames=20):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = max(int(fps * seconds_per_frame), total_frames // max_num_frames)
    curr_frame = 0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success or len(base64Frames) >= max_num_frames:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames


API_MAX_RETRY = 9999999999
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

import os

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

deployment_name = "gpt-4o"


def gpt_api_no_stream(question, pred, video_path, eval_type="correctness"):

    base64Frames = process_video(video_path, seconds_per_frame=1)

    output = API_ERROR_OUTPUT
    messages_correctness = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
            "Your task is to compare the predicted answer with the provided video and determine if they are factually consistent. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the factual consistency between the predicted answer and the provided video. The predicted answer should not contain any misinterpretations or misinformation.\n"
            "- The predicted answer must be factually accurate and align with the video content.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Evaluate the factual accuracy of the prediction compared to the answer.",
        },
        {
            "role": "system",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question: {question}\n"
            f"Predicted Answer: {pred}\n\n"
            "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {''score': 4.8}.",
        },
    ]

    messages_detail = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
            "Your task is to compare the predicted answer with the provided video and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
            "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.",
        },
        {
            "role": "system",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question: {question}\n"
            f"Predicted Answer: {pred}\n\n"
            "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {''score': 4.8}.",
        },
    ]

    messages_contextual = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
            "Your task is to compare the predicted answer with the provided video and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
            "- The predicted answer must capture the main themes and sentiments of the video.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Provide your evaluation of the contextual understanding of the prediction compared to the answer.",
        },
        {
            "role": "user",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question: {question}\n"
            f"Predicted Answer: {pred}\n\n"
            "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {''score': 4.8}.",
        },
    ]
    if eval_type == "correctness":
        messages = messages_correctness
    elif eval_type == "detail":
        messages = messages_detail
    elif eval_type == "contextual":
        messages = messages_contextual
    else:
        raise ValueError("Wrong eval_type %s." % eval_type)

    messages_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "These are the frames from the video."},
            *map(
                lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{x}", "detail": "low"}},
                base64Frames,
            ),
        ],
    }
    messages.append(messages_user)

    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=deployment_name, messages=messages, max_tokens=2000, temperature=0
            )
            output = response.choices[0].message.content
            print(output)
            break
        except openai.OpenAIError as e:
            print(type(e), e)
            if "The response was filtered" in e.message:
                break
            time.sleep(API_RETRY_SLEEP)
    return output


def main():
    args = parse_args()
    pred_path = args.pred_path
    save_dir = args.save_dir
    video_dir = args.video_dir
    eval_type = args.eval_type

    predictions = json.load(open(pred_path))
    if not os.path.exists(save_dir):
        os.system("mkdir %s" % save_dir)

    scores = []
    for pred in tqdm(predictions):
        key = pred["video_name"]
        print("Process %s" % key)
        save_path = os.path.join(save_dir, "%s.json" % key)
        if os.path.exists(save_path):
            print("Processed %s" % key)
            scores.append(json.load(open(save_path))[key])
            continue
        caption_content = pred["pred"]
        video_path = os.path.join(video_dir, "%s.mp4" % pred["video_name"])
        question = "Elaborate on the visual and narrative elements of the video in detail."
        output = gpt_api_no_stream(question, pred, video_path, eval_type)
        try:
            tag = "'score': "
            output = int(output[output.find(tag) + len(tag)])
        except:
            print("Skip", key, output)
            continue
        scores.append(output)
        print(output)
        json.dump({key: output}, open(save_path, "w"))

    print("Average score for %s is" % pred_path, sum(scores) / len(scores))


if __name__ == "__main__":
    main()
