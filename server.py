import argparse
import base64
import copy
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from threading import Thread
from typing import List, Literal, Optional, Union, get_args

import requests
import torch
import uvicorn
import tempfile
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import BaseModel
from transformers.generation.streamers import TextIteratorStreamer
from llava.utils.logging import logger
from llava.media import Video

from llava import conversation
from llava.constants import MEDIA_TOKENS
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class MediaURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: MediaURL

class VideoContent(BaseModel):
    type: Literal["video_url"]
    video_url: MediaURL
    frames: Optional[int] = 8
    fps: Optional[int] = 2


IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")
VIDEO_CONTENT_BASE64_REGEX = re.compile(r"^data:video/(mp4);base64,(.*)$")


def load_video(video_url: str) -> str:
    # download or parse video from base64
    if video_url.startswith("http") or video_url.startswith("https"):
        response = requests.get(video_url)
        video = BytesIO(response.content)
    else:
        match_results = VIDEO_CONTENT_BASE64_REGEX.match(video_url)
        if match_results is None:
            raise ValueError(f"Invalid video url: {video_url[:64]}")
        image_base64 = match_results.groups()[1]
        video = BytesIO(base64.b64decode(image_base64))

    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)

    temp_fpath = os.path.join(temp_dir, f"{uuid.uuid5(uuid.NAMESPACE_DNS, video_url)}.mp4")
    with open(temp_fpath, "wb") as f:
        f.write(video.getbuffer())

    return temp_fpath

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent, VideoContent]]]


class ChatCompletionRequest(BaseModel):
    model: Literal[
        "NVILA-15B",
        "VILA1.5-3B",
        "VILA1.5-3B-AWQ",
        "VILA1.5-3B-S2",
        "VILA1.5-3B-S2-AWQ",
        "Llama-3-VILA1.5-8B",
        "Llama-3-VILA1.5-8B-AWQ",
        "VILA1.5-13B",
        "VILA1.5-13B-AWQ",
        "VILA1.5-40B",
        "VILA1.5-40B-AWQ",
    ]
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    use_cache: Optional[bool] = True
    num_beams: Optional[int] = 1


model = None
model_name = None
tokenizer = None
image_processor = None
context_len = None


def load_image(image_url: str) -> Image:
    if image_url.startswith("http") or image_url.startswith("https"):
        print(f"[Server] Loading image from URL: {image_url}")
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content)).convert("RGB")
        print("[Server] Image loaded from URL successfully.")
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url format: {image_url}")
        image_base64 = match_results.groups()[1]
        try:
            image = PILImage.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
            print("[Server] Base64 image loaded successfully.")
        except Exception as e:
            print(f"[Server] Failed to decode base64 image: {e}")
            raise e
    return image



def get_literal_values(cls, field_name: str):
    field_type = cls.__annotations__.get(field_name)
    if field_type is None:
        raise ValueError(f"{field_name} is not a valid field name")
    if hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
        return get_args(field_type)
    raise ValueError(f"{field_name} is not a Literal type")


VILA_MODELS = get_literal_values(ChatCompletionRequest, "model")


def normalize_image_tags(qs: str) -> str:
    if MEDIA_TOKENS["image"] not in qs:
        logger.warning("No image was found in input messages.")
    elif MEDIA_TOKENS["video"] not in qs:
        logger.warning("No video was found in input messages.")
    return qs


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_name, tokenizer, image_processor, context_len
    disable_torch_init()
    model_path = app.args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, None)
    print(f"Model {model_name} loaded successfully. Context length: {context_len}")
    yield


app = FastAPI(lifespan=lifespan)


# Load model upon startup
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        global model, tokenizer, image_processor, context_len

        if request.model != model_name:
            raise ValueError(
                f"The endpoint is configured to use the model {model_name}, "
                f"but the request model is {request.model}"
            )

        generation_config = copy.deepcopy(model.default_generation_config)

        generation_config.max_new_tokens = request.max_tokens
        generation_config.temperature = request.temperature
        generation_config.top_p = request.top_p
        generation_config.do_sample = request.temperature > 0
        generation_config.num_beams = request.num_beams
        generation_config.use_cache = request.use_cache

        messages = request.messages
        conv_mode = app.args.conv_mode

        conv = conv_templates[conv_mode].copy()
        user_role = conv.roles[0]
        assistant_role = conv.roles[1]
        image = None
        video = None
        for message in messages:
            prompt = ""

            if message.role == "user":
                if isinstance(message.content, str):
                    prompt+= message.content
                elif isinstance(message.content, list):
                    for content in message.content:
                        if content.type == "text":
                            prompt += content.text
                        elif content.type == "image_url":
                            image = load_image(content.image_url.url)
                            prompt += MEDIA_TOKENS["image"]
                        elif content.type == "video_url":
                            video = load_video(content.video_url.url)
                            logger.info(f"loading {content.frames} frames from {video}")
                            model.config.num_video_frames = content.frames
                            model.config.fps = content.fps
                            video = Video(video)
                            prompt += MEDIA_TOKENS["video"]
                        else:
                            raise NotImplementedError(f"Unsupported content type: {content.type}")

                normalized_prompt = normalize_image_tags(prompt)
                conv.append_message(user_role, normalized_prompt)
            if message.role == "assistant":
                prompt = message.content
                conv.append_message(assistant_role, prompt)

        # add a last "assistant" message to complete the prompt
        if conv.sep_style == SeparatorStyle.LLAMA_3:
            conv.append_message(assistant_role, "")

        prompt_text = conv.get_prompt()
        logger.info(f"Prompt input: {prompt_text}")


        input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt").unsqueeze(0).to(model.device)


        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        if image is not None:
            prompt = [image, normalized_prompt]
        elif video is not None:
            prompt = [video, normalized_prompt]
        else:
            prompt = normalized_prompt

        with torch.inference_mode():
            if request.stream:
                streamer = model.generate_content(prompt, stream=True, generation_config = generation_config)

                def chunk_generator():
                    prepend_space = False
                    should_stop = False
                    chunk_id = 0
                    for new_text in streamer:
                        if new_text == " ":
                            prepend_space = True
                            continue
                        if new_text.endswith(stop_str):
                            new_text = new_text[: -len(stop_str)].strip()
                            prepend_space = False
                            should_stop = True
                        elif prepend_space:
                            new_text = " " + new_text
                            prepend_space = False
                        if len(new_text):
                            chunk = {
                                "id": str(chunk_id),
                                "object": "chat.completion.chunk",
                                "created": time.time(),
                                "model": request.model,
                                "choices": [{"delta": {"content": new_text}}],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(chunk_generator())

            else:

                outputs = model.generate_content(prompt=prompt, generation_config=generation_config)
                # Check if the response is None
                if not outputs:
                    raise ValueError("The model response is empty or malformed.")

                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
                outputs = outputs.strip()
                print("\nAssistant: ", outputs)

                resp_content = [TextContent(type="text", text=outputs)]
                return {
                    "id": uuid.uuid4().hex,
                    "object": "chat.completion",
                    "created": time.time(),
                    "model": request.model,
                    "choices": [{"message": ChatMessage(role="assistant", content=resp_content)}],
                }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


if __name__ == "__main__":

    host = os.getenv("VILA_HOST", "0.0.0.0")
    port = os.getenv("VILA_PORT", 8000)
    model_path = os.getenv("VILA_MODEL_PATH", "Efficient-Large-Model/VILA1.5-3B")
    conv_mode = os.getenv("VILA_CONV_MODE", "vicuna_v1")
    workers = os.getenv("VILA_WORKERS", 1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--conv-mode", type=str, default=conv_mode)
    parser.add_argument("--workers", type=int, default=workers)
    app.args = parser.parse_args()

    uvicorn.run(app, host=app.args.host, port=app.args.port, workers=app.args.workers)


