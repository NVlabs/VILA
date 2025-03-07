from pprint import pprint
import argparse
import base64
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
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image as PILImage
# from PIL.Image import Image
from pydantic import BaseModel
from transformers.generation.streamers import TextIteratorStreamer

import tempfile
from fastapi import FastAPI, Request

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava import media
import llava
import asyncio 
import cv2
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter

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


def semaphore(value: int):
    """Decorator to limit the number of concurrent executions of an async function."""
    sem = asyncio.Semaphore(value)
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with sem:
                return await func(*args, **kwargs)
        return wrapper
    return decorator

IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")
VIDEO_CONTENT_BASE64_REGEX = re.compile(r"^data:video/(mp4);base64,(.*)$")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent, VideoContent]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    # these params are not actually used by NVILA
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    use_cache: Optional[bool] = True
    num_beams: Optional[int] = 1
    # fastapi 
    client: Optional[dict] = None


model = None
model_name = None
tokenizer = None
image_processor = None
context_len = None

def get_timestamp():
    return int(time.time())

def sample_frames_from_video(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(total_frames / num_frames * i) for i in range(num_frames)]
    sampled_frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
        else:
            print(f"Failed to read frame at index {idx}")
    cap.release()
    sampled_frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in sampled_frames]
    sampled_frames_pil = [PILImage.fromarray(frame) for frame in sampled_frames_rgb]
    return sampled_frames_pil


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
    temp_dir = ".serving"
    temp_fpath = os.path.join(temp_dir, f"{uuid.uuid5(uuid.NAMESPACE_DNS, video_url)}.mp4")
    
    with open(temp_fpath, "wb") as f:
        f.write(video.getbuffer())
        
    return temp_fpath


def load_image(image_url: str) -> PILImage:
    if image_url.startswith("http") or image_url.startswith("https"):
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content)).convert("RGB")
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url: {image_url[:64]}")
        image_base64 = match_results.groups()[1]
        image = PILImage.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    return image


def get_literal_values(cls, field_name: str):
    field_type = cls.__annotations__.get(field_name)
    if field_type is None:
        raise ValueError(f"{field_name} is not a valid field name")
    if hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
        return get_args(field_type)
    raise ValueError(f"{field_name} is not a Literal type")


# VILA_MODELS = get_literal_values(ChatCompletionRequest, "model")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_name, tokenizer, image_processor, context_len
    disable_torch_init()
    model_path = app.args.model_path
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, None)
    model = llava.load(model_path)
    # model = None
    print(f"{model_name=} {model_path=} loaded successfully. Context length: {context_len}")
    print("start & set capacity limiter to 1")
    RunVar("_default_thread_limiter").set(CapacityLimiter(1))
    global globallock
    globallock = asyncio.Lock()
    yield
    scheduler.shutdown()  # Ensure the scheduler stops when the app shuts down


app = FastAPI(lifespan=lifespan)

from starlette.types import Receive, Scope, Send

class MyStreamingResponse(StreamingResponse):
    async def listen_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                if globallock.locked():
                    print("DEBUG5: release lock for disconnected http client")
                    globallock.release()
                break

@app.get("/")
async def read_root():
    return {"message": "Welcome to the VILA API. This is for internal use only. Please use /chat/completions for chat completions."}

        
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # print("DEBUG0")
    current_time = time.strftime("%H:%M:%S-%s", time.localtime())
    current_time_hash = uuid.uuid5(uuid.NAMESPACE_DNS, current_time)
    obj_hash = uuid.uuid5(uuid.NAMESPACE_DNS, str(request.dict()))
    print("[Req recv]", current_time_hash, current_time, request.dict().keys())
    try:
        global model, tokenizer, image_processor, context_len

        if request.model != model_name:
            raise ValueError(
                f"The endpoint is configured to use the model {model_name}, "
                f"but the request model is {request.model}"
            )
        # use these configs to generate completions
        max_tokens = request.max_tokens
        temperature = request.temperature
        top_p = request.top_p
        use_cache = request.use_cache
        num_beams = request.num_beams

        messages = request.messages
        conv_mode = app.args.conv_mode
        conv = conv_templates[conv_mode].copy()
        
        ########################################################################### 
        prompt = []
        for message in messages:
            if isinstance(message.content, str):
                prompt.append(message.content)

            if isinstance(message.content, list):
                for content in message.content:
                    # print(content.type)
                    if content.type == "text":
                        prompt.append(content.text)
                    elif content.type == "image_url":
                        image = load_image(content.image_url.url)
                        prompt.append(image)
                    elif content.type == "video_url":
                        video = load_video(content.video_url.url)
                        print("saving video to file", video)
                        frames = sample_frames_from_video(video, content.frames)
                        print(f"loading {content.frames} frames", video)
                        prompt += frames
                    else:
                        raise NotImplementedError(f"Unsupported content type: {content.type}")
        
        with torch.inference_mode():
            if request.stream:
                await globallock.acquire()
                streamer = model.generate_content(prompt, stream=True)
                # streamer = "helloworld!" 
                def chunk_generator():
                    for chunk_id, new_text in enumerate(streamer):
                        if len(new_text):
                            chunk = {
                                "id": str(chunk_id),
                                "object": "chat.completion.chunk",
                                "created": get_timestamp(),
                                "model": request.model,
                                "choices": [{"delta": {"content": new_text}}],
                                "usage": {
                                    "completion_tokens": 38, 
                                    "prompt_tokens": 8, 
                                    "total_tokens": 46
                                }
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                
                async def chunk_generator_wrapper():
                    for chunk in chunk_generator():
                        yield chunk
                    if globallock.locked():
                        globallock.release()
                return MyStreamingResponse(chunk_generator_wrapper())
            else:
                await globallock.acquire()
                outputs = model.generate_content(prompt)
                # outputs = "helloworld!" 
                if globallock.locked():
                    globallock.release()
                print("\nAssistant: ", outputs)
                resp_content = outputs
                return {
                    "id": uuid.uuid4().hex,
                    "object": "chat.completion",
                    "created": get_timestamp(),
                    "model": request.model,
                    "index": 0,
                    "choices": [
                        {"message": ChatMessage(role="assistant", content=resp_content)}
                    ],
                }
    except Exception as e:
        if globallock.locked():
            globallock.release()
            
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        pass
    
if __name__ == "__main__":
    global host, port
    host = os.getenv("VILA_HOST", "0.0.0.0")
    port = os.getenv("VILA_PORT", 8000)
    model_path = os.getenv("VILA_MODEL_PATH", "Efficient-Large-Model/NVILA-8B")
    conv_mode = os.getenv("VILA_CONV_MODE", "auto")
    workers = os.getenv("VILA_WORKERS", 1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--conv-mode", type=str, default=conv_mode)
    # parser.add_argument("--workers", type=int, default=1)
    app.args = parser.parse_args()
    port = int(app.args.port)
    uvicorn.run(app, 
        host = app.args.host, 
        port = app.args.port, 
        workers = 1,
        timeout_keep_alive = 60,
    )
