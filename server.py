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
from PIL.Image import Image
from pydantic import BaseModel
from transformers.generation.streamers import TextIteratorStreamer

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatCompletionRequest(BaseModel):
    model: Literal[
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
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content)).convert("RGB")
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url: {image_url}")
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


VILA_MODELS = get_literal_values(ChatCompletionRequest, "model")


def normalize_image_tags(qs: str) -> str:
    if DEFAULT_IMAGE_TOKEN not in qs:
        print("No image was found in input messages. Continuing with text only prompt.")
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
        max_tokens = request.max_tokens
        temperature = request.temperature
        top_p = request.top_p
        use_cache = request.use_cache
        num_beams = request.num_beams

        messages = request.messages
        conv_mode = app.args.conv_mode

        images = []

        conv = conv_templates[conv_mode].copy()
        user_role = conv.roles[0]
        assistant_role = conv.roles[1]

        for message in messages:
            if message.role == "user":
                prompt = ""

                if isinstance(message.content, str):
                    prompt += message.content
                if isinstance(message.content, list):
                    for content in message.content:
                        if content.type == "text":
                            prompt += content.text
                        if content.type == "image_url":
                            image = load_image(content.image_url.url)
                            images.append(image)
                            prompt += DEFAULT_IMAGE_TOKEN
                normalized_prompt = normalize_image_tags(prompt)
                conv.append_message(user_role, normalized_prompt)
            if message.role == "assistant":
                prompt = message.content
                conv.append_message(assistant_role, prompt)

        # add a last "assistant" message to complete the prompt
        if conv.sep_style == SeparatorStyle.LLAMA_3:
            conv.append_message(assistant_role, "")

        prompt_text = conv.get_prompt()
        print("Prompt input: ", prompt_text)

        # support generation with text only inputs
        if len(images) == 0:
            images_input = None
        else:
            images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
            images_input = [images_tensor]

        input_ids = tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt").unsqueeze(0).to(model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            if request.stream:
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=20.0)
                thread = Thread(
                    target=model.generate,
                    kwargs=dict(
                        input_ids=input_ids,
                        images=images_input,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_tokens,
                        streamer=streamer,
                        use_cache=use_cache,
                        stopping_criteria=[stopping_criteria],
                    ),
                )
                thread.start()

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
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": time.time(),
                                "model": request.model,
                                "choices": [{"delta": {"content": new_text}}],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(chunk_generator())

            else:
                output_ids = model.generate(
                    input_ids,
                    images=images_input,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_tokens,
                    use_cache=use_cache,
                    stopping_criteria=[stopping_criteria],
                )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()
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


