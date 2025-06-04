import os
from typing import List, Tuple

import accelerate
import requests
import torch
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.media import Video
from llava.utils import distributed as dist
from llava.utils import io


@register_model("vila_internal")
class VILA(lmms):
    def __init__(
        self,
        model_path: str,
        conv_mode: str,
        model_base: str = None,
        num_video_frames: int = 8,
        max_tiles: int = 12,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        assert batch_size == 1, "VILA only supports batch size of 1 at the moment."
        self._update_gpt_eval_model()

        devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
        torch.cuda.set_device(devices[0])

        self.model = llava.load(model_path, model_base=model_base, devices=devices)
        self.model.config.num_video_frames = num_video_frames
        context_length = num_video_frames * 512 * self.model.config.video_max_tiles

        self.model.config.min_tiles = 1
        self.model.config.max_tiles = max_tiles
        self.model.llm.config.min_tiles = 1
        self.model.llm.config.max_tiles = max_tiles

        # get PS3 configs from environment variables
        num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
        num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
        select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
        look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
        smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

        if num_look_close is not None:
            print("Num look close:", num_look_close)
            num_look_close = int(num_look_close)
            self.model.num_look_close = num_look_close
        if num_token_look_close is not None:
            print("Num token look close:", num_token_look_close)
            num_token_look_close = int(num_token_look_close)
            self.model.num_token_look_close = num_token_look_close
        if select_num_each_scale is not None:
            print("Select num each scale:", select_num_each_scale)
            select_num_each_scale = [int(x) for x in select_num_each_scale.split("+")]
            self.model.get_vision_tower().vision_tower.vision_model.max_select_num_each_scale = select_num_each_scale
        if look_close_mode is not None:
            print("Look close mode:", look_close_mode)
            self.model.look_close_mode = look_close_mode
        if smooth_selection_prob is not None:
            print("Smooth selection prob:", smooth_selection_prob)
            if smooth_selection_prob.lower() == "true":
                smooth_selection_prob = True
            elif smooth_selection_prob.lower() == "false":
                smooth_selection_prob = False
            else:
                raise ValueError(f"Invalid smooth selection prob: {smooth_selection_prob}")
            self.model.smooth_selection_prob = smooth_selection_prob

        # Adjust the max context length based on max_tiles and PS3 configs
        if max_tiles > 12:
            context_length = max(context_length, int(max_tiles / 12.0 * 4096))
        if num_look_close is not None:
            context_length = max(context_length, num_look_close * 2560 // 4 + 1024)
        if num_token_look_close is not None:
            context_length = max(context_length, num_token_look_close // 4 + 1024)
        context_length = max(getattr(self.model.tokenizer, "model_max_length", context_length), context_length)
        self.model.config.model_max_length = context_length
        self.model.config.tokenizer_model_max_length = context_length
        self.model.llm.config.model_max_length = context_length
        self.model.llm.config.tokenizer_model_max_length = context_length
        self.model.tokenizer.model_max_length = context_length

        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_mode].copy()

        self.accelerator = accelerate.Accelerator()
        self.device = torch.device(f"cuda:{devices[0]}")
        self._world_size = dist.size()
        self._rank = dist.rank()

    def _update_gpt_eval_model(self) -> None:
        _unpatched_post = requests.post

        def _patched_post(url, json, **kwargs):
            if json is not None and "model" in json:
                if json["model"] == "gpt-3.5-turbo-0613":
                    json["model"] = "gpt-4o-mini"
            return _unpatched_post(url, json=json, **kwargs)

        requests.post = _patched_post

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses = []
        for request in tqdm(requests, disable=not dist.is_main()):
            prompt, generation_kwargs, doc_to_visual, doc_id, task, split = self._patch(request.args)
            doc = self.task_dict[task][split][doc_id]

            # Generate multimodal prompt
            medias = []
            for media in doc_to_visual(doc):
                if isinstance(media, str):
                    if any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                        media = Video(media)
                    else:
                        raise NotImplementedError(f"Unsupported media type: {media}")
                medias.append(media)
            prompt = medias + [prompt]

            # Override generation config
            generation_config = self.model.default_generation_config
            generation_config.update(**generation_kwargs)

            # Generate and cache response
            cache_path = None
            if "CACHE_DIR" in os.environ:
                cache_path = os.path.join(os.environ["CACHE_DIR"], f"{task}_{split}_{doc_id}.txt")

            if cache_path is not None and os.path.exists(cache_path):
                response = io.load(cache_path)
            else:
                response = self.model.generate_content(prompt, generation_config=generation_config)
                if cache_path is not None:
                    io.save(cache_path, response)
            responses.append(response)

            print("Prompt:", prompt)
            print("Response:", response)
        return responses

    def _patch(self, args: Tuple) -> Tuple:
        prompt, generation_kwargs, doc_to_visual, doc_id, task, split = args
        doc = self.task_dict[task][split][doc_id]

        if task in ["videomme", "videomme_w_subtitle"]:
            from llava.eval.lmms.tasks.videomme import (
                videomme_doc_to_text,
                videomme_doc_to_text_subtitle,
                videomme_doc_to_visual,
            )

            # NOTE(zhijianl): This is a hack to make sure the video path is correct for Video-MME.
            doc_to_visual = videomme_doc_to_visual

            # Override the prompt for Video-MME, which can offer more than 1% improvement over the default prompt.
            if task == "videomme":
                prompt = videomme_doc_to_text(doc)
            if task == "videomme_w_subtitle":
                prompt = videomme_doc_to_text_subtitle(doc, num_frames=self.model.config.num_video_frames)

        return prompt, generation_kwargs, doc_to_visual, doc_id, task, split

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
