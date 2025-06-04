import copy
import os
import os.path as osp
import warnings
from collections import defaultdict
from io import BytesIO
from typing import List, Optional, Union

import PIL.Image
import requests
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

from .constants import DEFAULT_IMAGE_TOKEN, MEDIA_TOKENS
from .media import Image, Video, extract_media
from .mm_utils import process_image, process_images
from .tokenizer_utils import tokenize_conversation


def to_rgb(pil_image: PIL.Image.Image) -> PIL.Image.Image:
    if pil_image.mode == "RGBA":
        white_background = PIL.Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | PIL.Image.Image], size_factor=None) -> PIL.Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, PIL.Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = PIL.Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = PIL.Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = PIL.Image.open(BytesIO(data))
    else:
        image_obj = PIL.Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)

    return image


def fetch_image_url_or_fpath(url_or_fpath):
    if url_or_fpath.startswith("http") or url_or_fpath.startswith("https"):
        import tempfile

        import requests

        # Download the image to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, os.path.basename(url_or_fpath))

        response = requests.get(url_or_fpath, stream=True)
        response.raise_for_status()

        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return temp_file
    elif url_or_fpath.startswith("file://"):
        fpath = url_or_fpath.replace("file://", "")
        assert osp.exists(fpath), f"File {fpath} does not exist"
        return fpath
    elif osp.exists(url_or_fpath):
        assert osp.isfile(url_or_fpath), f"File {url_or_fpath} does not exist"
        return url_or_fpath
    else:
        raise ValueError(f"Unsupported image path: {url_or_fpath}")


def pad_fn(input_ids_list: List[torch.Tensor], padding_value=0, target_len=None, padding_side="left") -> torch.Tensor:
    # tensor shape is (batch_size, seq_len)
    max_len = max([ids.shape[1] for ids in input_ids_list])
    if target_len is not None:
        assert target_len >= max_len, "target_len must be greater than or equal to max_len"
        max_len = target_len

    new_input_ids_list = []
    for i, input_ids in enumerate(input_ids_list):
        pad_tensor = torch.ones_like(input_ids) * padding_value
        curr_len = input_ids.shape[1]
        pad_tensor = pad_tensor[:, : max_len - curr_len]
        if padding_side == "right":
            input_ids = torch.cat((input_ids, pad_tensor), dim=1)
        else:
            input_ids = torch.cat((pad_tensor, input_ids), dim=1)
        new_input_ids_list.append(input_ids)
    return torch.cat(new_input_ids_list, dim=0)


def extract_value_from_conv(chat):
    value = []
    if isinstance(chat["content"], str):
        # vila_chat["value"].append(chat["content"])
        value.append(chat["content"])
        return value

    # otherwise, it's a list of content
    for content in chat["content"]:
        if content["type"] == "image":
            if "path" in content:
                # VILA style, can be either filepath or http url
                value.append(Image(fetch_image_url_or_fpath(content["path"])))
            elif "image" in content:
                # Qwen style
                value.append(Image(fetch_image_url_or_fpath(content["image"])))
            elif "image_pil" in content:
                # Qwen style
                assert isinstance(content["image_pil"], PIL.Image.Image), f"Type of {media_key} must be PIL.Image.Image"
                value.append(content["image_pil"])
            else:
                raise ValueError(f"Type = `image` , but no `path` or `image` in | {content=}, {conversation=}")
        elif content["type"] == "text":
            value.append(content["text"])
        # NOTE(ligeng): video supports are needed here
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")
    return value


class VILAProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class VILAProcessor(ProcessorMixin):
    # attributes = ["image_processor", "tokenizer"]
    attributes = []
    # valid_kwargs = ["chat_template"]
    valid_kwargs = []
    # image_processor_class = "VILAImageProcessor"
    # tokenizer_class = ("VILATokenizer", "VILATokenizerFast")

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, config=None, padding_side="left", **kwargs
    ):
        self.image_token = MEDIA_TOKENS["image"]
        self.video_token = MEDIA_TOKENS["video"]
        self.config = config
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.padding_side = padding_side

        # This is a special setting for Qwen.
        # self.pad_token_id = tokenizer.pad_token_id
        self.pad_token_id = self.tokenizer("<|endoftext|>").input_ids[0]  # 151643
        self.eos_token_id = self.tokenizer.eos_token_id
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @staticmethod
    def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
        """
        referernce from qwen_vl_utils
        """
        vision_infos = []
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)
        return vision_infos

    @staticmethod
    def process_vision_info(
        conversations: list[dict] | list[list[dict]],
        return_video_kwargs: bool = False,
    ) -> tuple[list[PIL.Image.Image] | None, list[torch.Tensor | list[PIL.Image.Image]] | None, Optional[dict]]:
        """
        referernce from qwen_vl_utils
        NVILA does not depend on the function, but the interface is the same.
        """
        vision_infos = extract_vision_info(conversations)
        ## Read images or videos
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            elif "video" in vision_info:
                video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
            else:
                raise ValueError("image, image_url or video should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None
        if return_video_kwargs:
            return image_inputs, video_inputs, {"fps": video_sample_fps_list}
        return image_inputs, video_inputs

    @staticmethod
    def move_data_to_device(cls, prompt_inputs):
        def _move_data_to_device(item):
            # wrap function grpo trainer _prepare_input
            kwargs = {"device": cls.args.device}
            if cls.is_deepspeed_enabled and (torch.is_floating_point(item) or torch.is_complex(item)):
                kwargs.update({"dtype": cls.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return item.to(**kwargs)

        prompt_inputs.input_ids = _move_data_to_device(prompt_inputs.input_ids)
        prompt_inputs.attention_mask = _move_data_to_device(prompt_inputs.attention_mask)
        if "image" in prompt_inputs.media:
            prompt_inputs.media["image"] = [_move_data_to_device(img) for img in prompt_inputs.media["image"]]
        return prompt_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        padding_side = kwargs.get("padding_side", "left")
        if os.path.isdir(pretrained_model_name_or_path):
            pretrained_model_name_or_path = pretrained_model_name_or_path
        else:
            print(f"pretrained_model_name_or_path {pretrained_model_name_or_path} is not a directory, downloading")
            from huggingface_hub import snapshot_download

            pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)

        image_processor = AutoImageProcessor.from_pretrained(
            osp.join(pretrained_model_name_or_path, "vision_tower"), trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            osp.join(pretrained_model_name_or_path, "llm"), trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        return cls(image_processor=image_processor, tokenizer=tokenizer, config=config, padding_side=padding_side)

    def __repr__(self):
        # NOTE(ligeng):  hard coded image_processor to avoid serialization error. Dirty fix
        return f"VILAProcessor(image_processor=SigLip, tokenizer={self.tokenizer}, config={self.config})"

    def __call__(
        self,
        conversation=None,
        **kwargs: Unpack[VILAProcessorKwargs],
    ) -> BatchFeature:
        """
        The `conv` will be look like
        [
            {
                'from': 'human',
                'value': [
                    <transformers_modules.NVILA-Lite-2B-hf-preview.media.Image object at 0x154e68e4c460>,
                    'What are the common elements in these pictures?'
                ]
            }
        ]
        and `conversation` will be a list of such `conv`s
        """
        if kwargs.get("text", None) is not None:
            conversation = kwargs.get("text")
        assert conversation is not None, "`conversation` or `text` is required"
        padding_side = kwargs.get("padding_side", self.padding_side)

        input_ids_list = []
        attention_mask = []
        media = defaultdict(list)
        media_config = defaultdict(dict)
        for conv in conversation:
            feat = self.__single_call__(conv, **kwargs)
            input_ids_list.append(feat.input_ids)
            attention_mask.append(feat.attention_mask)
            for name in feat.media:
                media[name] += feat.media[name]
            for name in feat.media_config:
                media_config[name].update(feat.media_config[name])

        # pad the input_ids to batchfy
        input_ids = pad_fn(
            input_ids_list,
            padding_value=self.pad_token_id,
            padding_side=padding_side,
        )
        # ignore the pad token in the attention mask
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attention_mask[input_ids == self.pad_token_id] = False
        # print("[DEBUGAAA]", self.pad_token_id, self.tokenizer.pad_token_id); exit(0)
        input_texts = self.tokenizer.batch_decode(input_ids)
        bdata = BatchFeature(
            data={
                # "input_texts": input_texts,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "media": media,
                "media_config": media_config,
            }
        )
        # NOTE: hard coded to cuda
        # bdata.input_ids = bdata.input_ids.cuda()
        # bdata.attention_mask = bdata.attention_mask.cuda()
        # bdata.media["image"] = [img.cuda() for img in bdata.media["image"]]
        return bdata

    def __single_call__(
        self,
        conversation,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[VILAProcessorKwargs],
    ) -> BatchFeature:
        # TODO: should be merged with llava_arch.py/generate_content()
        # TODO (extract and preprocess should be done together, as the preprocess of image and video can be different, i.e. when dynamic res is used)
        conversation = copy.deepcopy(conversation)
        media = extract_media(conversation, self.config)
        # Process media
        media_config = defaultdict(dict)
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and self.config.image_aspect_ratio in ["dynamic", "dynamic_s2"]:
                    self.config.image_processor = self.image_processor
                    if self.config.image_aspect_ratio == "dynamic":
                        images = process_image(media["image"][0], self.config, None, enable_dynamic_res=True).half()
                        # print("DEBUG", len(images)); input()
                        # NOTE: this only works for images appears at the first conversation
                        conversation[0]["value"] = conversation[0]["value"].replace(
                            DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                        )
                    else:
                        if type(self.config.s2_scales) is str:
                            self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                        images, block_sizes = process_image(
                            media["image"][0], self.config, None, enable_dynamic_s2=True
                        )
                        images = images.half()
                        media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = process_images(media["image"], self.image_processor, self.config).half()
                media[name] = [image for image in images]
            elif name == "video":
                media[name] = [
                    process_images(images, self.image_processor, self.config).half() for images in media[name]
                ]
            else:
                raise ValueError(f"Unsupported media type: {name}")

        inputs = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True, return_ids_only=False)
        input_ids = inputs.input_ids[0].unsqueeze(0).cuda()

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "media": media,
                "media_config": media_config,
            }
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def convert_gpt_conv_to_vila_conv(self, conversation):
        vila_conv = []
        for chat in conversation:
            vila_chat = {"from": "", "value": []}
            if chat["role"] in ("user", "system"):
                # user allows to input image and text
                vila_chat["from"] = "human" if chat["role"] == "user" else "system"
                vila_chat["value"] = extract_value_from_conv(chat)
            elif chat["role"] == "assistant":
                vila_chat["from"] = "gpt"
                vila_chat["value"] = extract_value_from_conv(chat)
            else:
                raise ValueError(f"Unsupported role: {chat['role']} in chat {chat}")
            vila_conv.append(vila_chat)

        return vila_conv

    def apply_chat_template(self, conversation, add_generation_prompt=True, **kwargs):
        return self.convert_gpt_conv_to_vila_conv(conversation)


if __name__ == "__main__":
    # gpt style: user, assistant
    # vila style: human, gpt
    gpt_conv = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": "demo_images/demo_img_1.png"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    llavaconv = [
        {
            "from": "human",
            "value": [
                PIL.Image.open("demo_images/demo_img_1.png"),
                "Describe this image.",
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(output_dir, trust_remote_code=True)
    inputs = processor.apply_chat_template(conversation=gpt_conv, padding=True, return_tensors="pt")
    # model = llava.load("Efficient-Large-Model/qwen25_2B_3x3-sft").cuda()
    # print(model)
    model_path = "NVILA-Lite-2B-hf-preview"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    # res = model.generate_content(["how are you today?"])
    # print(model.config)
    # print(model.tokenizer)
    # print(res)

    processor = VILAProcessor(
        config=model.config,
        image_processor=model.vision_tower.image_processor,
        tokenizer=model.tokenizer,
    )

    # TODO: add padding, return_tensors,
    inputs = processor(conversation=llavaconv, padding=True, return_tensors="pt")
    print(inputs.keys(), inputs.input_ids.shape, [_.shape for _ in inputs.image])
    print("vila conv pass")

    inputs = processor.apply_chat_template(conversation=gpt_conv, padding=True, return_tensors="pt")
    print(inputs.keys(), inputs.input_ids.shape, [_.shape for _ in inputs.image])
    print("gpt conv pass")

    output_ids = model.generate(
        input_ids=inputs.input_ids,
        media={
            "image": inputs.image,
        },
        media_config={"image": {}},
        generation_config=model.generation_config,
        max_new_tokens=100,
    )
    print(output_ids)
