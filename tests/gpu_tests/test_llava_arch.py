'''
A inference test on llava_arch.py
This test can be simply run by
python llava_arch_unit_test.py \
            --model_path path_to_model \
            --question_file path_to_question_file \
            --image_folder image_directory \
            --device "cuda:0"
'''

import os
import json
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, AutoTokenizer

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/industrial/models/llava-v1.5-7b")
    parser.add_argument("--question_file", type=str, default="../tests/sample_data/llava_arch_test.json")
    parser.add_argument("--image_folder", type=str, default="../tests/sample_data/llava_arch_test_images")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # model initialization
    device = args.device
    dtype = torch.float16
    model = LlavaLlamaForCausalLM.from_pretrained(args.model_path).to(device).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=dtype)
    image_processor = vision_tower.image_processor

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        print("Checking Question: %s"%qs.split("\n")[0])
        if 'image' in line:
            image_file = line["image"]
            print("Image file: %s"%image_file)
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()

            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None

        input_ids = tokenizer_image_token(cur_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.int64)
        position_ids = torch.arange(input_ids.shape[-1], device=device)

        (
            input_ids_after,
            position_ids_after,
            attention_mask_after,
            _, inputs_embeds, _
        ) = model.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, None, None, images)

        if images is None:
            assert (position_ids_after - position_ids).abs().sum() == 0, "positions_ids should not be changed, without images"
            assert (attention_mask_after - attention_mask).abs().sum() == 0, "attention_mask should not be changed, without images"
            assert (input_ids_after - input_ids).abs().sum()==0, "input_ids should not be changed without images"
            assert inputs_embeds is None, "inputs_embeds should be None without images"
        else:
            assert position_ids_after.shape == (input_ids.shape[0], input_ids.shape[1] + 255), "positions_ids should not be changed, without images"
            assert attention_mask_after.shape == (input_ids.shape[0], input_ids.shape[1] + 255), "attention_mask should not be changed, without images"
            assert input_ids_after is None, "input_ids should not be changed without images"
            assert inputs_embeds.shape == (input_ids.shape[0], input_ids.shape[1] + 255, 4096), "inputs_embeds should have shape (batch size, num_tokens, hidden_dim)"

        print("Checking passed.")
