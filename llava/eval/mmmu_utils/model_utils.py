# This file is originated from the official MMMU codebase:
# https://github.com/MMMU-Benchmark/MMMU
import random

import torch

from llava.mm_utils import KeywordsStoppingCriteria, is_gemma_tokenizer


def call_llava_engine_df(args, sample, model, tokenizer=None, processor=None):
    from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates

    def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == "pt":
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f"Unsupported tensor type: {return_tensors}")
        return input_ids

    def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if DEFAULT_IMAGE_TOKEN not in qs:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        if mm_use_im_start_end:
            qs.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        return qs

    prompt = sample["final_input_prompt"]
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    image = sample["image"]

    if conv.sep_style == SeparatorStyle.LLAMA_3:
        keywords = [conv.sep, conv.sep2]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
    else:
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
            if args.conv_mode == "v0" or is_gemma_tokenizer(tokenizer)
            else None
        )

    if image is not None:
        output_ids = model.generate(
            input_ids,
            images=image.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=1,
            top_p=None,
            num_beams=5,
            max_new_tokens=128,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return outputs
    else:  # multiple images actually
        raise ValueError("INVALID GENERATION FOR MULTIPLE IMAGE INPUTS")
        # default behavior (random sample answer) from MMMU's offcials implementation
        if sample["question_type"] == "multiple-choice":
            all_choices = sample["all_choices"]
            outputs = random.choice(all_choices)
        else:
            outputs = "INVALID GENERATION FOR MULTIPLE IMAGE INPUTS"

    return outputs


def llava_image_processor(raw_image, vis_processors=None):
    image_tensor = vis_processors.preprocess(raw_image, return_tensors="pt")["pixel_values"][0]
    return image_tensor
