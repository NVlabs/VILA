# This file is originated from the official MMMU codebase:
# https://github.com/MMMU-Benchmark/MMMU
from random import random
import torch
from llava.mm_utils import is_gemma_tokenizer, KeywordsStoppingCriteria

def call_llava_engine_df(args, sample, model, tokenizer=None, processor=None):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle

    def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

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
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image = sample['image']
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if args.conv_mode == "v0" or is_gemma_tokenizer(tokenizer) else None
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
            stopping_criteria=stopping_criteria,)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response


def llava_image_processor(raw_image, vis_processors=None):
    image_tensor = vis_processors.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
    return image_tensor