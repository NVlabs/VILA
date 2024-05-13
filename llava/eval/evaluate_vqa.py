import argparse
import json
import os
from typing import Optional
import math

from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


ds_collections = {
    'docvqa_test': {
        'test': './playground/data/eval/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'test': './playground/data/eval/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'test': './playground/data/eval/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_val': {
        'test': './playground/data/eval/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'test': './playground/data/eval/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'test': './playground/data/eval/ai2d/test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    }
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base,)

    questions = [json.loads(q) for q in open(os.path.expanduser(ds_collections[args.dataset]['test']), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    outputs = []
    for line in tqdm(questions):
        qs = line["question"]
        image_file = line['image']
        question_id = line['question_id']
        annotation = line['answer']
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        pred = model.generate(
            input_ids=input_ids.cuda(),
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=ds_collections[args.dataset]['max_new_tokens'],
            num_return_sequences=1,
            use_cache=True
        )

        answer = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
        answer = answer.strip()

        if args.dataset in ['ocrvqa_val', 'ocrvqa_test']:
            outputs.append({
                'questionId': question_id,
                'answer': answer,
                'annotation': annotation,
            })
        elif args.dataset in ['ai2diagram_test']:
            outputs.append({
                'image': question_id,
                'answer': answer,
                'annotation': annotation,
            })
        elif args.dataset in ['chartqa_test_human', 'chartqa_test_augmented']:
            outputs.append({
                'answer': answer,
                'annotation': annotation,
            })
        elif args.dataset in ['docvqa_test']:
            outputs.append({
                'questionId': question_id,
                'answer': answer,
            })
        else:
            raise NotImplementedError

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, 'w') as ans_file:
        for output in outputs:
            ans_file.write(json.dumps(output) + "\n")

