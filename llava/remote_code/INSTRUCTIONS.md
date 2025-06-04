---
license: cc-by-nc-4.0
language:
- en
tags:
- vila
- nvila
- conversational
- multimodal
---

Dependency setups:

```bash
# other transformers version may also work, but we have not tested
pip install transformers==4.46 accelerate opencv-python torchvision einops pillow
pip install git+https://github.com/bfshi/scaling_on_scales.git
```

## Usage

```python
from transformers import AutoConfig, AutoModel
from termcolor import colored

model_path = "Efficient-Large-Model/NVILA-Lite-2B-hf-preview"

# you can use config
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_config(config, trust_remote_code=True)
# or directly from_pretrained
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

# examples generate with raw text
res = model.generate_content([
    "how are you today?"
])
print(colored(res, "cyan", attrs=["bold"]))

print("---" * 40)

# examples generate with text + image
import PIL.Image
response = model.generate_content([
    PIL.Image.open("inference_test/test_data/caption_meat.jpeg"),
    "describe the image?"
])
print(colored(response, "cyan", attrs=["bold"]))
```

## AutoProcessor

we also support `AutoProcessor` class to ease data preparation for training and finetuning.


### single call

```python
from transformers import AutoProcessor, AutoModel

model_path = "Efficient-Large-Model/NVILA-Lite-2B-hf-preview"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
# important: set model to eval mode, otherwise the model will be in training mode and will pad to right.
model.eval()

gpt_conv = [{
    "role": "user",
    "content": [
        {"type": "image", "path": "https://nvlabs.github.io/VILA/asset/example.jpg"},
        {"type": "text", "text": "Describe this image."}
    ]
}]
text = processor.apply_chat_template(gpt_conv, tokenize=False, add_generation_prompt=True)
inputs = processor([text])

output_ids = model.generate(
    input_ids=inputs.input_ids,
    media=inputs.media,
    media_config=inputs.media_config,
    generation_config=model.generation_config,
    max_new_tokens=256,
)
print(processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True))

##### the above code is equivalent to
# response = model.generate_content([
#     PIL.Image.open("demo_images/demo_img_1.png"),
#     "describe the image?"
# ])
# print(colored(response, "cyan", attrs=["bold"]))
```

### batch call

```python
from transformers import AutoProcessor, AutoModel

model_path = "Efficient-Large-Model/NVILA-Lite-2B-hf-preview"
model_path = "./NVILA-Lite-2B-hf-preview"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
# important: set model to eval mode, otherwise the model will be in training mode and will pad to right.
model.eval()

gpt_conv1 = [{
    "role": "user",
    "content": [
        {"type": "image", "path": "https://nvlabs.github.io/VILA/asset/example.jpg"},
        {"type": "text", "text": "Describe this image."}
    ]
}]
gpt_conv2 = [{
    "role": "user",
    "content": [
        {"type": "image", "path": "https://nvlabs.github.io/VILA/asset/example_vqa.jpg"},
        {"type": "text", "text": "Describe this image for me. Provide a detailed description of the image."}
    ]
}]

messages = [gpt_conv1, gpt_conv2]
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
inputs = processor(texts)

output_ids = model.generate(
    input_ids=inputs.input_ids,
    media=inputs.media,
    media_config=inputs.media_config,
    generation_config=model.generation_config,
    max_new_tokens=256,
)
output_texts = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(output_texts[0])
print("---" * 40)
print(output_texts[1])
```


## Model Convert

The follwing code converts a convetional NVILA model to a HF compatible model.

```python
import os, os.path as osp
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor

model_path = "Efficient-Large-Model/NVILA-Lite-2B"
output_dir = "NVILA-Lite-2B-hf-preview"

if osp.isdir(output_dir):
    shutil.rmtree(output_dir)
from llava.remote_code.modeling_vila import VILAForCasualLM
VILAForCasualLM.convert_vila_dev_ckpt_to_remote(model_path, output_dir, copy=False)
```
