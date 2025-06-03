---
license: cc
language:
- en
base_model:
- Qwen/Qwen2.5-1.5B-Instruct
---

todos:
* check training stablitiy
    * zero3 compatibility
    * auto_processor (sync with qwenvl)
* currently resize_emd takes a long time cpu, can we switch to GPU?

already finished
* check numerical output same as original VILA impl
* save_pretrained()
* AutoModel.from_pretrained() / device_map auto to shard
* loading
* fix recursive imports
* text conv
* image + text conv:
    * .generate() / .generate_content()
    * llava/cli/infer.py
    * tests/bash/test_inference.sh

## NVILA HF Comptatible Mode
Remote model loading example

```python
from transformers import AutoConfig, AutoModel
from termcolor import colored

model_path = "Efficient-Large-Model/nvila_lite_2b_dev"
print("main_dev.py, loading from ", model_path)

# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModel.from_config(config, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
res = model.generate_content([
    "how are you today?"
])
print(colored(res, "cyan", attrs=["bold"]))

print("---" * 40)

import PIL.Image
response = model.generate_content([
    PIL.Image.open("inference_test/test_data/caption_meat.jpeg"),
    "describe the image?"
])
print(colored(response, "cyan", attrs=["bold"]))
```
