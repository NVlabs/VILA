<p align="center">
  <img src="demo_images/vila-logo.jpg" width="20%"/>
</p>

# VILA: On Pre-training for Visual Language Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](CODE_LICENSE)
[![Model License](https://img.shields.io/badge/MODEL%20License-CC%20By%20NC%204.0-red.svg)](MODEL_LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

[VILA arxiv](https://arxiv.org/abs/2312.07533) / [VILA Demo](https://vila-demo.hanlab.ai/) / [VILA Huggingface](https://huggingface.co/collections/Efficient-Large-Model/vila-on-pre-training-for-visual-language-models-65d8022a3a52cd9bcd62698e)

## 💡 Introduction

VILA is a visual language model (VLM) pretrained with interleaved image-text data at scale, enabling **video understanding** and **multi-image understanding** capabilities. VILA is deployable on the edge by [AWQ](https://arxiv.org/pdf/2306.00978.pdf) 4bit quantization and [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat) framework. We find: (1) image-text pairs are not enough, interleaved image-text is essential; (2) unfreezing LLM during interleaved image-text pre-training enables in-context learning; (3)re-blending text-only instruction data is crucial to boost both VLM and text-only performance; (4) token compression extends #video frames. VILA unveils appealing capabilities, including: video reasoning, in-context learning, visual chain-of-thought, and better world knowledge.

## 💡 News
- [2024/10] We release [VILA-U](https://github.com/mit-han-lab/vila-u): a Unified foundation model that integrates Video, Image, Language understanding and generation.
- [2024/08] We release [LongVILA](./LongVILA.md) that supports long video understanding (Captioning, QA, Needle-in-a-Haystack) up to 1024 frames.
- [2024/07] VILA1.5 also ranks 1st place (OSS model) on [MLVU test leaderboard](https://github.com/JUNJIE99/MLVU).
- [2024/06] VILA1.5 is now the best open sourced VLM on [MMMU leaderboard](https://mmmu-benchmark.github.io/#leaderboard) and [Video-MME](https://video-mme.github.io/home_page.html#leaderboard) leaderboard!
- [2024/05] We release VILA-1.5, which offers **video understanding capability**. VILA-1.5 comes with four model sizes: 3B/8B/13B/40B.
- [2024/05] We release [AWQ](https://arxiv.org/pdf/2306.00978.pdf)-quantized 4bit VILA-1.5 models. VILA-1.5 is efficiently deployable on diverse NVIDIA GPUs (A100, 4090, 4070 Laptop, Orin, Orin Nano) by [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat) and [TensorRT-LLM](demo_trt_llm) backends.
- [2024/03] VILA has been accepted by CVPR 2024!
- [2024/02] We release [AWQ](https://arxiv.org/pdf/2306.00978.pdf)-quantized 4bit VILA models, deployable on Jetson Orin and laptops through [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat) and [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine).
- [2024/02] VILA is released. We propose interleaved image-text pretraining that enables **multi-image** VLM. VILA comes with impressive in-context learning capabilities. We open source everything: including training code, evaluation code, datasets, model ckpts.
- [2023/12] [Paper](https://arxiv.org/abs/2312.07533) is on Arxiv!

## Performance

### Image QA Benchmarks

| $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ | Prec. | VQAv2 | GQA  | VizWiz | SQA-I | VQA-T | POPE | MME     | MMB  | MMB-CN | SEED | SEED-I | MMMU (val) | MMMU (test) | llava-bench | MM-Vet | Average |
| -------------------------------- | ----- | ----- | ---- | ------ | ----- | ----- | ---- | ------- | ---- | ------ | ---- | ------ | ---------- | ----------- | ----------- | ------ | ------- |
| VILA1.5-3B                       | fp16  | 80.4  | 61.5 | 53.5   | 69.0  | 60.4  | 85.9 | 1442.44 | 63.4 | 52.7   | 60.9 | 67.9   | 33.3       | 30.8        | 75.9        | 35.4   | 60.2    |
| VILA1.5-3B-AWQ                   | int4  | 80.0  | 61.1 | 53.8   | 67.8  | 60.4  | 85.9 | 1437.34 | 63.3 | 51.4   | 59.8 | 66.6   | 32.7       | 31.1        | 75.0        | 37.3   | 59.9    |
| VILA1.5-3B-S2                    | fp16  | 79.8  | 61.4 | 61.3   | 69.6  | 63.4  | 85.3 | 1431.65 | 62.8 | 52.2   | 60.0 | 66.4   | 32.8       | 31.3        | 76.7        | 38.6   | 60.9    |
| VILA1.5-3B-S2-AWQ                | int4  | 79.4  | 61.3 | 62.3   | 69.2  | 63.0  | 85.8 | 1417.06 | 61.6 | 51.5   | 59.1 | 65.7   | 33.4       | 30.4        | 77.1        | 36.7   | 60.5    |
| Llama-3-VILA1.5-8B               | fp16  | 83.0  | 63.5 | 63.2   | 82.0  | 68.5  | 85.6 | 1634.91 | 75.3 | 69.9   | 66.4 | 73.8   | 38.6       | 32.7        | 71.9        | 43.2   | 66.6    |
| Llama-3-VILA1.5-8B-AWQ           | int4  | 80.3  | 61.7 | 59.3   | 79.0  | 65.4  | 82.9 | 1593.65 | 71.0 | 64.9   | 64.0 | 71.1   | 36.0       | 36.1        | 79.0        | 37.2   | 64.5    |
| VILA1.5-13B                      | fp16  | 82.8  | 64.3 | 62.6   | 80.1  | 65.0  | 86.3 | 1569.55 | 74.9 | 66.3   | 65.1 | 72.6   | 37.9       | 33.6        | 80.8        | 44.3   | 66.3    |
| VILA1.5-13B-AWQ                  | int4  | 82.7  | 64.5 | 63.3   | 79.7  | 64.7  | 86.7 | 1531.35 | 74.7 | 66.7   | 65.1 | 72.6   | 37.8       | 34.0        | 81.9        | 46.4   | 66.5    |
| VILA1.5-40B                      | fp16  | 84.3  | 64.6 | 62.2   | 87.2  | 73.6  | 87.3 | 1726.82 | 82.4 | 80.2   | 69.1 | 75.8   | 51.9       | 46.9        | 81.3        | 53.0   | 72.4    |
| VILA1.5-40B-AWQ                  | int4  | 84.1  | 64.4 | 61.3   | 86.7  | 73.2  | 88.2 | 1714.79 | 83.2 | 79.6   | 68.9 | 75.6   | 49.3       | 46.2        | 83.0        | 51.4   | 72.1    |

<sup>NOTE: VQAV2 and VizWiz are test-dev, the average accuracy is calculated over all datasets and MME numbers are divided by 20.</sup>

### Video QA Benchmarks

| $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ | Prec. | Perception Test | ActivityNet  | MSVD | MSRVTT | TGIF | EgoSchema (test) | CinePile
| -------------------------------- | ----- | ----- | ---- | ------ | ----- | ----- | ----- | ----- |
| VILA1.5-3B                     | fp16  | 47  | 50.2 | 76.6  | 57.5  | 51.7  | 42.6 | 37.9
| VILA1.5-3B-S2                  | fp16  | 49.7  | 50.7 | 76.9  | 57.6 | 51.7 |
| Llama-3-VILA1.5-8B               | fp16  | 54.1  | 54.3 | 78.3   | 60.1  | 54.1 | 50.4 | 48.7
| VILA1.5-13B                      | fp16  | 53.6  | 54.7 | 77.9   | 60.2  | 56  | 52.2 | 50.1
| VILA1.5-40B                      | fp16  | 54  | 58 | 80.1  | 63 | 58.2 | 58.7 | 51.3

### Inference speed ( Token/sec )

| $~~~~~~$               | Precision | A100  | 4090  | Orin |
| ---------------------- | --------- | ----- | ----- | ---- |
| VILA1.5-3B             | fp16      | 104.6 | 137.6 | 25.4 |
| VILA1.5-3B-AWQ         | int4      | 182.8 | 215.5 | 42.5 |
| VILA1.5-3B-S2          | fp16      | 104.3 | 137.2 | 24.6 |
| VILA1.5-3B-S2-AWQ      | int4      | 180.2 | 219.3 | 40.1 |
| Llama-3-VILA1.5-8B     | fp16      | 74.9  | 57.4  | 10.2 |
| Llama-3-VILA1.5-8B-AWQ | int4      | 168.9 | 150.2 | 28.7 |
| VILA1.5-13B            | fp16      | 50.9  | OOM   | 6.1  |
| VILA1.5-13B-AWQ        | int4      | 115.9 | 105.7 | 20.6 |
| VILA1.5-40B            | fp16      | OOM   | OOM   | --   |
| VILA1.5-40B-AWQ        | int4      | 57.0  | OOM   | --   |

<sup>NOTE: Measured using the [TinyChat](https://github.com/mit-han-lab/llm-awq/tinychat) backend at batch size = 1.</sup>

## VILA Examples

### Video captioning

https://github.com/Efficient-Large-Model/VILA/assets/156256291/c9520943-2478-4f97-bc95-121d625018a6

Prompt: Elaborate on the visual and narrative elements of the video in detail.

Caption: The video shows a person's hands working on a white surface. They are folding a piece of fabric with a checkered pattern in shades of blue and white. The fabric is being folded into a smaller, more compact shape. The person's fingernails are painted red, and they are wearing a black and red garment. There are also a ruler and a pencil on the surface, suggesting that measurements and precision are involved in the process.

### In context learning

<img src="demo_images/demo_img_1.png" height="239">
<img src="demo_images/demo_img_2.png" height="250">

### Multi-image reasoning

<img src="demo_images/demo_img_3.png" height="193">

### VILA on Jetson Orin

https://github.com/Efficient-Large-Model/VILA/assets/7783214/6079374c-0787-4bc4-b9c6-e1524b4c9dc4

### VILA on RTX 4090

https://github.com/Efficient-Large-Model/VILA/assets/7783214/80c47742-e873-4080-ad7d-d17c4700539f

</details>

## Installation

```bash
./environment_setup.sh vila
```

## Training

VILA training contains three steps, for specific hyperparameters, please check out the [scripts/v1_5](scripts/v1_5) folder:

### Step-1: Alignment

We utilize LLaVA-CC3M-Pretrain-595K dataset to align the textual and visual modalities.

The stage 1 script takes in two parameters and it can run on a single 8xA100 node. `BASE_MODEL_PATH` points to a online or local huggingface repository, such as `NousResearch/Llama-2-7b-hf`. `OUTPUT_NAME` points to a target directory under `checkpoints`, which will save the trained multimodal projector afterwards.

```bash
bash scripts/v1_5/paper/1_mm_align.sh [BASE_MODEL_PATH] [OUTPUT_NAME]
```

### Step-2: Pretraining

We use MMC4 and Coyo dataset to train VLM with interleaved image-text pairs.

```bash
bash scripts/v1_5/paper/2_pretrain_mmc4_coyo.sh [CODE_PATH] [BASE_MODEL_PATH] [STAGE1_PATH] [OUTPUT_NAME]
```

The stage 2 script takes in four arguments. `CODE_PATH` is the absolute path to our VILA codebase, `BASE_MODEL_PATH` has similar meaning to what is presented in the stage 1 script. `STAGE1_PATH` points to the `OUTPUT_NAME` of stage 1 (i.e. where the stage 1 checkpoint is stored). `OUTPUT_NAME` is the desired folder name under `checkpoints` that saves the pretraining checkpoint. The script we provided for this stage is executed on slurm, and we expect it to execute on 16 nodes (128 GPUs).

### Step-3: Supervised fine-tuning

This is the last stage of VILA training, in which we tune the model to follow multimodal instructions on a subset of M3IT, FLAN and ShareGPT4V. This stage runs on a 8xA100 node.

```bash
bash scripts/v1_5/paper/3_sft.sh [STAGE2_PATH] [OUTPUT_NAME]
```

The stage 3 script takes in two arguments. `STAGE2_PATH` points to the `OUTPUT_NAME` of the stage 2 script (i.e. where the stage 2 checkpoint is stored). `OUTPUT_NAME` is the desired folder name under `checkpoints` that stores the final checkpoint.

## Evaluations

### Image Benchmarks

You can follow [Llava1.5 eval](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) to download all datasets. After downloading all datasets, please put them under `playground/data/eval`.

Please make the following changes to the MME evaluation script. Please search for:

```python
data_path = "MME_Benchmark_release_version"
```

and replace it with:

```python
data_path = os.path.join(script_dir, "MME_Benchmark_release_version")
```

We provide a push-the-button script to perform evaluation on all 10 datasets that do not require GPT-assisted evaluation:

```bash
./scripts/v1_5/eval/eval_all.sh [CHECKPOINT_PATH] [MODEL_NAME] [CONV_MODE]
```

This script takes in two parameters, `CHECKPOINT_PATH` points to the stage 3 model checkpoint, and `MODEL_NAME` will be the name of evaluation results.

[VQAv2](https://eval.ai/web/challenges/challenge-page/830/my-submission) and [Vizwiz](https://eval.ai/web/challenges/challenge-page/2185/my-submission) evaluations are hosted on eval.ai. You need to register an account and create a team to be able to submit eval.

MMBench and MMBench_CN eval are hosted on another [evaluation server](https://opencompass.org.cn/leaderboard-multimodal). Make sure you change the name of the file before submitting, otherwise the server caches results and will always return wrong result to you.

We provide a quick script to automatically organize the prediction files that need to be submitted to servers:

```bash
python scripts/v1_5/eval/copy_predictions.py [MODEL_NAME]
```

You will be able to find the predictions under `playground/data/predictions_upload/[MODEL_NAME]` after executing this script.

### Video Benchmarks

Please follow the evaluation steps in [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md#data-for-validating) for dataset preparation.

```bash
./scripts/v1_5/eval/video_chatgpt/run_all.sh [CHECKPOINT_PATH] [MODEL_NAME] [CONV_MODE]
./scripts/v1_5/eval/video_chatgpt/eval_all.sh [MODEL_NAME]
```

## Inference

We provide snippets for quick inference with user prompts and images.

Llama-3-VILA1.5-8B inference:

```bash
python -W ignore llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix \
    --conv-mode llama_3 \
    --query "<image>\n Please describe the traffic condition." \
    --image-file "av.png"
```

VILA1.5-40B inference:

```bash
python -W ignore llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/VILA1.5-40b \
    --conv-mode hermes-2 \
    --query "<image>\n Please describe the traffic condition." \
    --image-file "av.png"
```

VILA1.5-3B video inference:

```bash
python -W ignore llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --conv-mode vicuna_v1 \
    --query "<video>\n Please describe this video." \
    --video-file "demo.mp4"
```

## Quantization and Deployment

Our VILA models are quantized by [AWQ](https://arxiv.org/abs/2306.00978) into 4 bits for efficient inference on the edge. We provide a push-the-button [script](https://github.com/mit-han-lab/llm-awq/blob/main/scripts/vila_example.sh) to quantize VILA with AWQ.

### Running VILA on desktop GPUs and edge GPUs

We support AWQ-quantized 4bit VILA on GPU platforms via [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat). We provide a [tutorial](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat#support-vlm-models-vila--llava) to run the model with TinyChat after quantization. We also provide an [instruction](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat/serve) to launch a Gradio server (powered by TinyChat and AWQ) to serve 4-bit quantized VILA models.

### Running VILA on laptops

We further support our AWQ-quantized 4bit VILA models on various CPU platforms with both x86 and ARM architectures with our [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine). We also provide a detailed [tutorial](https://github.com/mit-han-lab/TinyChatEngine/tree/main?tab=readme-ov-file#deploy-vision-language-model-vlm-chatbot-with-tinychatengine) to help the users deploy VILA on different CPUs.

### Running VILA API server

A simple API server has been provided to serve VILA models. The server is built on top of [FastAPI](https://fastapi.tiangolo.com/) and [Huggingface Transformers](https://huggingface.co/transformers/). The server can be run with the following command:

#### With CLI

```bash
python -W ignore server.py \
    --port 8000 \
    --model-path Efficient-Large-Model/VILA1.5-3B \
    --conv-mode vicuna_v1
```

#### With Docker

```bash
docker build -t vila-server:latest .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ./hub:/root/.cache/huggingface/hub \
    -it --rm -p 8000:8000 \
    -e VILA_MODEL_PATH=Efficient-Large-Model/VILA1.5-3B \
    -e VILA_CONV_MODE=vicuna_v1 \
    vila-server:latest
```

Then you can call the endpoint with the OpenAI SDK as follows:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000",
    api_key="fake-key",
)
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://blog.logomyway.com/wp-content/uploads/2022/01/NVIDIA-logo.jpg",
                        # Or you can pass in a base64 encoded image
                        # "url": "data:image/png;base64,<base64_encoded_image>",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
    model="VILA1.5-3B",
    # You can pass in extra parameters as follows
    extra_body={"num_beams": 1, "use_cache": False},
)
print(response.choices[0].message.content)
```

<sup>NOTE: This API server is intended for evaluation purposes only and has not been optimized for production use. It has only been tested on A100 and H100 GPUs.</sup>

## Checkpoints

We release [VILA1.5-3B](https://hf.co/Efficient-Large-Model/VILA1.5-3b), [VILA1.5-3B-S2](https://hf.co/Efficient-Large-Model/VILA1.5-3b-s2), [Llama-3-VILA1.5-8B](https://hf.co/Efficient-Large-Model/Llama-3-VILA1.5-8B-Fix), [VILA1.5-13B](https://hf.co/Efficient-Large-Model/VILA1.5-13b), [VILA1.5-40B](https://hf.co/Efficient-Large-Model/VILA1.5-40b) and the 4-bit [AWQ](https://arxiv.org/abs/2306.00978)-quantized models [VILA1.5-3B-AWQ](https://hf.co/Efficient-Large-Model/VILA1.5-3b-AWQ), [VILA1.5-3B-S2-AWQ](https://hf.co/Efficient-Large-Model/VILA1.5-3b-s2-AWQ), [Llama-3-VILA1.5-8B-AWQ](https://hf.co/Efficient-Large-Model/Llama-3-VILA1.5-8B-Fix-AWQ), [VILA1.5-13B-AWQ](https://hf.co/Efficient-Large-Model/VILA1.5-13b-AWQ), [VILA1.5-40B-AWQ](https://hf.co/Efficient-Large-Model/VILA1.5-40b-AWQ).

## 🔒 License

- The code is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.
- The pretrained weights are released under the [CC-BY-NC-SA-4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).
- The service is a research preview intended for non-commercial use only, and is subject to the following licenses and terms:
  - [Model License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA. For LLAMA3-VILA checkpoints terms of use, please refer to the [LLAMA3 License](https://llama.meta.com/llama3/license/) for additional details.
  - [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI
  - [Dataset Licenses](./data_prepare/LICENSE) for each one used during training.

## Team

| | | |
| --- | --- | ---|
[\*Yao Lu](https://scholar.google.com/citations?user=OI7zFmwAAAAJ&hl=en): Nvidia|  [\*Hongxu Yin](https://hongxu-yin.github.io/): Nvidia |  [\*Ji Lin](https://www.linji.me/): OpenAI (work done at Nvidia and MIT)
[Wei Ping](https://scholar.google.com/citations?user=6gKEYRgAAAAJ&hl=en): Nvidia |   [Pavlo Molchanov](https://www.pmolchanov.com/): Nvidia |  [Andrew Tao](https://scholar.google.com/citations?user=Wel9l1wAAAAJ&hl=en): Nvidia |
[Haotian Tang](http://kentang.net/): MIT |  [Shang Yang](https://ys-2020.github.io/): MIT |  [Ligeng Zhu](https://lzhu.me/): Nvidia, MIT |
[Wei-Chen Wang](https://weichenwang.me/): MIT |  [Fuzhao Xue](https://xuefuzhao.github.io/): Nvidia, NUS |  [Yunhao Fang](https://seerkfang.github.io/): Nvidia, UCSD |
[Yukang Chen](https://yukangchen.com/): Nvidia | [Zhuoyang Zhang](https://openreview.net/profile?id=~Zhuoyang_Zhang1): Nvidia | [Yue Shen](https://www.linkedin.com/in/yue-james-shen/): Nvidia |
[Wei-Ming Chen](https://scholar.google.com/citations?user=6xFvyJwAAAAJ&hl=en): Nvidia |  [Huizi Mao](https://scholar.google.com/citations?user=r5WezOYAAAAJ&hl=zh-CN): Nvidia | [Baifeng Shi](https://bfshi.github.io/): Nvidia, UC Berkeley |
[Jan Kautz](https://jankautz.com/): Nvidia | [Mohammad Shoeybi](https://scholar.google.com/citations?user=62ElavIAAAAJ&hl=en): Nvidia | [Song Han](http://songhan.mit.edu/): Nvidia, MIT

## Citations

```
@misc{lin2023vila,
      title={VILA: On Pre-training for Visual Language Models},
      author={Ji Lin and Hongxu Yin and Wei Ping and Yao Lu and Pavlo Molchanov and Andrew Tao and Huizi Mao and Jan Kautz and Mohammad Shoeybi and Song Han},
      year={2023},
      eprint={2312.07533},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work.
- [InternVL](https://github.com/OpenGVLab/InternVL): for open-sourcing InternViT (used in VILA1.5-40b) and the [InternVL-SFT](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#prepare-training-datasets) data blend (inspired by LLaVA-1.6) used in all VILA1.5 models.
- [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT): we borrowed video evaluation script from this repository.
- [MMC4](https://github.com/allenai/mmc4), [COYO-700M](https://github.com/kakaobrain/coyo-dataset), [M3IT](https://huggingface.co/datasets/MMInstruction/M3IT), [OpenORCA/FLAN](https://huggingface.co/datasets/Open-Orca/FLAN), [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V), [WIT](google-research-datasets/wit), [GSM8K-ScRel](https://github.com/OFA-Sys/gsm8k-ScRel/blob/main/data/train_use.jsonl), [VisualGenome](https://visualgenome.org/api/v0/api_home.html), [VCR](https://visualcommonsense.com/download/), [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [Shot2Story](https://github.com/bytedance/Shot2Story/blob/master/DATA.md), [Youcook2](http://youcook2.eecs.umich.edu/), [Vatex](https://eric-xw.github.io/vatex-website/download.html), [ShareGPT-Video](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction) for providing datasets used in this research.
