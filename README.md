<p align="center">
  <img src="demo_images/nvila-logo.png" width="20%"/>
</p>

# NVILA: Efficient Frontier Visual Language Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](CODE_LICENSE)
[![Model License](https://img.shields.io/badge/MODEL%20License-CC%20By%20NC%204.0-red.svg)](MODEL_LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

[NVILA arXiv](https://arxiv.org/abs/2412.04468) / [NVILA Demo](https://vila.mit.edu/) / [NVILA Models (coming soon)](https://huggingface.co/collections/Efficient-Large-Model/nvila-674f8163543890b35a91b428) / [Subscribe](https://forms.gle/6nf1QdPYdvC2vgxM8)

## 💡 Introduction

NVILA is a family of open VLMs designed to optimize both **efficiency** and **accuracy** for efficient **video understanding** and **multi-image understanding** . Building on top of VILA, we improve its model architecture by first scaling up the spatial and temporal resolutions, and then compressing visual tokens. This "scale-then-compress" approach enables NVILA to efficiently process high-resolution images and long videos. We also conduct a systematic investigation to enhance the efficiency of NVILA throughout its entire lifecycle, from training and fine-tuning to deployment. NVILA matches or surpasses the accuracy of many leading open and proprietary VLMs across a wide range of image and video benchmarks. At the same time, it reduces training costs by 4.5×, fine-tuning memory usage by 3.4×, pre-filling latency by 1.6-2.2×, and decoding latency by 1.2-2.8×. We make our code and models available to facilitate reproducibility.

## 💡 News

- \[2024/12\] We release [NVILA](https://arxiv.org/abs/2412.04468) (a.k.a VILA2.0) that explores the full stack efficiency of multi-modal design, achieving cheaper training, faster deployment and better performance.
- \[2024/12\] We release [LongVILA](./longvila/README.md) that supports long video understanding, with long-context VLM with more than 1M context length and multi-modal sequence parallel system.
- \[2024/10\] VILA-M3, a SOTA medical VLM finetuned on VILA1.5 is released! VILA-M3 significantly outperforms Llava-Med and on par w/ Med-Gemini and is fully opensourced! [code](https://github.com/Project-MONAI/VLM#-news) [model](https://huggingface.co/MONAI)
- \[2024/10\] We release [VILA-U](https://github.com/mit-han-lab/vila-u): a Unified foundation model that integrates Video, Image, Language understanding and generation.
- \[2024/07\] VILA1.5 also ranks 1st place (OSS model) on [MLVU test leaderboard](https://github.com/JUNJIE99/MLVU).
- \[2024/06\] VILA1.5 is now the best open sourced VLM on [MMMU leaderboard](https://mmmu-benchmark.github.io/#leaderboard) and [Video-MME](https://video-mme.github.io/home_page.html#leaderboard) leaderboard!
- \[2024/05\] We release VILA-1.5, which offers **video understanding capability**. VILA-1.5 comes with four model sizes: 3B/8B/13B/40B.

<details>
<summary>Click to show more news</summary>

- \[2024/05\] We release [AWQ](https://arxiv.org/pdf/2306.00978.pdf)-quantized 4bit VILA-1.5 models. VILA-1.5 is efficiently deployable on diverse NVIDIA GPUs (A100, 4090, 4070 Laptop, Orin, Orin Nano) by [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat) and [TensorRT-LLM](demo_trt_llm) backends.
- \[2024/03\] VILA has been accepted by CVPR 2024!
- \[2024/02\] We release [AWQ](https://arxiv.org/pdf/2306.00978.pdf)-quantized 4bit VILA models, deployable on Jetson Orin and laptops through [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat) and [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine).
- \[2024/02\] VILA is released. We propose interleaved image-text pretraining that enables **multi-image** VLM. VILA comes with impressive in-context learning capabilities. We open source everything: including training code, evaluation code, datasets, model ckpts.
- \[2023/12\] [Paper](https://arxiv.org/abs/2312.07533) is on Arxiv!

</details>

## Performance

## Image Benchmarks

![](https://nvlabs.github.io/VILA/asset/image_results.png)

### Video  Benchmarks

![](https://nvlabs.github.io/VILA/asset/video_results.png)

### Efficient Deployments

![](https://nvlabs.github.io/VILA/asset/deployment_viz.png)

<sup>NOTE: Measured using the [TinyChat](https://github.com/mit-han-lab/llm-awq/tinychat) backend at batch size = 1.</sup>

## VILA Examples

### Image

<img src="demo_images/demo_img.png" height="750">

### Video

https://github.com/Efficient-Large-Model/VILA/assets/156256291/c9520943-2478-4f97-bc95-121d625018a6

Prompt: Elaborate on the visual and narrative elements of the video in detail.

Response: The video features a person with red nail polish meticulously working on a piece of fabric with a black and white checkered pattern. The individual is dressed in a black top with a red inner layer. The setting includes a white table where various tools such as a ruler, a pencil, and a pin are placed. The person is seen folding the fabric in a precise manner, creating a triangular shape. The background includes a greenish cabinet with drawers, adding a contrasting color to the scene. Throughout the video, the text 'Country Sampler' is visible in the bottom right corner, indicating the source or brand associated with the content. The video emphasizes the careful and detailed process of fabric folding, showcasing the person's skill and attention to detail.

### VILA on Jetson Orin

https://github.com/Efficient-Large-Model/VILA/assets/7783214/6079374c-0787-4bc4-b9c6-e1524b4c9dc4

### VILA on RTX 4090

https://github.com/Efficient-Large-Model/VILA/assets/7783214/80c47742-e873-4080-ad7d-d17c4700539f

## Installation

```bash
./environment_setup.sh vila
```

## Training

VILA training contains three steps, for specific hyperparameters, please check out the [scripts/v1_5](scripts/v1_5) folder:

### Step-1: Alignment

We utilize LLaVA-CC3M-Pretrain-595K dataset to align the textual and visual modalities.

The stage 1 script takes in two parameters and it can run on a single 8xA100 node.

```bash
bash scripts/NVILA-Lite/align.sh Efficient-Large-Model/Qwen2-VL-7B-Instruct <alias to data>
```

and the trained models will be saved to `runs/train/nvila-8b-align`.

### Step-1.5:

```bash
bash scripts/NVILA-Lite/stage15.sh runs/train/nvila-8b-align/model <alias to data>
```

and the trained models will be saved to `runs/train/nvila-8b-align-1.5`.

### Step-2: Pretraining

We use MMC4 and Coyo dataset to train VLM with interleaved image-text pairs.

```bash
bash scripts/NVILA-Lite/pretrain.sh runs/train/nvila-8b-align-1.5 <alias to data>
```

and the trained models will be saved to `runs/train/nvila-8b-pretraining`.

### Step-3: Supervised fine-tuning

This is the last stage of VILA training, in which we tune the model to follow multimodal instructions on a subset of M3IT, FLAN and ShareGPT4V. This stage runs on a 8xA100 node.

```bash
bash scripts/NVILA-Lite/sft.sh runs/train/nvila-8b-pretraining <alias to data>
```

and the trained models will be saved to `runs/train/nvila-8b-SFT`.

## Evaluations

We have introduce `vila-eval` command to simplify the evaluation. Once the data is prepared, the evaluation can be launched via

```bash
MODEL_NAME=NVILA-15B
MODEL_ID=Efficient-Large-Model/$MODEL_NAME
huggingface-cli download $MODEL_ID

vila-eval \
    --model-name $MODEL_NAME \
    --model-path $MODEL_ID \
    --conv-mode auto \
    --tags-include local
```

it will launch all evaluations and return a summarized result.

## Inference

We provide `vila-infer` for quick inference with user prompts and images.

```bash
# image description
vila-infer \
    --model-path Efficient-Large-Model/NVILA-15B \
    --conv-mode auto \
    --text "Please describe the image" \
    --media inference_test/test_data/caption_meat.jpeg

# video description
vila-infer \
    --model-path Efficient-Large-Model/NVILA-15B \
    --conv-mode auto \
    --text "Please describe the video" \
    --media https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4
```

<sup>NOTE: `vila-infer` is also compatible with VILA-1.5 models. You may find the usage example in [tests/bash/test_inference.sh](./tests/bash/test_inference.sh).</sup>

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
    --model-path Efficient-Large-Model/NVILA-15B \
    --conv-mode auto
```

#### With Docker

```bash
docker build -t vila-server:latest .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ./hub:/root/.cache/huggingface/hub \
    -it --rm -p 8000:8000 \
    -e VILA_MODEL_PATH=Efficient-Large-Model/NVILA-15B \
    -e VILA_CONV_MODE=auto \
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
    model="NVILA-15B",
)
print(response.choices[0].message.content)
```

<sup>NOTE: This API server is intended for evaluation purposes only and has not been optimized for production use. SGLang support is coming on the way.</sup>

## Checkpoints

We release the following models:

- NVILA-8B / NVILA-8B-Lite
- NVILA-15B / NVILA-15B-Lite

## 🔒 License

- The code is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.
- The pretrained weights are released under the [CC-BY-NC-SA-4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).
- The service is a research preview intended for non-commercial use only, and is subject to the following licenses and terms:
  - [Model License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA. For LLAMA3-VILA checkpoints terms of use, please refer to the [LLAMA3 License](https://llama.meta.com/llama3/license/) for additional details.
  - [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI
  - [Dataset Licenses](./data_prepare/LICENSE) for each one used during training.

## Team

NVILA Core contributors: [Zhijian Liu](https://zhijianliu.com), [Ligeng Zhu](https://lzhu.me/), [Baifeng Shi](https://bfshi.github.io/), [Zhuoyang Zhang](https://openreview.net/profile?id=~Zhuoyang_Zhang1), [Yuming Lou](<>), [Shang Yang](https://ys-2020.github.io/), [Haocheng Xi](<>), [Shiyi Cao](<>), [Yuxian Gu](<>), [Dacheng Li](<>), [Xiuyu Li](<>), [Yunhao Fang](https://seerkfang.github.io/), [Yukang Chen](https://yukangchen.com/), [Cheng-Yu Hsieh](<>), [De-An Huang](<>), [An-Chieh Cheng](<>), [Vishwesh Nath](<>), [Jinyi Hu](<>), [Sifei Liu](<>), [Ranjay Krishna](<>), [Daguang Xu](<>), [Xiaolong Wang](<>), [Pavlo Molchanov](https://www.pmolchanov.com/), [Jan Kautz](https://jankautz.com/), [Hongxu Yin](https://hongxu-yin.github.io/), [Song Han](http://songhan.mit.edu/), [Yao Lu](https://scholar.google.com/citations?user=OI7zFmwAAAAJ&hl=en)

LongVILA contributors: [Yukang Chen](https://yukangchen.com/), [Fuzhao Xue](https://xuefuzhao.github.io/), [Dacheng Li](<https://dachengli1.github.io>), [Qinghao Hu](<https://tonyhao.xyz>), [Ligeng Zhu](https://lzhu.me/), [Xiuyu Li](<https://xiuyuli.com>), [Yunhao Fang](https://seerkfang.github.io/), [Haotian Tang](http://kentang.net/), [Shang Yang](https://ys-2020.github.io/), [Zhijian Liu](https://zhijianliu.com), [Ethan He](<>), [Hongxu Yin](https://hongxu-yin.github.io/), [Pavlo Molchanov](https://www.pmolchanov.com/), [Jan Kautz](<https://jankautz.com>), [Linxi Fan](<https://jimfan.me>), [Yuke Zhu](<https://yukezhu.me>), [Yao Lu](https://scholar.google.com/citations?user=OI7zFmwAAAAJ&hl=en), [Song Han](http://songhan.mit.edu/)

<details>
<summary> VILA-1.5 contributors </summary>

[\*Yao Lu](https://scholar.google.com/citations?user=OI7zFmwAAAAJ&hl=en): Nvidia, [\*Hongxu Yin](https://hongxu-yin.github.io/): Nvidia, [\*Ji Lin](https://www.linji.me/): OpenAI (work done at Nvidia and MIT), [Wei Ping](https://scholar.google.com/citations?user=6gKEYRgAAAAJ&hl=en): Nvidia, [Pavlo Molchanov](https://www.pmolchanov.com/): Nvidia, [Andrew Tao](https://scholar.google.com/citations?user=Wel9l1wAAAAJ&hl=en): Nvidia, [Haotian Tang](http://kentang.net/): MIT, [Shang Yang](https://ys-2020.github.io/): MIT, [Ligeng Zhu](https://lzhu.me/): Nvidia, MIT, [Wei-Chen Wang](https://weichenwang.me/): MIT, [Fuzhao Xue](https://xuefuzhao.github.io/): Nvidia, NUS, [Yunhao Fang](https://seerkfang.github.io/): Nvidia, UCSD, [Yukang Chen](https://yukangchen.com/): Nvidia, [Zhuoyang Zhang](https://openreview.net/profile?id=~Zhuoyang_Zhang1): Nvidia, [Yue Shen](https://www.linkedin.com/in/yue-james-shen/): Nvidia, [Wei-Ming Chen](https://scholar.google.com/citations?user=6xFvyJwAAAAJ&hl=en): Nvidia, [Huizi Mao](https://scholar.google.com/citations?user=r5WezOYAAAAJ&hl=zh-CN): Nvidia, [Baifeng Shi](https://bfshi.github.io/): Nvidia, UC Berkeley, [Jan Kautz](https://jankautz.com/): Nvidia, [Mohammad Shoeybi](https://scholar.google.com/citations?user=62ElavIAAAAJ&hl=en): Nvidia, [Song Han](http://songhan.mit.edu/): Nvidia, MIT

</details>

## Citations

```
@misc{liu2024nvila,
      title={NVILA: Efficient Frontier Visual Language Models},
      author={Zhijian Liu and Ligeng Zhu and Baifeng Shi and Zhuoyang Zhang and Yuming Lou and Shang Yang and Haocheng Xi and Shiyi Cao and Yuxian Gu and Dacheng Li and Xiuyu Li and Yunhao Fang and Yukang Chen and Cheng-Yu Hsieh and De-An Huang and An-Chieh Cheng and Vishwesh Nath and Jinyi Hu and Sifei Liu and Ranjay Krishna and Daguang Xu and Xiaolong Wang and Pavlo Molchanov and Jan Kautz and Hongxu Yin and Song Han and Yao Lu},
      year={2024},
      eprint={2412.04468},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.04468},
}
```

```
@misc{chen2024longvila,
      title={LongVILA: Scaling Long-Context Visual Language Models for Long Videos},
      author={Yukang Chen and Fuzhao Xue and Dacheng Li and Qinghao Hu and Ligeng Zhu and Xiuyu Li and Yunhao Fang and Haotian Tang and Shang Yang and Zhijian Liu and Ethan He and Hongxu Yin and Pavlo Molchanov and Jan Kautz and Linxi Fan and Yuke Zhu and Yao Lu and Song Han},
      year={2024},
      eprint={2408.10188},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

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
