# Run VILA demo on x86_64 machine

## Build TensorRT-LLM
The first step to build TensorRT-LLM is to fetch the sources:
```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git 
cd TensorRT-LLM
git checkout 66ef1df492f7bc9c8eeb01d7e14db01838e3f0bd
git submodule update --init --recursive
git lfs pull
```
Create a TensorRT-LLM Docker image and approximate disk space required to build the image is 63 GB:
```bash
make -C docker release_build
```

After launching the docker image, please install the following dependency:
```bash
pip install git+https://github.com/bfshi/scaling_on_scales.git
pip install git+https://github.com/huggingface/transformers@v4.36.2
```
## Build TensorRT engine of VILA model

### For VILA 1.0:

Please refer to the [documentation from TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal#llava-and-vila) to deploy the model.

### For VILA 1.5:

1. Setup
```bash
# clone vila
git clone https://github.com/Efficient-Large-Model/VILA.git

# enter the demo folder
cd <VILA-repo>/demo_trt_llm

# apply patch to /usr/local/lib/python3.10/dist-packages/tensorrt_llm/models/llama/convert.py for vila1.5
sh apply_patch.sh

# download vila checkpoint
export MODEL_NAME="vila1.5-2.7b"
git clone https://huggingface.co/Efficient-Large-Model/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
```

2. TensorRT Engine building using `FP16` and inference

Build TensorRT engine for LLaMA part of VILA from HF checkpoint using `FP16`:
```bash
python convert_checkpoint.py \
    --model_dir tmp/hf_models/${MODEL_NAME} \
    --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
    --dtype float16

trtllm-build \
    --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
    --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
    --gemm_plugin float16 \
    --use_fused_mlp \
    --max_batch_size 1 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --max_multimodal_len 4096
```

3. Build TensorRT engines for visual components

```bash
python build_visual_engine.py --model_path tmp/hf_models/${MODEL_NAME} --model_type vila --vila_path ../
```

4. Run the example script
```bash
python run.py  \
    --max_new_tokens 100 \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir visual_engines/${MODEL_NAME} \
    --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
    --image_file=av.png,https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png \
    --input_text="<image>\n<image>\n Please elaborate what you see in the images?" \
    --run_profiling

# example output:
...
[Q] <image>\n<image>\n Please elaborate what you see in the images?
[04/30/2024-21:32:11] [TRT-LLM] [I] 
[A] ['The first image shows a busy street scene with a car driving through a crosswalk. There are several people walking on the sidewalk, and a cyclist is also visible. The second image captures a beautiful sunset with the iconic Merlion statue spouting water into the water body in the foreground. The Merlion statue is a famous landmark in Singapore, and the water spout is a popular feature of the statue.']
...
```

5. (Optional) One can also use VILA with other quantization options, like SmoothQuant and INT4 AWQ, that are supported by LLaMA. Instructions in LLaMA README to enable SmoothQuant and INT4 AWQ can be re-used to generate quantized TRT engines for LLM component of VILA.
```bash
python quantization/quantize.py \
     --model_dir tmp/hf_models/${MODEL_NAME} \
     --output_dir tmp/trt_models/${MODEL_NAME}/int4_awq/1-gpu \
     --dtype float16 \
     --qformat int4_awq \
     --calib_size 32

 trtllm-build \
     --checkpoint_dir tmp/trt_models/${MODEL_NAME}/int4_awq/1-gpu \
     --output_dir trt_engines/${MODEL_NAME}/int4_awq/1-gpu \
     --gemm_plugin float16 \
     --max_batch_size 1 \
     --max_input_len 2048 \
     --max_output_len 512 \
     --max_multimodal_len 4096

python run.py  \
    --max_new_tokens 100 \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir visual_engines/${MODEL_NAME} \
    --llm_engine_dir trt_engines/${MODEL_NAME}/int4_awq/1-gpu \
    --image_file=av.png,https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png \
    --input_text="<image>\n<image>\n Please elaborate what you see in the images?" \
    --run_profiling
```
