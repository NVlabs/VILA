#!/usr/bin/env bash

# This is required to activate conda environment
eval "$(conda shell.bash hook)"

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    conda create -n $CONDA_ENV python=3.10 -y
    conda activate $CONDA_ENV
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# This is required to enable PEP 660 support
pip install --upgrade pip

# This is optional if you prefer to use built-in nvcc
conda install -c nvidia cuda-toolkit -y

# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install VILA
pip install -e .
pip install -e ".[train]"
pip install -e ".[eval]"

# Install HF's Transformers
pip install git+https://github.com/huggingface/transformers@v4.37.2
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
