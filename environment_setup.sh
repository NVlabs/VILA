conda create -n vila python=3.10 -y
conda activate vila

pip install --upgrade pip  # enable PEP 660 support
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e .
pip install -e ".[train]"

pip install git+https://github.com/huggingface/transformers@v4.36.2
cp -r ./llava/train/transformers_replace/* ~/anaconda3/envs/vila/lib/python3.10/site-packages/transformers/models/

