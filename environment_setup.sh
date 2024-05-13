CONDA_ENV=${1:-""}

if [ -n "$CONDA_ENV" ]; then
    conda create -n $CONDA_ENV python=3.10 -y
    conda activate $CONDA_ENV
    # this is optional if you prefer to system built-in nvcc.
    conda install -c nvidia cuda-toolkit -y
else
    echo "skip conda enviroments"
fi

pip install --upgrade pip  # enable PEP 660 support
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e .
pip install -e ".[train]"

pip install git+https://github.com/huggingface/transformers@v4.36.2
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
