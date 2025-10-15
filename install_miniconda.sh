#!/bin/bash
set -e  # Exit immediately on error

# -------------------------------
# Download & install Miniconda
# -------------------------------
echo ">>> Installing Miniconda..."
if [ ! -f "Miniconda3-latest-Linux-x86_64.sh" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi

bash Miniconda3-latest-Linux-x86_64.sh
exec bash
