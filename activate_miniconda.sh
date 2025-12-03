# -----------------------------
# Initialize Conda
# -----------------------------
echo ">>> Initializing Conda..."

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# -------------------------------
# Create and activate environment
# -------------------------------
echo ">>> Creating Conda environment 'vila'..."
conda create -n vila python=3.10 -y

# Then we have to install activate the conda environment