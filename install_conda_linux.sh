#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

FFS_CONDA_ENV="${FFS_CONDA_ENV:-PyEditorFromFFS}"
CONDA_EXE=""

if command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  CONDA_EXE="$HOME/anaconda3/bin/conda"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  CONDA_EXE="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/mambaforge/bin/conda" ]]; then
  CONDA_EXE="$HOME/mambaforge/bin/conda"
elif [[ -x "/opt/conda/bin/conda" ]]; then
  CONDA_EXE="/opt/conda/bin/conda"
fi

if [[ -z "$CONDA_EXE" ]]; then
  echo "Could not find conda. Install Miniconda first:" >&2
  echo "  https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

echo "Using conda: $CONDA_EXE"

if "$CONDA_EXE" env list | awk '{print $1}' | grep -qx "$FFS_CONDA_ENV"; then
  echo "Conda env '$FFS_CONDA_ENV' already exists."
else
  echo "Creating conda env '$FFS_CONDA_ENV' ..."
  if ! "$CONDA_EXE" env create -f environment.yml; then
    "$CONDA_EXE" create -n "$FFS_CONDA_ENV" python=3.9 -y
  fi
fi

# shellcheck source=/dev/null
source "$("$CONDA_EXE" info --base)/etc/profile.d/conda.sh"
conda activate "$FFS_CONDA_ENV"

python -m pip install --upgrade pip wheel

echo "Installing PyTorch (CUDA 11.8 wheels)..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

echo "Installing ONNX Runtime GPU + TensorFlow..."
pip install onnxruntime-gpu==1.17.0 || pip install onnxruntime==1.17.0
pip install "tensorflow==2.10.1" || pip install "tensorflow==2.10.0"
pip install protobuf==3.20.2

echo "Installing application dependencies..."
pip install -r requirements-pip.txt
pip install python-magic

pip uninstall -y opencv-python opencv-headless-python opencv-contrib-python 2>/dev/null || true
pip install opencv-python==4.9.0.80

./download_models.sh

echo ""
echo "Done. Conda env: $FFS_CONDA_ENV"
echo "Run: ./fastfaceswap.sh  or  ./quick_image_faceswap.sh"
echo "If file-type detection fails: sudo apt install libmagic1  (Debian/Ubuntu)"
