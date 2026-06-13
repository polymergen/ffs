#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL_BASE="https://github.com/RichardErkhov/FastFaceSwap/releases/download/model"

download_model() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "Already present: $out"
    return 0
  fi
  echo "Downloading $out..."
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$out" "$url"
  else
    echo "Install wget or curl to download model files." >&2
    exit 1
  fi
}

download_model "$MODEL_BASE/inswapper_128.onnx" "inswapper_128.onnx"
download_model "$MODEL_BASE/GFPGANv1.4.pth" "GFPGANv1.4.pth"
download_model "$MODEL_BASE/complex_256_v7_stage3_12999.h5" "complex_256_v7_stage3_12999.h5"
download_model "$MODEL_BASE/GFPGANv1.4.onnx" "GFPGANv1.4.onnx"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv venv
# shellcheck source=/dev/null
source venv/bin/activate

pip install torch torchvision torchaudio
pip install onnxruntime-gpu || pip install onnxruntime
grep -v '^python-magic-bin' requirements.txt | pip install -r /dev/stdin
pip install python-magic
pip install tensorflow==2.10.1 || pip install tensorflow
pip install protobuf==3.20.2
pip uninstall -y opencv-python opencv-headless-python opencv-contrib-python 2>/dev/null || true
pip install opencv-python

echo ""
echo "Install complete. Run:"
echo "  ./fastfaceswap.sh          # full GUI app"
echo "  ./quick_image_faceswap.sh  # quick image swap"
echo ""
echo "If file-type detection fails, install libmagic (e.g. apt install libmagic1)."
