#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MODEL_BASE="https://github.com/RichardErkhov/FastFaceSwap/releases/download/model"
for f in inswapper_128.onnx GFPGANv1.4.pth complex_256_v7_stage3_12999.h5 GFPGANv1.4.onnx; do
  if [[ -f "$f" ]]; then
    echo "Already present: $f"
  else
    echo "Downloading $f ..."
    if command -v wget >/dev/null 2>&1; then
      wget -q --show-progress -O "$f" "$MODEL_BASE/$f"
    else
      curl -L -o "$f" "$MODEL_BASE/$f"
    fi
  fi
done
