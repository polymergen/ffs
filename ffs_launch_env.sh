#!/usr/bin/env bash
# Resolve Python for FastFaceSwap launchers. Source this file, then use $FFS_PYTHON.
FFS_CONDA_ENV="${FFS_CONDA_ENV:-ffs-gold-2024-09}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [[ -n "${FFS_PYTHON:-}" && -x "$FFS_PYTHON" ]]; then
  :
elif [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
  FFS_PYTHON="$SCRIPT_DIR/venv/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  FFS_PYTHON="$CONDA_PREFIX/bin/python"
else
  for base in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/mambaforge" "/opt/conda"; do
    if [[ -x "$base/envs/$FFS_CONDA_ENV/bin/python" ]]; then
      FFS_PYTHON="$base/envs/$FFS_CONDA_ENV/bin/python"
      break
    fi
  done
fi

if [[ -z "${FFS_PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    FFS_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    FFS_PYTHON="$(command -v python)"
  fi
fi

if [[ -z "${FFS_PYTHON:-}" || ! -x "$FFS_PYTHON" ]]; then
  echo "Could not find Python for FastFaceSwap." >&2
  echo "Run ./install_linux.sh, set FFS_PYTHON, or create conda env: $FFS_CONDA_ENV" >&2
  return 1 2>/dev/null || exit 1
fi

export FFS_PYTHON
