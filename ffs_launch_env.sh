#!/usr/bin/env bash
# Activate conda env used by FastFaceSwap launchers. Source this file, then use $FFS_PYTHON.
FFS_CONDA_ENV="${FFS_CONDA_ENV:-PyEditorFromFFS}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [[ -n "${FFS_PYTHON:-}" && -x "$FFS_PYTHON" ]]; then
  export FFS_PYTHON
  return 0 2>/dev/null || exit 0
fi

_activate_conda_env() {
  local conda_base="$1"
  if [[ -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$FFS_CONDA_ENV"
    FFS_PYTHON="$(command -v python)"
    return 0
  fi
  if [[ -x "$conda_base/envs/$FFS_CONDA_ENV/bin/python" ]]; then
    FFS_PYTHON="$conda_base/envs/$FFS_CONDA_ENV/bin/python"
    # shellcheck source=/dev/null
    source "$conda_base/envs/$FFS_CONDA_ENV/bin/activate"
    return 0
  fi
  return 1
}

if command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$conda_base" ]] && _activate_conda_env "$conda_base"; then
    export FFS_PYTHON
    return 0 2>/dev/null || exit 0
  fi
fi

for conda_base in \
  "$HOME/anaconda3" \
  "$HOME/miniconda3" \
  "$HOME/mambaforge" \
  "/opt/conda"; do
  if [[ -d "$conda_base" ]] && _activate_conda_env "$conda_base"; then
    export FFS_PYTHON
    return 0 2>/dev/null || exit 0
  fi
done

echo "Could not find or activate conda env: $FFS_CONDA_ENV" >&2
echo "Run ./install_conda_linux.sh once to create it." >&2
echo "Or set FFS_PYTHON to your python binary." >&2
return 1 2>/dev/null || exit 1
