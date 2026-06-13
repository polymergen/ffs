#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=ffs_launch_env.sh
. "./ffs_launch_env.sh"
exec "$FFS_PYTHON" "$PWD/main.py" "$@"
