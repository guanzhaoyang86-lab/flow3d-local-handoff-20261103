#!/usr/bin/env bash
set -euo pipefail
package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec "${FLOW3D_PYTHON:-python3}" -B "$package_dir/run_generation.py" "$@"
