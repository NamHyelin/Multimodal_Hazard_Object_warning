#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(cd -- "${script_dir}/../.." && pwd)"
output_dir="${runtime_dir}/captures/yolox"

if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  echo "${container} is not running" >&2
  exit 1
fi

mkdir -p "${output_dir}"

docker exec "${container}" bash -lc \
  'source /autoware_runtime/scripts/enter_autoware_runtime_bash/container_env.sh
   exec python3 /autoware_runtime/scripts/yolox_debug/capture_yolox_frame.py "$@"' \
  bash --output-dir /autoware_runtime/captures/yolox "$@"

echo "Captured YOLOX images are available in: ${output_dir}"
