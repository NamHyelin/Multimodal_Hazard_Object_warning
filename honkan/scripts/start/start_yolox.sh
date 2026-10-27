#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(cd -- "${script_dir}/../.." && pwd)"

if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  echo "${container} is not running" >&2
  exit 1
fi

"${runtime_dir}/scripts/stop/stop_yolox.sh" >/dev/null
for _ in $(seq 1 10); do
  if ! docker exec "${container}" bash -lc \
    "pgrep -f '^/autoware_runtime/install/autoware_tensorrt_yolox/.*/autoware_tensorrt_yolox_node_exe' >/dev/null"; then
    break
  fi
  sleep 1
done

log_file="${runtime_dir}/logs/yolox.log"
if [[ -f "${log_file}" ]] && [[ $(stat -c %s "${log_file}") -gt 52428800 ]]; then
  mv -f "${log_file}" "${log_file}.1"
fi

docker exec -d "${container}" bash -lc \
  'exec /autoware_runtime/scripts/run/run_yolox.sh >> /autoware_runtime/logs/yolox.log 2>&1'

for _ in $(seq 1 30); do
  process_count=$(docker exec "${container}" bash -lc \
    "pgrep -fc '^/autoware_runtime/install/autoware_tensorrt_yolox/.*/autoware_tensorrt_yolox_node_exe'" || true)
  if [[ "${process_count}" -ge 2 ]]; then
    echo "Dual-camera YOLOX started. Log: ${runtime_dir}/logs/yolox.log"
    exit 0
  fi
  sleep 1
done

echo "Dual-camera YOLOX failed to start. Check ${runtime_dir}/logs/yolox.log" >&2
exit 1
