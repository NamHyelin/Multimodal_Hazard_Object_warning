#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(cd -- "${script_dir}/../.." && pwd)"

if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  echo "${container} is not running" >&2
  exit 1
fi

"${runtime_dir}/scripts/stop/stop_camera_receiver.sh" >/dev/null
log_file="${runtime_dir}/logs/camera_receiver_cpp.log"
if [[ -f "${log_file}" ]] && [[ $(stat -c %s "${log_file}") -gt 52428800 ]]; then
  mv -f "${log_file}" "${log_file}.1"
fi

docker exec -d "${container}" bash -lc \
  'source /autoware_runtime/scripts/enter_autoware_runtime_bash/container_env.sh; exec ros2 run camera_mjpeg_receiver camera_mjpeg_receiver_node >> /autoware_runtime/logs/camera_receiver_cpp.log 2>&1'

for _ in $(seq 1 20); do
  if docker exec "${container}" pgrep -f '^/autoware_runtime/install/(.*/)?camera_mjpeg_receiver/lib/camera_mjpeg_receiver/camera_mjpeg_receiver_node' >/dev/null; then
    echo "C++ camera receiver started. Log: ${log_file}"
    exit 0
  fi
  sleep 1
done

echo "C++ camera receiver failed to start. Check ${log_file}" >&2
exit 1
