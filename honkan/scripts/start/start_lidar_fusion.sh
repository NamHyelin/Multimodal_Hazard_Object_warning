#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(cd -- "${script_dir}/../.." && pwd)"

if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  echo "${container} is not running" >&2
  exit 1
fi

docker exec "${container}" bash -lc \
  "pkill -f '[c]oncatenate_and_time_sync_node' || true; pkill -f '[x]yzi_to_xyzirc_node' || true; pkill -f '[s]trict_front_fusion_node' || true; pkill -f '[s]tatic_transform_publisher.*lidar_fusion' || true; pkill -f '[r]un_lidar_fusion.sh' || true"

log_file="${runtime_dir}/logs/lidar_fusion.log"
if [[ -f "${log_file}" ]] && [[ $(stat -c %s "${log_file}") -gt 52428800 ]]; then
  mv -f "${log_file}" "${log_file}.1"
fi

docker exec -d "${container}" bash -lc \
  'exec /autoware_runtime/scripts/run/run_lidar_fusion.sh >> /autoware_runtime/logs/lidar_fusion.log 2>&1'

for _ in $(seq 1 20); do
  if docker exec "${container}" bash -lc \
    "pgrep -f '^/autoware_runtime/install/(.*/)?lidar_fusion_adapter/lib/lidar_fusion_adapter/strict_front_fusion_node' >/dev/null"; then
    echo "LiDAR front fusion started. Log: ${runtime_dir}/logs/lidar_fusion.log"
    exit 0
  fi
  sleep 1
done

echo "LiDAR front fusion failed to start. Check ${runtime_dir}/logs/lidar_fusion.log" >&2
exit 1
