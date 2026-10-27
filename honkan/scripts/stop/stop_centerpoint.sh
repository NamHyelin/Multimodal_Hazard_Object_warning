#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  exit 0
fi

docker exec "${container}" bash -lc \
  "pkill -TERM -f '[r]os2 launch autoware_lidar_centerpoint lidar_centerpoint.launch.xml' || true; \
   pkill -TERM -f '^/autoware_runtime/install/autoware_lidar_centerpoint/.*/autoware_lidar_centerpoint_node' || true"

echo "CenterPoint stopped."

