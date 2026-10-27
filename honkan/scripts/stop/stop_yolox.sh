#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  exit 0
fi

docker exec "${container}" bash -lc \
  "pkill -TERM -f '[r]os2 launch autoware_tensorrt_yolox yolox.launch.xml' || true; \
   pkill -TERM -f '^/autoware_runtime/install/autoware_tensorrt_yolox/.*/autoware_tensorrt_yolox_node_exe' || true; \
   pkill -TERM -f '[r]un_yolox.sh' || true"

echo "Dual-camera YOLOX stopped."
