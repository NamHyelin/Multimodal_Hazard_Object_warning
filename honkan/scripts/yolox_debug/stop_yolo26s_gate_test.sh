#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  exit 0
fi

docker exec "${container}" bash -lc \
  "pkill -TERM -f '[t]ensorrt_yolo26s_gate_test' || true"

echo "YOLO26s gate test node stopped."
