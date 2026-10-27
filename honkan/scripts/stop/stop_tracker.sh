#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  exit 0
fi

docker exec "${container}" bash -lc \
  "pkill -TERM -f '[r]os2 launch autoware_multi_object_tracker multi_object_tracker.launch.xml' || true; \
   pkill -TERM -f '^/(opt/autoware|autoware_runtime/install)/.*/multi_object_tracker_node' || true"

echo "Multi Object Tracker stopped."
