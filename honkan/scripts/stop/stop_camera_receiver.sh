#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
if docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  docker exec "${container}" pkill -TERM -f '^/autoware_runtime/install/(.*/)?camera_mjpeg_receiver/lib/camera_mjpeg_receiver/camera_mjpeg_receiver_node' 2>/dev/null || true
  docker exec "${container}" pkill -TERM -f '[f]fmpeg.*(gate_camera|intersection_camera)\.sdp' 2>/dev/null || true
  sleep 1
  docker exec "${container}" pkill -KILL -f '^/autoware_runtime/install/(.*/)?camera_mjpeg_receiver/lib/camera_mjpeg_receiver/camera_mjpeg_receiver_node' 2>/dev/null || true
  docker exec "${container}" pkill -KILL -f '[f]fmpeg.*(gate_camera|intersection_camera)\.sdp' 2>/dev/null || true
fi
echo "Camera receiver stopped"
