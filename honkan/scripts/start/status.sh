#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(cd -- "${script_dir}/../.." && pwd)"
cd "${runtime_dir}"
docker compose ps autoware_runtime rslidar

docker exec tian_autoware_runtime bash -lc '
  source /autoware_runtime/scripts/enter_autoware_runtime_bash/container_env.sh
  echo
  echo "ROS nodes:"
  ros2 node list | grep -E "^/(camera_mjpeg_receiver|lidar_centerpoint|multi_object_tracker|strict_front_fusion|tensorrt_yolox_gate|tensorrt_yolox_intersection)$" || true
  echo
  echo "Camera image publishers:"
  ros2 topic info /rsu1/gate/image_raw || true
  ros2 topic info /rsu1/intersection/image_raw || true
  echo
  echo "Detection publisher:"
  ros2 topic info /perception/object_recognition/detection/centerpoint/objects || true
  echo
  echo "Tracking publisher:"
  ros2 topic info /perception/object_recognition/tracking/objects || true
  echo
  echo "YOLOX gate publisher:"
  ros2 topic info /rsu1/gate/yolox/objects || true
  echo
  echo "YOLOX intersection publisher:"
  ros2 topic info /rsu1/intersection/yolox/objects || true
  echo
  echo "Processes:"
  ps -eo pid,etime,pcpu,pmem,args | grep -E "(autoware_lidar_centerpoint|lidar_fusion_adapter|multi_object_tracker_node|autoware_tensorrt_yolox_node_exe)" | grep -v grep || true
'

echo
echo "The tracker subscribes to CenterPoint and keeps 3D detection active."
