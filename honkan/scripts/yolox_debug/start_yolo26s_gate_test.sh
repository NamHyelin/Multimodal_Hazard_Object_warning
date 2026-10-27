#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(cd -- "${script_dir}/../.." && pwd)"
log_file="${runtime_dir}/logs/yolo26s_test.log"

if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  echo "${container} is not running" >&2
  exit 1
fi

"${script_dir}/stop_yolo26s_gate_test.sh" >/dev/null

docker exec -d "${container}" bash -lc '
  source /autoware_runtime/scripts/enter_autoware_runtime_bash/container_env.sh
  exec ros2 run autoware_tensorrt_yolox autoware_tensorrt_yolox_node_exe \
    --ros-args \
    --params-file /autoware_runtime/yolo_benchmark/yolo26s_autoware.param.yaml \
    -p model_path:=/autoware_runtime/yolo_benchmark/yolo26s_autoware.onnx \
    -p label_path:=/autoware_runtime/yolo_benchmark/coco80_labels.txt \
    -p roi_remap_path:=/autoware_runtime/yolo_benchmark/coco80_to_autoware.csv \
    -r __node:=tensorrt_yolo26s_gate_test \
    -r "~/in/image:=/rsu1/gate/image_raw" \
    -r "~/out/objects:=/rsu1/gate/yolo26s/objects" \
    -r "~/out/image:=/tensorrt_yolo26s_gate_test/out/image" \
    >> /autoware_runtime/logs/yolo26s_test.log 2>&1
'

for _ in $(seq 1 20); do
  if docker exec "${container}" bash -lc '
    source /autoware_runtime/scripts/enter_autoware_runtime_bash/container_env.sh >/dev/null 2>&1
    ros2 node list 2>/dev/null | grep -qx /tensorrt_yolo26s_gate_test
  '; then
    echo "YOLO26s gate test node started. Log: ${log_file}"
    exit 0
  fi
  sleep 1
done

echo "YOLO26s gate test node failed to start. Check ${log_file}" >&2
exit 1
