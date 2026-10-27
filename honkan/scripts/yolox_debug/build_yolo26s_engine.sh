#!/usr/bin/env bash
set -euo pipefail

container=tian_autoware_runtime
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(cd -- "${script_dir}/../.." && pwd)"
onnx_path="${runtime_dir}/yolo_benchmark/yolo26s_autoware.onnx"
engine_path="${runtime_dir}/yolo_benchmark/yolo26s_autoware.engine"

if [[ ! -f "${onnx_path}" ]]; then
  echo "Missing compatible ONNX: ${onnx_path}" >&2
  exit 1
fi
if ! docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null | grep -qx true; then
  echo "${container} is not running" >&2
  exit 1
fi

docker exec "${container}" bash -lc '
  source /autoware_runtime/scripts/enter_autoware_runtime_bash/container_env.sh
  exec ros2 run autoware_tensorrt_yolox autoware_tensorrt_yolox_node_exe \
    --ros-args \
    --params-file /autoware_runtime/yolo_benchmark/yolo26s_autoware.param.yaml \
    -p model_path:=/autoware_runtime/yolo_benchmark/yolo26s_autoware.onnx \
    -p label_path:=/autoware_runtime/yolo_benchmark/coco80_labels.txt \
    -p roi_remap_path:=/autoware_runtime/yolo_benchmark/coco80_to_autoware.csv \
    -p build_only:=true \
    -r __node:=tensorrt_yolo26s_builder
'

if [[ ! -f "${engine_path}" ]]; then
  echo "TensorRT build finished without creating ${engine_path}" >&2
  exit 1
fi
echo "YOLO26s TensorRT engine ready: ${engine_path}"
