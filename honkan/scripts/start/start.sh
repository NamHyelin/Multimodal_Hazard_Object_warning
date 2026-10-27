#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${RUNTIME_DIR}"

if [ ! -f install/setup.bash ]; then
  echo "runtime overlay 尚未构建：${RUNTIME_DIR}/install/setup.bash 不存在" >&2
  echo "请先运行：docker exec tian_autoware_runtime bash -lc '/autoware_runtime/scripts/build/build.sh'" >&2
  exit 1
fi

docker compose up -d autoware_runtime rslidar
docker compose ps autoware_runtime rslidar
"${SCRIPT_DIR}/start_camera_receiver.sh"
"${SCRIPT_DIR}/start_lidar_fusion.sh"
"${SCRIPT_DIR}/start_centerpoint.sh"
"${SCRIPT_DIR}/start_tracker.sh"
"${SCRIPT_DIR}/start_yolox.sh"

echo
echo "查看雷达日志：docker logs -f tian_rslidar"
echo "查看相机日志：tail -f ${RUNTIME_DIR}/logs/camera_receiver_cpp.log"
echo "查看融合日志：tail -f ${RUNTIME_DIR}/logs/lidar_fusion.log"
echo "查看 CenterPoint 日志：tail -f ${RUNTIME_DIR}/logs/centerpoint.log"
echo "查看 Tracker 日志：tail -f ${RUNTIME_DIR}/logs/tracker.log"
echo "查看 YOLOX 日志：tail -f ${RUNTIME_DIR}/logs/yolox.log"
echo "检查运行状态：${SCRIPT_DIR}/status.sh"
echo "进入 Autoware：${RUNTIME_DIR}/scripts/enter_autoware_runtime_bash/enter.sh"
