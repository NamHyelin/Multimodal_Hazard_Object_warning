#!/usr/bin/env bash
set -euo pipefail
if docker exec tian_autoware_runtime pgrep -f '^python3 -u /autoware_runtime/scripts/perception_web_monitor/monitor.py$' >/dev/null; then
  echo 'Monitor already running'; exit 0
fi
docker exec tian_autoware_runtime bash -lc '
  p=/autoware_runtime/scripts/perception_web_monitor/monitor.py
  if pgrep -f "^python3 -u ${p}$" >/dev/null; then
    echo "Monitor already running"; exit 0
  fi
  mkdir -p /autoware_runtime/logs/perception_web_monitor
'
docker exec -d tian_autoware_runtime bash -lc '
  source /autoware_runtime/scripts/perception_standard_v2/env.sh
  export CYCLONEDDS_URI=file:///autoware_runtime/config/perception_web_monitor/cyclonedds.xml
  exec python3 -u /autoware_runtime/scripts/perception_web_monitor/monitor.py >> /autoware_runtime/logs/perception_web_monitor/monitor.log 2>&1
'
for attempt in {1..20}; do
  if curl --fail --silent --max-time 1 http://127.0.0.1:8088/healthz; then
    echo; echo 'Monitor: http://127.0.0.1:8088/ (remote access via SSH tunnel)'; exit 0
  fi
  sleep 1
done
echo 'Monitor failed: inspect logs/perception_web_monitor/monitor.log' >&2
exit 1
