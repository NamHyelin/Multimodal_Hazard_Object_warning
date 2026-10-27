#!/usr/bin/env bash
set -euo pipefail
docker exec tian_autoware_runtime bash -lc 'pkill -INT -f "^python3 -u /autoware_runtime/scripts/perception_web_monitor/monitor.py$" || true'
for attempt in {1..30}; do
  if ! docker exec tian_autoware_runtime pgrep -f '^python3 -u /autoware_runtime/scripts/perception_web_monitor/monitor.py$' >/dev/null; then
    echo 'Monitor stopped.'; exit 0
  fi
  sleep 0.2
done
echo 'Monitor still stopping; inspect status before restarting.' >&2
exit 1
