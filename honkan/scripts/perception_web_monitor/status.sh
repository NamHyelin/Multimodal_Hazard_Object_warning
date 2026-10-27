#!/usr/bin/env bash
set -euo pipefail
docker exec tian_autoware_runtime pgrep -af '^python3 -u /autoware_runtime/scripts/perception_web_monitor/monitor.py$' || true
curl --fail --silent --max-time 3 http://127.0.0.1:8088/api/status | python3 -m json.tool
