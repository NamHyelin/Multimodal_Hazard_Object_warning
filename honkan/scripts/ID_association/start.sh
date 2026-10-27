#!/usr/bin/env bash
set -euo pipefail
docker exec tian_autoware_runtime bash -lc '
  if pgrep -f "^python3 -u /autoware_runtime/scripts/ID_association/node.py$" >/dev/null; then
    echo "ID association already running"; exit 0
  fi
  mkdir -p /autoware_runtime/logs/ID_association
  source /autoware_runtime/scripts/perception_standard_v2/env.sh
  python3 -c "import scipy, cv2"
'
if ! docker exec tian_autoware_runtime pgrep -f '^python3 -u /autoware_runtime/scripts/ID_association/node.py$' >/dev/null; then
  docker exec -d tian_autoware_runtime bash -lc '
    source /autoware_runtime/scripts/perception_standard_v2/env.sh
    exec python3 -u /autoware_runtime/scripts/ID_association/node.py >> /autoware_runtime/logs/ID_association/node.log 2>&1
  '
fi
for attempt in {1..15}; do
  if docker exec tian_autoware_runtime pgrep -af '^python3 -u /autoware_runtime/scripts/ID_association/node.py$'; then
    exit 0
  fi
  sleep 1
done
echo 'Association failed to start; inspect logs/ID_association/node.log' >&2
exit 1
