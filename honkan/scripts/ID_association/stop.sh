#!/usr/bin/env bash
set -euo pipefail
docker exec tian_autoware_runtime bash -lc 'pkill -INT -f "^python3 -u /autoware_runtime/scripts/ID_association/node.py$" || true'
for attempt in {1..30}; do
  if ! docker exec tian_autoware_runtime pgrep -f '^python3 -u /autoware_runtime/scripts/ID_association/node.py$' >/dev/null; then
    echo 'ID association stopped'; exit 0
  fi
  sleep .2
done
echo 'ID association still stopping' >&2
exit 1
