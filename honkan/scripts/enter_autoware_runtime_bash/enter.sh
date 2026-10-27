#!/usr/bin/env bash
set -euo pipefail

docker exec -it tian_autoware_runtime \
  bash --rcfile /autoware_runtime/scripts/enter_autoware_runtime_bash/container_env.sh -i
