#!/usr/bin/env bash

# Shared environment for commands executed inside tian_autoware_runtime.
# When this file is used as an interactive Bash rc file, retain the container's
# normal prompt and aliases, and enable programmable completion before ROS
# registers its completers. Non-interactive run scripts skip this block.
if [[ $- == *i* ]]; then
  if [[ -r /root/.bashrc ]]; then
    source /root/.bashrc
  fi
  if [[ -r /etc/bash_completion ]]; then
    source /etc/bash_completion
  fi
fi

set +u
source /opt/ros/humble/setup.bash
source /opt/autoware/setup.bash
if [[ ! -f /autoware_runtime/install/setup.bash ]]; then
  echo "Missing runtime overlay: /autoware_runtime/install/setup.bash" >&2
  return 1 2>/dev/null || exit 1
fi
source /autoware_runtime/install/setup.bash
if [[ -f /autoware_runtime/config/transport_overlay.env ]]; then
  source /autoware_runtime/config/transport_overlay.env
  source "${PERCEPTION_TRANSPORT_OVERLAY}/local_setup.bash"
fi
set -u

# Autoware setup appends its own library directories. Put the R39-compatible
# TensorRT 10.16 runtime first for the rebuilt CenterPoint overlay.
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
