#!/bin/bash
# MuJoCo Viewer for H2 Robot (topstar_mujoco simulator)
#
# Usage:
#   ./run_mujoco_viewer.sh              # Start with loopback networking
#   ./run_mujoco_viewer.sh -n enp3s0    # Start with specific interface
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUJOCO_DIR="${SCRIPT_DIR}/../topstar_mujoco/simulate"

if [ ! -f "${MUJOCO_DIR}/build/topstar_mujoco" ]; then
  echo "Error: topstar_mujoco not found at ${MUJOCO_DIR}/build/topstar_mujoco"
  echo "Build it first: cd ${MUJOCO_DIR} && mkdir -p build && cd build && cmake .. && make"
  exit 1
fi

# Default to loopback if no -n flag provided
if [[ ! " $* " =~ " -n " ]]; then
  exec "${MUJOCO_DIR}/build/topstar_mujoco" -n lo "$@"
else
  exec "${MUJOCO_DIR}/build/topstar_mujoco" "$@"
fi
