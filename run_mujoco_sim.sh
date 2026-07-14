#!/bin/bash
# MuJoCo Simulation Bridge for H2 Robot
#
# Standalone simulation (no hardware):
#   ./run_mujoco_sim.sh
#
# Digital twin mirror (with hardware running):
#   ./run_mujoco_sim.sh --mirror
#
# Prerequisites:
#   Terminal 1: Start MuJoCo simulator
#     ./run_mujoco_viewer.sh
#
#   Terminal 2: Start this bridge
#     ./run_mujoco_sim.sh
#
#   Terminal 3: Run the RL policy
#     python3 python/h2_amp_rl_runner.py --sim

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="${SCRIPT_DIR}/build_sim/lib:${LD_LIBRARY_PATH}"

exec "${SCRIPT_DIR}/build_sim/bin/mujoco_sim_bridge_v2" --network_interface=lo "$@"
