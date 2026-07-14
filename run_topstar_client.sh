#!/bin/bash
# Helper script to run the G1 loco client
#
# Usage: ./run_topstar_client.sh --stand_up --get_fsm_id
#
# Uses our own g1_loco_client_test binary which is built with the same
# CycloneDDS library as the topstar_bridge, ensuring DDS compatibility.
#
# The external topstar_sdk2 g1_loco_client uses a different CycloneDDS version
# (0.10.2 vs system 0.10.5) which causes DDS type descriptor mismatches.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/build_sim/bin/h2_loco_client_test" "$@"
