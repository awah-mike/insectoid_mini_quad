#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/insectoid_mini_quad"
ISAAC_LAB="/home/ubuntu/IsaacLab/isaaclab.sh"

DISPLAY="${DISPLAY:-:1}" \
XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}" \
TERM="${TERM:-xterm}" \
"${ISAAC_LAB}" -p "${REPO_ROOT}/rl/scripts/render_reversed_joint_trajectory.py" \
  --trajectory-npz "${REPO_ROOT}/data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz" \
  --source actual \
  --start-s 2.46 \
  --end-s 3.04 \
  --duration 6.0 \
  --playback-speed 1.0 \
  --reverse \
  --cycle \
  --start-from-default \
  --front-up-deg -15 \
  --root-yaw-rate-damping 0.2 \
  --camera-eye 3.8 -0.15 2.5 \
  --camera-lookat 0.0 -0.45 0.12 \
  --fps 50 \
  --width 1280 \
  --height 720 \
  --headless \
  --output "${REPO_ROOT}/outputs/renders/model_1399_refined_clean_reverse_actual_246_304_side_iso.mp4" \
  --summary-output "${REPO_ROOT}/outputs/renders/model_1399_refined_clean_reverse_actual_246_304_side_iso.summary.json"
