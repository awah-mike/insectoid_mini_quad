#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/insectoid_mini_quad"
ISAACLAB_PATH="/home/ubuntu/IsaacLab"
OUTPUT_DIR="${REPO_ROOT}/renders"

DISPLAY="${DISPLAY:-:1}" \
XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}" \
TERM="${TERM:-xterm}" \
"${ISAACLAB_PATH}/isaaclab.sh" -p \
"${REPO_ROOT}/rl/scripts/render_reversed_joint_trajectory.py" \
  --headless \
  --device cuda:0 \
  --trajectory-npz "${REPO_ROOT}/data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz" \
  --source actual \
  --reverse \
  --cycle \
  --start-from-default \
  --root-z-offset 0.04 \
  --playback-speed 1.0 \
  --start-s 2.46 \
  --end-s 3.04 \
  --duration 6.0 \
  --fps 50 \
  --width 1280 \
  --height 720 \
  --camera-eye 3.8 -0.15 2.5 \
  --camera-lookat 0.0 -0.20 0.12 \
  --output "${OUTPUT_DIR}/model_1399_reversed_actual_joint_traj_v1_side_iso.mp4" \
  --summary-output "${OUTPUT_DIR}/model_1399_reversed_actual_joint_traj_v1_side_iso.summary.json"
