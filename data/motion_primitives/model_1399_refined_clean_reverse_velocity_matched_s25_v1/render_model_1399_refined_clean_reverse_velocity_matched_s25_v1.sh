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
  --playback-speed 2.5 \
  --reverse \
  --cycle \
  --start-from-default \
  --front-up-deg -15 \
  --root-yaw-rate-damping 0.2 \
  --yaw-feedback-kp 0.45 \
  --yaw-rate-feedback-kd 0.70 \
  --max-yaw-feedback 0.35 \
  --root-body-vel-y-servo-target -0.45 \
  --root-body-vel-y-servo-alpha 1.0 \
  --camera-eye 4.6 -1.2 3.0 \
  --camera-lookat 0.0 -1.2 0.12 \
  --fps 50 \
  --width 1280 \
  --height 720 \
  --headless \
  --output "${REPO_ROOT}/outputs/renders/model_1399_refined_clean_reverse_velocity_matched_s25_side_iso.mp4" \
  --summary-output "${REPO_ROOT}/outputs/renders/model_1399_refined_clean_reverse_velocity_matched_s25_side_iso.summary.json"
