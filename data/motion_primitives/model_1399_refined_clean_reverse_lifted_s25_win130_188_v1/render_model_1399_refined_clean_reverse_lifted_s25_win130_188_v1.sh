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
  --start-s 1.30 \
  --end-s 1.88 \
  --duration 6.0 \
  --playback-speed 2.5 \
  --reverse \
  --cycle \
  --start-from-default \
  --front-up-deg -15 \
  --root-z-offset 0.085 \
  --root-yaw-rate-damping 0.4 \
  --yaw-feedback-kp 0.45 \
  --yaw-rate-feedback-kd 0.70 \
  --max-yaw-feedback 0.35 \
  --root-body-vel-y-servo-target -0.45 \
  --root-body-vel-y-servo-alpha 1.0 \
  --swing-lift-height-m 0.03 \
  --swing-lift-joints femur_tibia \
  --swing-lift-rear-scale 1.0 \
  --swing-lift-middle-scale 2.0 \
  --swing-lift-max-action-delta 0.55 \
  --camera-eye 4.6 -1.2 3.0 \
  --camera-lookat 0.0 -1.2 0.12 \
  --fps 50 \
  --width 1280 \
  --height 720 \
  --headless \
  --output "${REPO_ROOT}/outputs/renders/model_1399_refined_clean_reverse_lifted_s25_win130_188_side_iso.mp4" \
  --summary-output "${REPO_ROOT}/outputs/renders/model_1399_refined_clean_reverse_lifted_s25_win130_188_side_iso.summary.json"
