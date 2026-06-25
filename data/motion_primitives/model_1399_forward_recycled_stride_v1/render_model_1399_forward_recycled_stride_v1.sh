#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ubuntu/insectoid_mini_quad"
ISAACLAB_ROOT="/home/ubuntu/IsaacLab"
DISPLAY="${DISPLAY:-:1}"
XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
TERM="${TERM:-xterm}"

OUTPUT_DIR="${REPO_ROOT}/renders"

DISPLAY="${DISPLAY}" \
XAUTHORITY="${XAUTHORITY}" \
TERM="${TERM}" \
"${ISAACLAB_ROOT}/isaaclab.sh" -p \
"${REPO_ROOT}/rl/scripts/render_stabilized_policy_video.py" \
  --headless \
  --device cuda:0 \
  --video-length 300 \
  --fps 50 \
  --width 1280 \
  --height 720 \
  --seed 17 \
  --forward-vel 0.60 \
  --action-filter-alpha 1.0 \
  --yaw-feedback-sign 1.0 \
  --yaw-feedback-kp 0.45 \
  --yaw-rate-feedback-kd 0.70 \
  --max-yaw-feedback 0.35 \
  --mirror-bl-from-br \
  --mirror-bl-joints distal \
  --mirror-bl-blend 0.55 \
  --mirror-bl-delay-phase 0.55 \
  --mirror-cycle-steps 17.77777777777778 \
  --stride-ik-profile swing_reach \
  --stride-ik-rear-forward-m -0.080 \
  --stride-ik-middle-forward-m 0.0 \
  --stride-ik-rear-joints coxa \
  --stride-ik-middle-joints coxa_tibia \
  --stride-ik-duration 0.12 \
  --stride-ik-max-action-delta 0.60 \
  --recycle-actions \
  --recycle-start-s 0.72 \
  --recycle-end-s 1.30 \
  --camera-eye 6.2 -0.2 4.0 \
  --camera-lookat 0.0 1.05 0.12 \
  --output "${OUTPUT_DIR}/model_1399_recycled_early_phase_072_130_side_iso.mp4" \
  --attitude-output "${OUTPUT_DIR}/model_1399_recycled_early_phase_072_130_side_iso.attitude.json" \
  --telemetry-output "${OUTPUT_DIR}/model_1399_recycled_early_phase_072_130_side_iso.telemetry.csv"
