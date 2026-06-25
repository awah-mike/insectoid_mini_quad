#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export TERM="${TERM:-xterm}"

cd /home/ubuntu/insectoid_mini_quad

/home/ubuntu/IsaacLab/isaaclab.sh -p rl/scripts/render_reversed_joint_trajectory.py --headless \
  --trajectory-npz /home/ubuntu/insectoid_mini_quad/data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz \
  --output /home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_forward_canonical_sideways_xneg_clean_s25_front_iso.mp4 \
  --summary-output /home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_forward_canonical_sideways_xneg_clean_s25_front_iso.summary.json \
  --source actual --start-s 0.72 --end-s 1.30 --duration 6.0 --playback-speed 2.5 \
  --cycle --start-from-default --root-z-offset 0.085 \
  --root-yaw-rate-damping 0.75 --yaw-feedback-sign 1.0 --yaw-feedback-kp 0.85 --yaw-rate-feedback-kd 1.30 --max-yaw-feedback 0.55 \
  --root-body-vel-x-servo-target -0.30 --root-body-vel-x-servo-alpha 1.0 \
  --root-body-vel-y-servo-target 0.0 --root-body-vel-y-servo-alpha 1.0 \
  --sideways-remap-scale 1.0 --sideways-remap-sign -1.0 --sideways-remap-y-keep-scale 0.10 --sideways-remap-x-keep-scale 1.0 --sideways-remap-max-action-delta 0.65 \
  --swing-lift-height-m 0.020 --swing-lift-rear-scale 1.0 --swing-lift-middle-scale 1.5 --swing-lift-max-action-delta 0.45 --swing-lift-dilate-steps 1 \
  --camera-eye -0.8 -4.2 2.8 --camera-lookat -0.8 0.0 0.12
