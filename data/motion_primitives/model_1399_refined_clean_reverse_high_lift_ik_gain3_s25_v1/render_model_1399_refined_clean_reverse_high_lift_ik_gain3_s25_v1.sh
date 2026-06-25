#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export TERM="${TERM:-xterm}"

cd /home/ubuntu/insectoid_mini_quad

/home/ubuntu/IsaacLab/isaaclab.sh -p rl/scripts/render_reversed_joint_trajectory.py --headless \
  --trajectory-npz /home/ubuntu/insectoid_mini_quad/data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz \
  --output /home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_refined_clean_reverse_high_lift_ik_gain3_s25_side_iso_close.mp4 \
  --summary-output /home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_refined_clean_reverse_high_lift_ik_gain3_s25_side_iso_close.summary.json \
  --source actual \
  --start-s 1.30 \
  --end-s 1.88 \
  --duration 6.0 \
  --playback-speed 2.5 \
  --reverse \
  --cycle \
  --start-from-default \
  --front-up-deg -15 \
  --root-z-offset 0.12 \
  --root-yaw-rate-damping 0.4 \
  --yaw-feedback-kp 0.45 \
  --yaw-rate-feedback-kd 0.70 \
  --max-yaw-feedback 0.35 \
  --root-body-vel-y-servo-target -0.52 \
  --root-body-vel-y-servo-alpha 1.0 \
  --root-body-vel-y-servo-lift-relief 0.15 \
  --swing-lift-height-m 0.040 \
  --swing-lift-rear-scale 1.0 \
  --swing-lift-middle-scale 2.5 \
  --swing-lift-max-action-delta 0.95 \
  --swing-lift-action-gain 3.0 \
  --swing-lift-dilate-steps 2 \
  --camera-eye 3.2 -1.15 2.2 \
  --camera-lookat 0.0 -1.2 0.12
