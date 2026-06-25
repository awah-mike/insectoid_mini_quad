# model_1399_reversed_actual_joint_traj_reclined_yawdamp_v2

Canonical label: `model_1399_reversed_actual_joint_traj_reclined_yawdamp_v2`

This is the second backwards primitive built from the canonical forward primitive without training a reverse policy. It reuses the achieved joint trajectory from `model_1399_forward_recycled_stride_v1`, reverses the best backward-producing time window, then adds a render-time body recline and yaw-rate damper to improve reverse stability.

## Source

- Base checkpoint: `/home/ubuntu/insectoid_mini_quad/trained_models/forward_strict_best/model_1399.pt`
- Forward primitive: `model_1399_forward_recycled_stride_v1`
- Forward trajectory dump: `/home/ubuntu/insectoid_mini_quad/data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz`
- Replay script: `/home/ubuntu/insectoid_mini_quad/rl/scripts/render_reversed_joint_trajectory.py`

## Method

1. Start from the v1 reverse primitive: replay achieved joint positions from the canonical forward gait, reversed over the `2.46s` to `3.04s` window.
2. Test body recline angles in both signs. Positive values made the reverse primitive worse and caused more roll/yaw excursion.
3. Select the useful recline sign: `--front-up-deg -15`. In this replay script's axis convention, this is the direction that shifts the body toward the rear support region during reverse playback.
4. Add root yaw-rate damping at `0.20` to reduce yaw drift while preserving useful backward displacement.

## Replay Settings

- Source: achieved joint positions, `actual_joint_pos`
- Time window: `2.46s` to `3.04s`
- Direction: reversed
- Looping: enabled
- Start pose: default standing pose
- Playback speed: `1.0`
- Root velocity injection: disabled
- Root yaw-rate damping: `0.20`
- Body recline command: `--front-up-deg -15`

## Current Render

- Video: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_reversed_actual_joint_traj_reclined_yawdamp_v2_side_iso.mp4`
- Contact sheet: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_reversed_actual_joint_traj_reclined_yawdamp_v2_side_iso_contact_sheet.png`
- Summary: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_reversed_actual_joint_traj_reclined_yawdamp_v2_side_iso.summary.json`

## Metrics

- World-Y displacement over 6s: `-0.313 m`
- Mean body-Y velocity: `-0.057 m/s`
- Roll peak-to-peak: `17.40 deg`
- Pitch peak-to-peak: `10.83 deg`
- Yaw drift peak-to-peak: `29.07 deg`
- Non-foot contact fraction: `0.0`

## Comparison To v1

- v1 displacement: `-0.266 m`; v2 displacement: `-0.313 m`
- v1 yaw drift: `38.33 deg`; v2 yaw drift: `29.07 deg`
- v1 non-foot contact fraction: `0.0`; v2 non-foot contact fraction: `0.0`

## Notes

- This is still a hardcoded post-training replay primitive, not a learned reverse policy.
- The body recline is a render-time root pose/velocity intervention. It is useful for visual motion primitives, but it should not be treated as a policy checkpoint.
- Larger recline angles such as `-20` and `-25` increased instability or introduced body contacts, so `-15` is the selected stable limit from this sweep.
