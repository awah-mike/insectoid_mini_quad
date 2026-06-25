# model_1399_raw_reverse_action_tape_yaw20_v1

Canonical label: `model_1399_raw_reverse_action_tape_yaw20_v1`

This primitive reverses our local `model_1399.pt` using the same mechanism discovered in the transport handoff: record a clean forward action tape, flip the action sequence in time, and replay it as normalized actions through physics.

## Source

- Base checkpoint: `/home/ubuntu/insectoid_mini_quad/trained_models/forward_strict_best/model_1399.pt`
- Forward raw tape capture: `/home/ubuntu/insectoid_mini_quad/data/trajectories/reversed_trajectory/model_1399_raw_forward_tape_capture.trajectory.npz`
- Replay script: `/home/ubuntu/insectoid_mini_quad/rl/scripts/render_reversed_joint_trajectory.py`

## Method

1. Captured a raw model-1399 forward rollout with `forward_vel=0.30`.
2. Disabled the canonical post-policy cleanup stack during capture:
   - no action recycling
   - no BL mirroring
   - no stride IK
   - no yaw feedback
   - no action target filtering
3. Used the first 80 steps as warmup.
4. Replayed the following 180-step action tape backward in time, matching the handoff method.
5. Added root yaw-rate damping `0.20` for the rendered primitive.

## Important Finding

The post-processed canonical forward tape from `model_1399_forward_recycled_stride_v1` does not reverse correctly: flipping that tape in time still moves the robot forward. The raw model-1399 policy action tape does reverse into a valid backwards gait.

## Replay Settings

- Source: raw policy actions
- Direction: reversed in time
- Time window: `1.60s` to `5.20s`
- Looping: enabled
- Start pose: default standing pose
- Playback speed: `1.0`
- Root yaw-rate damping: `0.20`

## Current Render

- Video: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_raw_reverse_action_tape_yaw20_side_iso.mp4`
- Contact sheet: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_raw_reverse_action_tape_yaw20_side_iso_contact_sheet.png`
- Summary: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_raw_reverse_action_tape_yaw20_side_iso.summary.json`

## Metrics

- World-Y displacement over 6s: `-0.722 m`
- Mean body-Y velocity: `-0.138 m/s`
- Roll peak-to-peak: `13.69 deg`
- Pitch peak-to-peak: `11.06 deg`
- Yaw drift peak-to-peak: `30.62 deg`
- Non-foot contact fraction: `0.0`

## Sweep Notes

- Exact raw reversed tape, no yaw damping: `-0.553 m`, yaw drift `43.50 deg`
- Raw reversed tape with yaw damping `0.20`: `-0.718 m` in telemetry, rendered at `-0.722 m`, yaw drift about `29-31 deg`
- Raw reversed tape with yaw damping `0.40`: `-0.797 m`, but yaw drift increased to `39.72 deg`

`0.20` is selected as the cleaner visual base.
