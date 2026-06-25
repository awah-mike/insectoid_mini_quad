# model_1399_reversed_actual_joint_traj_v1

Canonical label: `model_1399_reversed_actual_joint_traj_v1`

This is the first backwards primitive built exactly from the canonical forward primitive, without training a new policy. It records the post-policy forward rollout from `model_1399_forward_recycled_stride_v1`, extracts the achieved joint positions, reverses a selected time window, and replays those joint positions as targets.

## Source

- Base checkpoint: `/home/ubuntu/insectoid_mini_quad/trained_models/forward_strict_best/model_1399.pt`
- Forward primitive: `model_1399_forward_recycled_stride_v1`
- Forward trajectory dump: `/home/ubuntu/insectoid_mini_quad/data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz`
- Replay script: `/home/ubuntu/insectoid_mini_quad/rl/scripts/render_reversed_joint_trajectory.py`

## Method

1. Rendered the canonical forward primitive with the same post-policy cleanup stack: BL distal mirroring, rear stride IK, action recycling, and yaw stabilization.
2. Added `--trajectory-output` to dump the actual achieved joint positions, target joint positions, post-policy actions, root velocity, root pose, and contacts.
3. Tested reversing three possible trajectory sources:
   - recorded post-policy actions
   - recorded target joint positions
   - achieved actual joint positions
4. The action and target reversals still moved forward in simulation. The achieved joint-position reversal was the only one that produced backward body displacement.
5. Scanned repeated windows from the canonical forward rollout. The best no-contact backwards result came from reversing `2.46s` to `3.04s`.

## Replay Settings

- Source: achieved joint positions, `actual_joint_pos`
- Time window: `2.46s` to `3.04s`
- Direction: reversed
- Looping: enabled
- Start pose: default standing pose
- Playback speed: `1.0`
- Root velocity injection: disabled
- Root yaw-rate damping: disabled

## Current Render

- Video: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_reversed_actual_joint_traj_v1_side_iso.mp4`
- Contact sheet: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_reversed_actual_joint_traj_v1_side_iso_contact_sheet.png`
- Summary: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_reversed_actual_joint_traj_v1_side_iso.summary.json`

## Metrics

- World-Y displacement over 6s: `-0.266 m`
- Mean body-Y velocity: `-0.052 m/s`
- Roll peak-to-peak: `6.45 deg`
- Pitch peak-to-peak: `11.23 deg`
- Yaw drift peak-to-peak: `38.33 deg`
- Non-foot contact fraction: `0.0`

## Notes

- This is a direct reversed-joint replay primitive, not a learned reverse policy.
- It is physically valid and contact-clean, but slow. The forward gait's useful displacement is not fully invertible by simply reversing actions or target positions, because the contacts and body momentum do not reverse automatically.
- The best next improvement is to use this reversed achieved-joint sequence as a seed, then add small foot-space corrections to increase backward stance impulse while preserving the reversed visual gait.
