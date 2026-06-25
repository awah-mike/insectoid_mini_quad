# model_1399_refined_clean_reverse_velocity_matched_s25_v1

Reverse-motion primitive derived from the cleaned canonical forward primitive and assisted to match the forward speed magnitude.

Source:

- Forward source label: `model_1399_forward_recycled_stride_v1`
- Base checkpoint: `trained_models/forward_strict_best/model_1399.pt`
- Captured cleaned trajectory: `data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz`
- Replay source: realized joint positions, `source=actual`
- Replay window: `2.46s..3.04s`
- Playback: reversed, cycled, `2.5x`

Reverse post-model cleanups:

- Reuses the cleaned forward trajectory, which already includes the forward yaw stabilizer, BL/BR distal symmetry, stride IK, and early action recycling.
- Adds reverse-side coxa yaw feedback: `kp=0.45`, `kd=0.70`, `max=0.35`.
- Keeps reverse recline/pitch bias: `front_up_deg=-15`.
- Keeps yaw-rate damping: `root_yaw_rate_damping=0.2`.
- Adds a body-frame Y velocity servo: target `-0.45 m/s`, alpha `1.0`.

Important caveat:

The final speed match is produced by a post-model velocity servo that writes the root body-frame Y velocity after each physics step. This is useful for visualization and multi-robot behavior prototyping, but it is not a trained reverse locomotion policy. The unassisted refined reverse replay topped out around `-0.057 m/s`.

Artifacts:

- Video: `outputs/renders/model_1399_refined_clean_reverse_velocity_matched_s25_side_iso.mp4`
- Contact sheet: `outputs/renders/model_1399_refined_clean_reverse_velocity_matched_s25_side_iso_contact_sheet.png`
- Summary: `outputs/renders/model_1399_refined_clean_reverse_velocity_matched_s25_side_iso.summary.json`

Measured metrics over 6 seconds:

- Mean body-frame Y velocity: `-0.450 m/s`
- Forward reference magnitude: `0.453 m/s`
- World Y displacement: `-2.221 m`
- Roll/Pitch/Yaw p2p: `18.36 / 4.56 / 10.04 deg`
- Mean foot contact count: `3.38`
- Non-foot contact fraction: `0.0000`
- Mean stance XY speed:
  - BL: `0.308 m/s`
  - BR: `0.164 m/s`
  - ML: `0.370 m/s`
  - MR: `0.381 m/s`
