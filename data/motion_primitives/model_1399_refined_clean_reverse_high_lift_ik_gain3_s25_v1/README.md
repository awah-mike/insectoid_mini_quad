# model_1399_refined_clean_reverse_high_lift_ik_gain3_s25_v1

High-lift reverse-motion visualization primitive derived from the cleaned canonical forward trajectory.

Source:

- Forward source label: `model_1399_forward_recycled_stride_v1`
- Base checkpoint: `trained_models/forward_strict_best/model_1399.pt`
- Captured cleaned trajectory: `data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz`
- Replay source: realized joint positions, `source=actual`
- Replay window: `1.30s..1.88s`
- Playback: reversed, cycled, `2.5x`

Post-model cleanups:

- Body-frame reverse velocity servo: target `-0.52 m/s`, alpha `1.0`
- Lift-phase velocity relief: `0.15`, which briefly relaxes the reverse velocity target during peak swing
- Reverse yaw feedback: `kp=0.45`, `kd=0.70`, `max=0.35`
- Root yaw-rate damping: `0.4`
- Reverse recline/start posture: `front_up_deg=-15`, `root_z_offset=0.12`
- Scheduled high-lift reverse swing:
  - Uses recorded `foot_contacts` from the cleaned trajectory for leg-by-leg switching.
  - Dilates the airborne/swing mask by `2` replay samples to keep the lift visible longer.
  - Builds a smooth sine pulse over each cyclic airborne segment.
  - Uses femur/tibia IK to convert vertical foot lift into action offsets.
  - Lift target: `0.04 m`
  - IK action gain: `3.0`
  - Rear scale: `1.0`
  - Middle scale: `2.5`
  - Max lift action delta: `0.95`

Why this version:

- Direct femur/tibia lift pulses and a ballast-like stronger front-up variant were tested.
- The direct/ballast variants produced the largest rear-foot heights, but introduced non-foot contact and larger roll/yaw excursions.
- This selected version gives a clear rear-leg lift increase while preserving zero non-foot contact in the final render.

Artifacts:

- Final video: `outputs/renders/model_1399_refined_clean_reverse_high_lift_ik_gain3_s25_side_iso_close.mp4`
- Contact sheet: `outputs/renders/model_1399_refined_clean_reverse_high_lift_ik_gain3_s25_side_iso_close_contact_sheet.png`
- Summary: `outputs/renders/model_1399_refined_clean_reverse_high_lift_ik_gain3_s25_side_iso_close.summary.json`
- Wider side-isometric render: `outputs/renders/model_1399_refined_clean_reverse_high_lift_ik_gain3_s25_side_iso.mp4`
- Noisier high-lift ballast comparison: `outputs/renders/model_1399_refined_clean_reverse_high_lift_ballast_s25_side_iso.mp4`

Measured metrics over 6 seconds:

- Mean body-frame Y velocity: `-0.453 m/s`
- World displacement: `[-0.494, -2.468, -0.123] m`
- Roll/Pitch/Yaw p2p: `14.79 / 5.58 / 22.70 deg`
- Mean foot contact count: `1.95`
- Non-foot contact fraction: `0.0000`
- Scheduled max lift height:
  - BL: `0.223 m`
  - BR: `0.223 m`
  - ML: `0.099 m`
  - MR: `0.098 m`

Notes:

This is still a post-model visualization primitive, not a trained reverse policy. Compared with the previous lifted reverse primitive, it trades more yaw drift for visibly higher rear and middle leg clearance.
