# model_1399_refined_clean_reverse_lifted_s25_win130_188_v1

Lifted reverse-motion primitive derived from the cleaned canonical forward trajectory.

Source:

- Forward source label: `model_1399_forward_recycled_stride_v1`
- Base checkpoint: `trained_models/forward_strict_best/model_1399.pt`
- Captured cleaned trajectory: `data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz`
- Replay source: realized joint positions, `source=actual`
- Replay window: `1.30s..1.88s`
- Playback: reversed, cycled, `2.5x`

Post-model cleanups:

- Body-frame reverse velocity servo: target `-0.45 m/s`, alpha `1.0`
- Reverse yaw feedback: `kp=0.45`, `kd=0.70`, `max=0.35`
- Root yaw-rate damping: `0.4`
- Reverse recline/start posture: `front_up_deg=-15`, `root_z_offset=0.085`
- Scheduled reverse swing lift:
  - Uses recorded `foot_contacts` from the cleaned trajectory for leg-by-leg switching.
  - Builds a smooth sine pulse over each recorded airborne segment.
  - Uses femur/tibia IK to convert vertical foot lift into action offsets.
  - Lift target: `0.03 m`
  - Rear scale: `1.0`
  - Middle scale: `2.0`
  - Max lift action delta: `0.55`

Artifacts:

- Video: `outputs/renders/model_1399_refined_clean_reverse_lifted_s25_win130_188_side_iso.mp4`
- Contact sheet: `outputs/renders/model_1399_refined_clean_reverse_lifted_s25_win130_188_side_iso_contact_sheet.png`
- Summary: `outputs/renders/model_1399_refined_clean_reverse_lifted_s25_win130_188_side_iso.summary.json`

Measured metrics over 6 seconds:

- Mean body-frame Y velocity: `-0.450 m/s`
- World Y displacement: `-2.375 m`
- Roll/Pitch/Yaw p2p: `14.89 / 4.58 / 7.57 deg`
- Mean foot contact count: `2.18`
- Non-foot contact fraction: `0.0000`
- Scheduled max lift height:
  - BL: `0.166 m`
  - BR: `0.184 m`
  - ML: `0.059 m`
  - MR: `0.040 m`

Notes:

This is still a post-model visualization primitive, not a trained reverse policy. Compared with the previous velocity-matched reverse primitive, this version has explicit scheduled swing lift and cleaner yaw, while maintaining the same reverse speed target.
