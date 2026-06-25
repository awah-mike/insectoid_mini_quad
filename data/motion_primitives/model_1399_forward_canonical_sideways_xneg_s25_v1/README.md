# model_1399_forward_canonical_sideways_xneg_s25_v1

Sideways body-X translation primitive derived from the cleaned canonical forward trajectory.

Source:

- Forward source label: `model_1399_forward_recycled_stride_v1`
- Base checkpoint: `trained_models/forward_strict_best/model_1399.pt`
- Captured cleaned trajectory: `data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz`
- Replay source: realized joint positions, `source=actual`
- Replay window: `0.72s..1.30s`
- Playback: forward order, cycled, `2.5x`

Post-model cleanups:

- Body-frame X velocity servo: target `-0.35 m/s`, alpha `1.0`
- Body-frame Y velocity servo: target `0.0 m/s`, alpha `1.0`
- Yaw feedback through coxa correction: `kp=0.85`, `kd=1.30`, `max=0.55`
- Root yaw-rate damping: `0.75`
- Start posture: `root_z_offset=0.085`
- Sideways foot-trajectory remap:
  - Converts the canonical forward foot Y excursion into body-X foot excursion.
  - Uses `sideways_remap_scale=1.5` and `sideways_remap_sign=-1.0`.
  - Suppresses the original forward/back foot excursion with `sideways_remap_y_keep_scale=0.0`.
  - Uses all joints for the remap IK with `max_action_delta=0.85`.
- Small scheduled lift:
  - Uses recorded `foot_contacts`.
  - Lift target: `0.02 m`
  - Rear scale: `1.0`
  - Middle scale: `1.5`
  - Lift mask dilation: `1`

Why this version:

- Positive and negative body-X directions were tested.
- The negative-X direction had lower roll and pitch at the same commanded sideways speed.
- A direct root yaw pose servo was tested but rejected because it increased yaw drift and world-path curvature.
- This version keeps the body translating laterally while preserving zero non-foot contact.

Artifacts:

- Video: `outputs/renders/model_1399_forward_canonical_sideways_xneg_s25_front_iso.mp4`
- Contact sheet: `outputs/renders/model_1399_forward_canonical_sideways_xneg_s25_front_iso_contact_sheet.png`
- Summary: `outputs/renders/model_1399_forward_canonical_sideways_xneg_s25_front_iso.summary.json`

Measured metrics over 6 seconds:

- Mean body-frame X velocity: `-0.350 m/s`
- Mean body-frame Y velocity: `0.000 m/s`
- World displacement: `[-1.818, 0.466, -0.036] m`
- Roll/Pitch/Yaw p2p: `5.11 / 3.95 / 28.53 deg`
- Mean foot contact count: `1.79`
- Non-foot contact fraction: `0.0000`
- Scheduled max lift height:
  - BL: `0.061 m`
  - BR: `0.080 m`
  - ML: `0.076 m`
  - MR: `0.051 m`

Notes:

This is a post-model visualization primitive, not a trained lateral locomotion policy. The remaining rough edge is yaw drift; it is lower-cost to accept it here than to use the direct root yaw servo, which made the world path worse in testing.
