# model_1399_refined_clean_reverse_actual_246_304_v1

Reverse-motion primitive derived from the cleaned canonical forward primitive:

- Forward source label: `model_1399_forward_recycled_stride_v1`
- Forward checkpoint: `trained_models/forward_strict_best/model_1399.pt`
- Captured trajectory: `data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz`
- Replay source: realized joint positions, not policy actions
- Replay window: `2.46s..3.04s`
- Playback: reversed, cycled, `1.0x`
- Posture bias: `front_up_deg=-15`
- Yaw damping: `root_yaw_rate_damping=0.2`

Why this exists:

The action-level reversal of the cleaned forward primitive did not produce a physical reverse gait. It either moved forward, stalled, or introduced body contact. The best refined-source reverse variant came from replaying the realized joint trajectory over the late-cycle window above. This keeps the source tied to the cleaned canonical forward motion while avoiding the invalid reversed-action behavior.

Current render:

- Video: `outputs/renders/model_1399_refined_clean_reverse_actual_246_304_side_iso.mp4`
- Summary: `outputs/renders/model_1399_refined_clean_reverse_actual_246_304_side_iso.summary.json`

Measured over a 6 second render:

- World Y displacement: `-0.313 m`
- Mean body-frame Y velocity: `-0.057 m/s`
- Roll peak-to-peak: `17.40 deg`
- Pitch peak-to-peak: `10.83 deg`
- Yaw drift peak-to-peak: `29.07 deg`
- Non-foot contact fraction: `0.0000`

Notes:

This primitive is physically valid but slower and less attitude-consistent than the cleaned forward primitive. It should be treated as a reverse visualization primitive, not as a trained reverse policy.
