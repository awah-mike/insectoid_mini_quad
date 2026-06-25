# model_1399_reverse_tape_exact_1x_yaw20_v1

Canonical label: `model_1399_reverse_tape_exact_1x_yaw20_v1`

This backwards primitive comes from the cooperative payload handoff package `hexapod_transport_handoff_best_m30_selected_20260618_142349.zip`. It uses the included `reverse_tape_exact_1x_tape.pt`, which stores a model-1399 forward action recording replayed backward in time.

## Source

- Handoff archive: `/home/ubuntu/insectoid_mini_quad/external/references/hexapod_transport_handoff_best_m30_selected_20260618_142349.zip`
- Extracted package: `/home/ubuntu/insectoid_mini_quad/external/references/hexapod_transport_handoff_best_m30_selected_20260618_142349`
- Source tape: `/home/ubuntu/insectoid_mini_quad/external/references/hexapod_transport_handoff_best_m30_selected_20260618_142349/priors/reverse_tape_exact_reverse_tape_exact_1x_tape.pt`
- Converted local NPZ: `/home/ubuntu/insectoid_mini_quad/data/trajectories/reversed_trajectory/reverse_tape_exact_1x_from_transport_handoff.npz`
- Replay script: `/home/ubuntu/insectoid_mini_quad/rl/scripts/render_reversed_joint_trajectory.py`

## What The Handoff Did

The transport environment did not learn a clean reverse gait from scratch. It used:

- `model_1399` as the forward locomotion prior for one robot.
- `reverse_replay_tape` as an open-loop action-tape prior for the opposite-facing robot.
- Residual policy actions on top of the priors.
- Coupled residual projection so the two robots make mirrored residual corrections.
- A virtual weld that keeps the two robots in a useful opposed formation.
- A `BL_tibia_joint` target bias of `-30 deg` for the reverse transport robot.

The saved `reverse_replay_tape` is exactly the saved forward tape reversed in time. It is not a coxa sign mirror. The transport setup makes this useful because the reverse robot is physically rotated to face the payload; walking backward in its body frame moves it in the desired world direction.

## Single-Robot Replay Result

For our standalone robot, the best direct transfer was the time-reversed action tape without the transport-specific `BL_tibia_joint=-30 deg` bias. Adding that bias reduced backward displacement and increased yaw drift, so it is not part of this primitive.

Replay settings:

- Source: `reverse_replay_tape`
- Source form: normalized action tape, clipped to `[-1, 1]` before replay
- Time window: `0.0s` to `3.6s`
- Looping: enabled
- Start pose: default standing pose
- Playback speed: `1.0`
- Root yaw-rate damping: `0.20`
- `BL_tibia_joint` bias: disabled

## Current Render

- Video: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_reverse_tape_exact_1x_yaw20_side_iso.mp4`
- Contact sheet: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_reverse_tape_exact_1x_yaw20_side_iso_contact_sheet.png`
- Summary: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_reverse_tape_exact_1x_yaw20_side_iso.summary.json`

## Metrics

- World-Y displacement over 6s: `-0.758 m`
- Mean body-Y velocity: `-0.136 m/s`
- Roll peak-to-peak: `5.58 deg`
- Pitch peak-to-peak: `6.32 deg`
- Yaw drift peak-to-peak: `24.83 deg`
- Non-foot contact fraction: `0.0`

## Comparison

- Previous reversed achieved-joint v2 displacement: `-0.313 m`
- Tape primitive displacement: `-0.758 m`
- Previous v2 yaw drift: `29.07 deg`
- Tape primitive yaw drift: `24.83 deg`

## Notes

- This is an action-tape primitive, not a trained reverse policy checkpoint.
- The tape is the best current base for reverse motion and is a better seed for multi-robot payload demos than the earlier achieved-joint reversal.
- The transport policy checkpoints still include learned residuals and virtual-weld formation logic; those are separate from this single-robot tape replay.
