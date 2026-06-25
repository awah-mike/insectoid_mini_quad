# model_1399_forward_recycled_stride_v1

Canonical label: `model_1399_forward_recycled_stride_v1`

This is a forward locomotion motion primitive built from the trained policy checkpoint `model_1399.pt` plus post-policy controls. The trained policy weights were not modified.

## Purpose

Create a reusable forward motion primitive with visibly larger displacement than the clean stabilized gait. This primitive prioritizes forward speed and step displacement over perfectly stable torso attitude.

## Base Inputs

- Base policy: `/home/ubuntu/insectoid_mini_quad/trained_models/forward_strict_best/model_1399.pt`
- Task: `Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0`
- Renderer/controller script: `/home/ubuntu/insectoid_mini_quad/rl/scripts/render_stabilized_policy_video.py`
- Camera used for current review render: side-overhead isometric

## How We Achieved It

1. Started from the best learned policy checkpoint, `model_1399.pt`.
2. Added yaw stabilization as a post-policy coxa correction.
3. Added partial BL rear-leg mirroring from BR to reduce the worst rear-leg asymmetry while preserving more speed than the fully mirrored cleanup.
4. Added rear coxa stride IK to increase rear swing reach and step displacement.
5. Observed that the high-displacement policy had a strong early transient but slowed down later as contact timing became more stance-heavy.
6. Added telemetry logging to confirm the slowdown was real and not only camera perspective.
7. Captured the early fast action window from `0.72s` to `1.30s`.
8. Replayed that post-policy action window as a loop, while keeping live yaw stabilization on top.

The important design point is that the recycled action sequence is captured after policy smoothing, BL partial mirroring, and rear stride IK, but before live yaw feedback. During replay, yaw feedback is recomputed online so the robot still tries to correct heading drift.

## Post-Policy Recipe

- `forward_vel`: `0.60`
- `action_filter_alpha`: `1.0`
- Yaw stabilizer:
  - `yaw_feedback_kp`: `0.45`
  - `yaw_rate_feedback_kd`: `0.70`
  - `max_yaw_feedback`: `0.35`
- BL mirror:
  - enabled
  - joints: distal only
  - blend: `0.55`
  - delay phase: `0.55`
  - cycle steps: `17.77777777777778`
- Stride IK:
  - profile: `swing_reach`
  - rear forward offset: `-0.080 m`
  - middle forward offset: `0.0 m`
  - rear joints: coxa only
  - duration: `0.12 s`
  - max action delta: `0.60`
- Action recycling:
  - enabled
  - capture start: `0.72 s`
  - capture end: `1.30 s`

## Current Reference Artifacts

- Render: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_recycled_early_phase_072_130_side_iso.mp4`
- Contact sheet: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_recycled_early_phase_072_130_side_iso_contact_sheet.png`
- Attitude metrics: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_recycled_early_phase_072_130_side_iso.attitude.json`
- Telemetry: `/home/ubuntu/insectoid_mini_quad/outputs/renders/model_1399_recycled_early_phase_072_130_side_iso.telemetry.csv`

## Metrics From Reference Render

- Mean body-frame forward velocity: `0.453 m/s`
- Non-foot contact fraction: `0.0`
- Roll peak-to-peak: `7.85 deg`
- Pitch peak-to-peak: `7.96 deg`
- Yaw drift peak-to-peak: `19.96 deg`
- BL mean stance XY speed: `0.327 m/s`

Telemetry confirmed this primitive avoids the previous late-run slowdown:

- First half path speed: about `0.438 m/s`
- Second half path speed: about `0.492 m/s`

## Known Tradeoffs

- This is not a clean learned gait. It is a post-policy open-loop primitive with live yaw correction.
- Torso attitude and yaw drift are worse than the slower stabilized gait.
- Foot slip is higher than the clean gait.
- It is suitable as a forward-motion primitive for visual demonstrations, multi-robot choreography, and payload experiments where displacement is more important than strict attitude stability.
- It should not be treated as a robust training checkpoint without folding the behavior into training or a proper phase-based controller.

## Reproduction

Use `render_model_1399_forward_recycled_stride_v1.sh` in this directory to render a new video with the same primitive settings.
