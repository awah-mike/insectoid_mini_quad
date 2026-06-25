# Trained Insectoid Mini Quad Models

This directory contains the best checkpoints, exported policies, and strict
evaluator outputs from the Isaac Lab RSL-RL bringup as of 2026-05-16.

Generated rollout videos are intentionally stored outside this folder under
`../outputs/training_rollouts/` so training artifacts and presentation outputs
do not mix.

Final project selection: **`forward_strict_best/model_1399.pt` is the favorite
forward locomotion model**. Use it as the canonical mini-quad walking baseline
for future comparison, export, or deployment work.

## Forward Strict Best

- Checkpoint: `forward_strict_best/model_1399.pt`
- Source run: `2026-05-14_23-32-36_step_height_test_from_stride_best_200`
- Eval: `forward_strict_best/eval_model_1399_strict_0p30_800.txt`
- Video: `../outputs/training_rollouts/forward_strict_best/forward_strict_best_model_1399.mp4`
- Command: `+Y = 0.30 m/s`
- Strict evaluator score: `TASK_SCORE=0.9211`

This is the highest-scoring forward locomotion policy and the selected favorite
model for this project.

## Forward Visual High-Step Benchmark

- Checkpoint: `forward_visual_high_step/model_1558.pt`
- Source run: `2026-05-15_18-58-07_higher_step_height_from_best_160`
- Eval: `forward_visual_high_step/eval_model_1558_strict_0p30_800.txt`
- Video: `../outputs/training_rollouts/forward_visual_high_step/forward_visual_high_step_model_1558.mp4`
- Command: `+Y = 0.30 m/s`
- Strict evaluator score: `TASK_SCORE=0.8751`

This policy is visually useful because it has clearer foot lift, but it scores
below `model_1399.pt` because tracking, stance duration, and balance regressed.

## Reverse Straighter Best

- Checkpoint: `reverse_straighter_best/model_1779.pt`
- Source run: `2026-05-15_20-53-12_reverse_y_stronger_yaw_cleanup_80`
- Eval: `reverse_straighter_best/eval_model_1779_reverse_y_neg0p30_800.txt`
- Video: `../outputs/training_rollouts/reverse_straighter_best/reverse_straighter_model_1779.mp4`
- Command: `-Y = 0.30 m/s`
- Strict evaluator score: `TASK_SCORE=0.5555`

This is the best reverse-motion checkpoint so far. It is much straighter than
the first reverse policy, but it still has short strides and swing drag. Treat
it as a reverse smoke-test baseline, not a finished gait.

## Loading

Install the RL package from the repo:

```bash
/workspace/isaaclab/isaaclab.sh -p -m pip install -e /workspace/insectoid_mini_quad/rl
```

Then render a checkpoint with the helper scripts under `rl/scripts/`.
