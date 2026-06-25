# Model 1399 Training Handoff

This package is a portable handoff for continuing training from the best current
four-walking-leg insectoid mini quad policy.

## Selected Checkpoint

- Checkpoint: `trained_models/forward_strict_best/model_1399.pt`
- Task: `Isaac-InsectoidMiniQuad-Flat-Direct-v0`
- Command convention: forward motion is `+Y`
- Strict eval command: `+Y = 0.30 m/s`
- Strict eval score: `TASK_SCORE=0.9211`
- Eval file: `trained_models/forward_strict_best/eval_model_1399_strict_0p30_800.txt`
- Reference video: `outputs/training_rollouts/forward_strict_best/forward_strict_best_model_1399.mp4`

Use `model_1399.pt` as the baseline to fine-tune gait quality. It outscored
`model_1558.pt`, even though `model_1558.pt` has more visible foot lift.

## External Requirements

This zip includes the robot asset, RL task package, helper scripts, best model,
and reference artifacts. It does not include Isaac Sim or Isaac Lab.

Verified source environment:

- Isaac Sim 5.1.0
- Isaac Lab checkout at `/home/ubuntu/IsaacLab`
- NVIDIA L40S
- RSL-RL installed through Isaac Lab

On another workstation, install the matching Isaac Sim/Isaac Lab stack first,
then install the Isaac Lab RSL-RL extension:

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
TERM=xterm "$ISAACLAB_PATH/isaaclab.sh" --install rsl_rl
```

## Unpack And Install

```bash
unzip insectoid_mini_quad_model_1399_handoff_*.zip
cd insectoid_mini_quad_model_1399_handoff

export ISAACLAB_PATH=/path/to/IsaacLab
TERM=xterm "$ISAACLAB_PATH/isaaclab.sh" -p -m pip install -e ./rl
```

The RL package registers:

- `Isaac-InsectoidMiniQuad-Flat-Direct-v0`
- `Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0`

## Continue Training

From the unpacked repo root:

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
TERM=xterm "$ISAACLAB_PATH/isaaclab.sh" -p rl/scripts/train_rsl_rl.py \
  --task Isaac-InsectoidMiniQuad-Flat-Direct-v0 \
  --device cuda:0 \
  --num_envs 4096 \
  --resume \
  --checkpoint trained_models/forward_strict_best/model_1399.pt \
  --max_iterations 300 \
  --run_name gait_v2_from_model_1399
```

The handoff wrapper accepts a direct checkpoint path. Stock Isaac Lab `train.py`
normally expects `--checkpoint` to be a filename pattern under `logs/rsl_rl`;
use the wrapper above for this package.

If the resumed policy is too deterministic for fine-tuning, use the reset-std
wrapper:

```bash
TERM=xterm "$ISAACLAB_PATH/isaaclab.sh" -p rl/scripts/train_rsl_rl_reset_std.py \
  --task Isaac-InsectoidMiniQuad-Flat-Direct-v0 \
  --device cuda:0 \
  --num_envs 4096 \
  --resume \
  --checkpoint trained_models/forward_strict_best/model_1399.pt \
  --reset_action_std 0.35 \
  --max_iterations 300 \
  --run_name gait_v2_from_model_1399_reset_std
```

## Render A Comparison Video

Preferred comparison camera:

- `eye = 3.6,-3.6,3.9`
- `lookat = 0.0,0.0,0.12`

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority TERM=xterm \
"$ISAACLAB_PATH/isaaclab.sh" -p rl/scripts/render_policy_video.py \
  --headless \
  --device cuda:0 \
  --checkpoint trained_models/forward_strict_best/model_1399.pt \
  --forward-vel 0.30 \
  --video-length 300 \
  --fps 50 \
  --camera-eye 3.6,-3.6,3.9 \
  --camera-lookat 0.0,0.0,0.12 \
  --output outputs/training_rollouts/forward_strict_best/model_1399_rollout_check.mp4
```

On some AWS GUI instances, fully displayless RTX rendering may stall. Passing
the active X display with `--headless`, as shown above, avoids opening the full
Isaac Sim UI while still giving the renderer a valid display context.

## Current Morphology Notes

The robot is a quad locomotion task built from a six-leg body:

- Front `FL` and `FR` chains are fixed swept-forward arms.
- Walking legs are `ML`, `MR`, `BL`, and `BR`.
- Action dimension is `12`.
- Valid walking foot bodies are `BL_FOOT`, `BR_FOOT`, `ML_FOOT`, `MR_FOOT`.
- Tibia/body contacts are undesired contacts.

The current best policy tracks forward speed well, but the gait is not yet a
clean balanced four-legged gait. The next training pass should focus on reward
changes for contact balance, touchdown timing, duty factor, and swing clearance.

## Portability Notes

- The USD path in `rl/insectoid_mini_quad_rl/articulation.py` resolves relative
  to the unpacked repo root.
- The train/play wrappers search `ISAACLAB_PATH` first, then common Isaac Lab
  locations.
- Keep this package as a standalone repo; install `./rl` editable after moving
  it.
