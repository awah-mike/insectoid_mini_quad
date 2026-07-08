# Real Robot Codex Integration Summary

This package is intended for a Codex session running on the robot computer.
The Isaac workstation used to produce this zip does not contain the live
`gram-ros_mini` checkout. On the robot, integrate these files into:

```text
gram-ros_mini/src/locomotion/policy
```

## What This Package Contains

The package contains two real-robot deploy checkpoints:

- Forward policy:
  `trained_models/deploy_quad_kinematic/model_1798_forward_deploy_kinematic.pt`
- Backward policy:
  `trained_models/deploy_quad_kinematic/model_2615_backward_deploy_kinematic.pt`

Both were adapted from the selected Isaac Lab checkpoints:

- Forward source: `model_1798.pt`
- Backward source: `model_2615.pt`

The adaptation target is the robot's experimental `quad_test` style policy
interface, not the default legacy policy.

## Critical Contract

Do not install these checkpoints as default legacy 83D/18D policies.
They are 60D/12D quad policies and need a matching adapter.

Expected model shape:

```text
60D observation -> MLP [256, 128, 128] with ELU -> 12D action
```

Expected runtime policy variant:

```text
policy_variant: quad_test_kinematic
```

## 60D Observation Layout

All values are `float32`.

```text
[0:3]    Base linear velocity estimate in body frame
[3:6]    Base angular velocity from IMU
[6:9]    Projected gravity from orientation estimate
[9:12]   Velocity command in policy frame
[12:24]  Active joint positions - default active joint positions
[24:36]  Active joint velocities
[36:48]  Previous clipped 12D policy actions
[48:52]  Contact flags in foot order [BL, BR, ML, MR]
[52:56]  Normalized stance timers
[56:60]  Normalized swing timers
```

The important difference from the older quad test setup is observation
`[0:3]`. It must not be true simulator/root base velocity. On the real robot,
there is no direct base linear velocity measurement. Use the included
stance-foot kinematic estimator:

```text
docs/handoff/gram_ros_mini_quad_kinematic_policy/quad_test_kinematic_runtime.py
```

Estimator behavior:

- Uses body-frame foot positions derived from joint kinematics.
- Uses stance/contact flags for `[BL, BR, ML, MR]`.
- Estimates base velocity from stance foot motion and IMU angular velocity.
- Smooths the estimate with `smoothing = 0.82`.
- Clips each component to `[-1.0, 1.0]`.
- Forces vertical velocity to zero.

If reliable contact sensing is not available, provide conservative contact
flags from gait phase timing. The policy was trained with this estimator and
observation noise, so do not replace `[0:3]` with raw zeros unless doing a
deliberate ablation.

## Command Frame

Use the existing robot command mapping:

```text
policy_cmd[0] = -cmd_vel.linear.y
policy_cmd[1] =  cmd_vel.linear.x
policy_cmd[2] =  cmd_vel.angular.z
```

Forward motion should appear as positive `cmd_vel.linear.x`, which maps to
positive policy command Y.

## Active Joint Order

The 12 active joints must be ordered exactly as:

```text
BL_coxa, BR_coxa, ML_coxa, MR_coxa,
BL_femur, BR_femur, ML_femur, MR_femur,
BL_tibia, BR_tibia, ML_tibia, MR_tibia
```

The front limbs are fixed:

```text
FL_coxa  = 1.047198
FL_femur = 0.0
FL_tibia = 0.0
FR_coxa  = 1.047198
FR_femur = 0.0
FR_tibia = 0.0
```

Publish the final command in the robot's existing 18-joint publish order:

```text
BL_coxa, BL_femur, BL_tibia,
BR_coxa, BR_femur, BR_tibia,
FL_coxa, FL_femur, FL_tibia,
FR_coxa, FR_femur, FR_tibia,
ML_coxa, ML_femur, ML_tibia,
MR_coxa, MR_femur, MR_tibia
```

## Action Processing

The checkpoints output a raw 12D action.

Required processing:

```text
clipped_action = clip(raw_action, -1.0, +1.0)
active_target = default_active_position + 0.5 * clipped_action
```

Then insert the fixed front-limb targets and publish all 18 joint-space
positions. These are robot joint coordinates. Do not pre-apply motor signs,
offsets, CAN IDs, or motor routing. The downstream motor controller already
does that.

The included config files encode this:

```text
docs/handoff/gram_ros_mini_quad_kinematic_policy/policy_config_quad_forward_kinematic.yaml
docs/handoff/gram_ros_mini_quad_kinematic_policy/policy_config_quad_backward_kinematic.yaml
```

## Recommended Robot-Side Integration Steps

1. Unzip this package on the robot computer.
2. Inspect the live repo:

   ```text
   gram-ros_mini/src/locomotion/policy
   ```

3. Run the included installer in dry-run mode first:

   ```bash
   python3 docs/handoff/gram_ros_mini_quad_kinematic_policy/install_into_gram_ros_mini.py \
     /path/to/gram-ros_mini/src/locomotion/policy
   ```

4. If the planned paths are correct, apply the copy:

   ```bash
   python3 docs/handoff/gram_ros_mini_quad_kinematic_policy/install_into_gram_ros_mini.py \
     /path/to/gram-ros_mini/src/locomotion/policy \
     --apply
   ```

5. Add or update the robot policy loader so `policy_variant:
   quad_test_kinematic` uses the 60D observation builder and 12D action
   processor described above.
6. Preserve existing idle behavior, command timeout, target rate limiting, and
   joint-space publishing semantics from the current policy node.
7. Start with a low command and test suspended or unloaded before floor trials.

## Validation Evidence In This Package

The manifest and validation files record the selected checkpoints and basic
shape checks:

```text
trained_models/deploy_quad_kinematic/manifest.json
trained_models/deploy_quad_kinematic/model_1798_forward_deploy_kinematic.validation.json
trained_models/deploy_quad_kinematic/model_2615_backward_deploy_kinematic.validation.json
```

The original Isaac-side handoff document has more detail:

```text
docs/handoff/QUAD_DEPLOY_KINEMATIC_HANDOFF.md
```

