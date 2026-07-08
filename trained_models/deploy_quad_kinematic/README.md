# Quad Kinematic Deploy Models

This folder contains the selected deployment-adapted quad-test checkpoints for
the real robot.

These checkpoints keep the existing experimental quad-test contract:

- 60D observation
- 12D action
- active joint order:
  `BL_coxa, BR_coxa, ML_coxa, MR_coxa, BL_femur, BR_femur, ML_femur, MR_femur, BL_tibia, BR_tibia, ML_tibia, MR_tibia`
- fixed front limbs:
  `FL_coxa = 1.047198`, `FL_femur = 0`, `FL_tibia = 0`,
  `FR_coxa = 1.047198`, `FR_femur = 0`, `FR_tibia = 0`

## Selected Checkpoints

- Forward:
  `model_1798_forward_deploy_kinematic.pt`
  - source: accepted forward `model_1798.pt`
  - command used for adaptation: `+0.30 m/s` body-forward policy command
  - trained with kinematic/noisy deploy observations

- Backward:
  `model_2615_backward_deploy_kinematic.pt`
  - source: accepted reverse baseline `model_2615.pt`
  - command used for adaptation: `-0.18 m/s` body-forward policy command
  - trained with kinematic/noisy deploy observations

Each selected checkpoint has:

- a `*.summary.json` file with distillation and rollout metrics
- a `*.validation.json` file from an offline 60D-to-12D actor load test

## Required Adapter Changes

Do not deploy these by only changing `model_path`.

The current `quad_test` adapter described in the deployment note fills
observation `[0:3]` with zeros. These policies expect `[0:3]` to be a
deployment-safe kinematic base-velocity estimate computed from IMU angular
velocity, joint states, foot forward kinematics, and stance/contact flags.

The adapter should also clip raw policy actions before scaling:

```python
raw_action = np.clip(raw_action, -1.0, 1.0)
target = default_active_position + 0.5 * raw_action
```

This is especially important for the forward checkpoint. Its verification
action p95 is close to the clip range but occasional raw actions exceed 1.0.

See `docs/handoff/QUAD_DEPLOY_KINEMATIC_HANDOFF.md` and
`docs/handoff/quad_test_kinematic_velocity_estimator_reference.py` for the
exact observation contract and estimator reference.

The closest ROS-side drop-in template is:

`docs/handoff/gram_ros_mini_quad_kinematic_policy/`
