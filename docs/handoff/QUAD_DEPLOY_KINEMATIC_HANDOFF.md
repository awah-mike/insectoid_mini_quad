# Quad Kinematic Deployment Handoff

Date: 2026-06-28

This handoff covers the deployment-adapted forward and reverse quad-test
policies derived from the accepted baselines:

- Forward source: `model_1798.pt`
- Reverse source: `model_2615.pt`

The goal was to remove reliance on privileged simulator/root base linear
velocity while preserving the accepted visual gaits. The resulting policies
were adapted with a frozen-teacher supervised distillation pass: the teacher
received privileged simulator velocity, while the student received noisy
deployment-style observations using a stance-foot kinematic velocity estimate.

## Selected Artifacts

Stable handoff folder:

`trained_models/deploy_quad_kinematic/`

Portable handoff archive:

`packages/handoff/quad_kinematic_deploy_20260628.zip`

Selected checkpoints:

- Forward:
  `trained_models/deploy_quad_kinematic/model_1798_forward_deploy_kinematic.pt`
- Backward:
  `trained_models/deploy_quad_kinematic/model_2615_backward_deploy_kinematic.pt`

Verification videos:

- Forward:
  `outputs/training_rollouts/deploy_obs/model_1798_deploy_distill_v3_forward_kinematic_nominal.mp4`
- Backward:
  `outputs/training_rollouts/deploy_obs/model_2615_deploy_distill_v1_backward_kinematic_nominal.mp4`

The original baseline checkpoints remain untouched in their source run folders.

## Policy Contract

These checkpoints use the experimental `quad_test` shape from the deployment
note:

- 60D observation -> MLP `[256, 128, 128]` -> 12D action
- active joint order:
  `BL_coxa, BR_coxa, ML_coxa, MR_coxa, BL_femur, BR_femur, ML_femur, MR_femur, BL_tibia, BR_tibia, ML_tibia, MR_tibia`
- fixed front limbs:
  `FL_coxa = 1.047198`, `FL_femur = 0`, `FL_tibia = 0`,
  `FR_coxa = 1.047198`, `FR_femur = 0`, `FR_tibia = 0`

Observation layout:

- `[0:3]` kinematic base linear velocity estimate, body frame
- `[3:6]` IMU angular velocity
- `[6:9]` projected gravity
- `[9:12]` velocity command
- `[12:24]` active joint positions minus default active positions
- `[24:36]` active joint velocities
- `[36:48]` previous actions
- `[48:52]` contact or stance flags `[BL, BR, ML, MR]`
- `[52:56]` normalized stance timers
- `[56:60]` normalized swing timers

Action processing must be:

```python
raw_action = policy(obs)
clipped_action = np.clip(raw_action, -1.0, 1.0)
target = default_active_position + 0.5 * clipped_action
```

The deployment note said the current `quad_test` adapter does not clip actions.
That must be changed before hardware deployment. Do not pre-apply motor signs;
publish joint-space targets in the same joint convention used by the current
policy adapter.

## Required Adapter Change: Observation `[0:3]`

The deployment note said the current `quad_test` adapter currently sets
observation `[0:3]` to zero. These deploy checkpoints expect `[0:3]` to be a
kinematic estimate, not zeros and not true world/root velocity.

At each policy tick:

1. Compute body-frame FK foot positions for the four walking feet in order
   `[BL, BR, ML, MR]`.
2. Use the same stance/contact flags that are placed in observation `[48:52]`.
3. Estimate stance foot velocity in body frame with finite differences.
4. Correct for body angular velocity from the IMU:
   `per_foot_base_vel_b = -(foot_vel_b + omega_b x foot_pos_b)`.
5. Average only stance/contact feet.
6. Smooth:
   `estimate = 0.82 * previous + 0.18 * measured`.
7. If no feet are in stance/contact, decay:
   `estimate = 0.98 * previous`.
8. Clamp each component to `[-1.0, 1.0]` and set z velocity to zero.

Reference code:

`docs/handoff/quad_test_kinematic_velocity_estimator_reference.py`

ROS-side implementation template:

`docs/handoff/gram_ros_mini_quad_kinematic_policy/`

The template includes a dry-run-first installer:

```bash
python3 docs/handoff/gram_ros_mini_quad_kinematic_policy/install_into_gram_ros_mini.py \
  /path/to/gram-ros_mini/src/locomotion/policy
```

This estimator uses only joint states/FK, IMU angular velocity, and existing
contact/stance schedule signals. It does not use privileged simulator velocity.

## Training Method

Script:

`rl/scripts/distill_deploy_observation_policy.py`

Forward command used:

```bash
PYTHONPATH=/home/ubuntu/insectoid_mini_quad/rl \
/home/ubuntu/IsaacLab/isaaclab.sh -p rl/scripts/distill_deploy_observation_policy.py \
  --task Isaac-InsectoidMiniQuad-GaitRefineDeploy-Direct-v0 \
  --checkpoint logs/rsl_rl/insectoid_mini_quad_gait_refine/2026-06-25_22-56-37_gait_refine_long_stride_v1_from_model_1399/model_1798.pt \
  --output logs/rsl_rl/insectoid_mini_quad_deploy_distill/forward_1798_kinematic_noise_v3_clipped_strong/model_1798_deploy_distill_kinematic_noise_v3_clipped_strong.pt \
  --num-envs 96 --steps 1000 --epochs 10 --minibatch-size 8192 \
  --verify-steps 420 --student-noise-samples 2 --command-jitter 0.025 \
  --forward-vel 0.30 --target-action-clip 1.0 --action-limit-loss-weight 0.50 \
  --device cuda:0
```

Backward command used:

```bash
PYTHONPATH=/home/ubuntu/insectoid_mini_quad/rl \
/home/ubuntu/IsaacLab/isaaclab.sh -p rl/scripts/distill_deploy_observation_policy.py \
  --task Isaac-InsectoidMiniQuad-BackwardDeploy-Direct-v0 \
  --checkpoint logs/rsl_rl/insectoid_mini_quad_backward_bl_tibia_link_balance_v3/2026-06-27_01-32-41_bl_tibia_link_balance_v3_floor_from_2590/model_2615.pt \
  --output logs/rsl_rl/insectoid_mini_quad_deploy_distill/backward_2615_kinematic_noise_v1_clipped/model_2615_deploy_distill_kinematic_noise_v1_clipped.pt \
  --num-envs 96 --steps 1000 --epochs 10 --minibatch-size 8192 \
  --verify-steps 420 --student-noise-samples 2 --command-jitter 0.02 \
  --forward-vel -0.18 --target-action-clip 1.0 --action-limit-loss-weight 0.50 \
  --device cuda:0
```

## Verification Metrics

Forward deploy checkpoint:

- clean mean velocity: `0.3346 m/s`
- noisy mean velocity: `0.3341 m/s`
- clean yaw abs: `0.1039 rad/s`
- noisy yaw abs: `0.1048 rad/s`
- noisy action p95 abs: `1.1461`

Reverse deploy checkpoint:

- clean mean velocity: `-0.1020 m/s`
- noisy mean velocity: `-0.1068 m/s`
- clean yaw abs: `0.1108 rad/s`
- noisy yaw abs: `0.1226 rad/s`
- noisy action p95 abs: `0.9705`

Rendering note: on the AWS instance, headless `raw_env.render()` failed unless
the existing DCV display was bound. The successful videos were rendered with
`DISPLAY=:1`.

## Hardware Bring-Up Checklist

1. Add the two model files to:
   `gram-ros_mini/src/locomotion/policy/models/`.
2. Copy or translate the config examples from
   `docs/handoff/gram_ros_mini_quad_kinematic_policy/`.
3. Keep `policy_variant: quad_test` or add a new variant derived from it.
4. Replace obs `[0:3] = zeros` with the kinematic estimator above.
5. Clip raw 12D policy action to `[-1, 1]` before applying the `0.5` action
   scale.
6. Keep the existing ROS command mapping:
   `policy_cmd[0] = -linear.y`,
   `policy_cmd[1] = linear.x`,
   `policy_cmd[2] = angular.z`.
7. Keep publishing joint-space commands; do not apply motor signs or offsets in
   the policy node.
8. Start hardware testing with the robot suspended or lightly supported.
9. Log `/policy_debug` slices for obs `[0:3]`, angular velocity, projected
   gravity, joint position deltas, previous actions, raw actions, and targets.
10. On the floor, begin with low command magnitudes and verify no target jumps
    exceed the motor controller rate limit.

## Current Caveats

- The forward policy still occasionally emits raw actions above `1.0`; action
  clipping in the adapter is mandatory.
- The reverse deploy policy is stable but slower than the visual baseline. It
  should be treated as a first deploy-safe reverse primitive, not the final
  reverse gait.
- The kinematic velocity estimator depends on reasonable stance/contact flags.
  If those flags are only a clock schedule, make sure they match the contact
  timing used in the deployed gait.
