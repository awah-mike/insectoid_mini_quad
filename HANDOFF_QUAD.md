# insectoid_mini_quad Handoff

This folder is a separate side project derived from `/workspace/insectoid_mini`.
It is not part of the `insectoid_mini` git repo.

## Current State As Of 2026-05-16

This repo now contains both the quad robot asset and the current Isaac Lab
DirectRL package.

Important paths:

- Robot asset: `URDF_description/usd/insectoid_mini_quad.usd`
- RL package: `rl/`
- Full experiment progress log: `rl/PROGRESS_LOG.md`
- Best checkpoints/videos/evals: `trained_models/`

Install the RL package from this repo:

```bash
/workspace/isaaclab/isaaclab.sh -p -m pip install -e /workspace/insectoid_mini_quad/rl
```

Registered Gym tasks:

- `Isaac-InsectoidMiniQuad-Flat-Direct-v0`
- `Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0`

Best model checkpoints:

- Main forward baseline: `trained_models/forward_strict_best/model_1399.pt`
  - strict score: `TASK_SCORE=0.9211`
  - source run: `2026-05-14_23-32-36_step_height_test_from_stride_best_200`
- Forward visual high-step benchmark:
  `trained_models/forward_visual_high_step/model_1558.pt`
  - strict score: `TASK_SCORE=0.8751`
  - useful for comparing clearer swing lift
- Best reverse smoke-test checkpoint:
  `trained_models/reverse_straighter_best/model_1779.pt`
  - strict reverse score: `TASK_SCORE=0.5555`
  - reverse command: `Y=-0.30 m/s`

The best forward checkpoint number is **`model_1399.pt`**. If a future session
needs a single policy to start from, use that one.

Recent important engineering decisions:

- The four walking feet are fixed spherical foot bodies:
  `BL_FOOT`, `BR_FOOT`, `ML_FOOT`, `MR_FOOT`.
- Contact metrics and rewards should use the spherical feet, not tibia bodies.
- Front legs are fixed swept-forward arms, not locomotion legs.
- Current policy observation is 60D and includes simulator base linear velocity.
  For real onboard deployment without motion capture, the next major training
  variant should remove base linear velocity from the policy observation while
  still using simulator velocity in the reward/evaluator.
- A direct action-remapping attempt to reverse the forward policy was tested and
  then removed. It did not recover good reverse walking. Continue with trained
  reverse or bidirectional policies instead.

Recommended next step:

1. Create a deployable forward policy variant that removes `root_lin_vel_b`
   from the policy observation.
2. Keep true simulator base velocity in the reward/evaluator.
3. Train with IMU-realistic observations: angular velocity, projected gravity,
   joint positions, joint velocities, previous actions, foot contacts, and
   stance/swing timers.
4. Compare it against `model_1399.pt` and `model_1558.pt`.

## Purpose

`insectoid_mini_quad` is a four-walking-leg variant of the insectoid mini robot.
The full front legs are kept, but both front leg chains are rigid/non-actuated
and swept forward so they behave like suspended arms or front payload structure.
Locomotion should be trained using only the middle and rear four legs.

## Primary Asset

- USD: `URDF_description/usd/insectoid_mini_quad.usd`
- URDF used to generate it: `URDF_description/urdf/URDF_quad.urdf`
- Source six-leg URDF copy: `URDF_description/urdf/URDF_full_source.urdf`
- Meshes: `URDF_description/meshes_local/*.stl`

The USD was generated with Isaac Lab 5.1 tooling:

```bash
cd /workspace/isaaclab
TERM=xterm ./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /workspace/insectoid_mini_quad/URDF_description/urdf/URDF_quad.urdf \
  /workspace/insectoid_mini_quad/URDF_description/usd/insectoid_mini_quad.usd \
  --joint-stiffness 0 \
  --joint-damping 0
```

Important conversion choices:

- `merge_fixed_joints: false`, so the rigid front-leg links remain separate
  USD rigid bodies instead of being merged into `base_link`.
- Joint drives are neutral in the USD: `stiffness=0`, `damping=0`.
- Colliders are convex hulls.
- Self-collision is disabled in the imported USD by default.

## Morphology

Body/link count:

- 23 rigid bodies total.
- `base_link`
- Six complete leg link sets:
  - `FL_coxa_1`, `FL_femur_1`, `FL_tibia_1`
  - `FR_coxa_1`, `FR_femur_1`, `FR_tibia_1`
  - `ML_coxa_1`, `ML_femur_1`, `ML_tibia_1`
  - `MR_coxa_1`, `MR_femur_1`, `MR_tibia_1`
  - `BL_coxa_1`, `BL_femur_1`, `BL_tibia_1`
  - `BR_coxa_1`, `BR_femur_1`, `BR_tibia_1`
- Four fixed spherical walking foot links:
  - `BL_FOOT`, `BR_FOOT`, `ML_FOOT`, `MR_FOOT`

Fixed joints:

- `FL_coxa_joint`
- `FL_femur_joint`
- `FL_tibia_joint`
- `FR_coxa_joint`
- `FR_femur_joint`
- `FR_tibia_joint`
- `BL_foot_fixed_joint`
- `BR_foot_fixed_joint`
- `ML_foot_fixed_joint`
- `MR_foot_fixed_joint`

The front coxa joints are baked to a 60 degree sweep:

- `FR_coxa_joint`: `+60 deg` local coxa rotation.
- `FL_coxa_joint`: `+60 deg` local coxa rotation.

The front-left coxa was corrected by +90 deg from the earlier generated asset,
then both front coxas were advanced by another +15 deg after visual validation.

Front femur and tibia joints are fixed at their URDF zero angles. These front
legs are therefore rigid structures, not controllable walking legs.

Walking revolute joints:

- `ML_coxa_joint`, `ML_femur_joint`, `ML_tibia_joint`
- `MR_coxa_joint`, `MR_femur_joint`, `MR_tibia_joint`
- `BL_coxa_joint`, `BL_femur_joint`, `BL_tibia_joint`
- `BR_coxa_joint`, `BR_femur_joint`, `BR_tibia_joint`

Action dimension for a DirectRL locomotion task should be `12` if only the
walking joints are actuated.

## Joint Limits And Actuator Model

The 12 walking revolute joints keep the same URDF joint limits as the current
six-leg `insectoid_mini` asset:

- coxa: `[-1.047198, +1.047198] rad`, effort `24.0 Nm`, URDF velocity `5.445427 rad/s`
- femur: `[-1.221730, 0.0] rad`, effort `24.0 Nm`, URDF velocity `5.445427 rad/s`
- tibia: `[0.0, +2.617994] rad`, effort `24.0 Nm`, URDF velocity `5.445427 rad/s`

The fixed front-leg joints intentionally have no torque or velocity limits
because they are fixed, non-actuated structure.

For Isaac Lab training, copy the same runtime actuator model currently used by
`/workspace/insectoid_mini/insectoid_mini_rl/articulation.py` onto only the 12
walking joints:

- peak simulator torque cap: `24.0 Nm`
- rated/continuous diagnostic threshold: `8.0 Nm`
- runtime simulator velocity cap: `3.0 rad/s`
- rated speed reference: `40 rpm = 4.189 rad/s`
- no-load speed reference: `52 rpm = 5.445 rad/s`
- gear ratio: `36:1`
- reflected rotor armature: about `0.02357 kg*m^2`
- stiffness: `40.0`
- damping: `1.0`

The `3.0 rad/s` cap is a runtime Isaac Lab actuator setting, not a URDF velocity
limit. The URDF keeps the motor no-load speed limit, matching the current
six-leg URDF.

## Masses And Payload Tuning

Current masses are inherited from the hexapod:

- coxa: `0.2 kg`
- femur: `0.4 kg`
- tibia: `0.4 kg`
- base: about `1.74665 kg`

The fixed front legs are intended to represent arms or front payload structure.
To simulate carried payload, change the masses and inertias of front links in
`URDF_description/urdf/URDF_quad.urdf`, then regenerate the USD. Likely payload
links to tune first:

- `FL_coxa_1`, `FR_coxa_1`
- optionally `FL_femur_1`, `FR_femur_1`
- optionally `FL_tibia_1`, `FR_tibia_1`

Because fixed joints are not merged during conversion, those front link masses
remain separately inspectable/editable in the USD.

## Suggested Isaac Lab Setup

For a new Dr. Eureka/Isaac Lab project:

- Reference the USD at:
  `/workspace/insectoid_mini_quad/URDF_description/usd/insectoid_mini_quad.usd`
- Use identity root orientation if starting from the current insectoid mini
  standing convention.
- Actuate only the 12 middle/rear walking joints listed above.
- Do not include front-leg fixed joints in the action set.
- Foot/contact bodies should be the four fixed spherical foot links:
  - `BL_FOOT`, `BR_FOOT`, `ML_FOOT`, `MR_FOOT`
- Treat tibia links as shanks/non-foot bodies. If a tibia contacts the ground,
  that is an undesired contact rather than a valid foot plant.

The old six-leg DirectRL task uses +Y as the natural forward axis. Preserve
that convention unless the new task explicitly changes it.

For default joint targets on the four walking legs, use the visually approved
quad stance:

- middle coxa (`ML_coxa_joint`, `MR_coxa_joint`): `+30 deg` (`0.523599 rad`)
- rear coxa (`BL_coxa_joint`, `BR_coxa_joint`): `-20 deg` (`-0.349066 rad`)
- femur: `-25 deg` (`-0.436332 rad`)
- tibia: `+110 deg` (`1.919862 rad`)

## Regeneration

The active URDF is produced by:

```bash
cd /workspace/insectoid_mini_quad
TERM=xterm /workspace/isaaclab/isaaclab.sh -p scripts/make_quad_urdf.py
```

Then regenerate the USD with the conversion command above.

## Verification

Verification script:

```bash
cd /workspace/insectoid_mini_quad
TERM=xterm /workspace/isaaclab/isaaclab.sh -p scripts/verify_quad_usd.py --headless
```

Expected summary:

```text
BODIES 23 [...]
REVOLUTE_JOINTS 12 [...]
FIXED_JOINTS 10 [...]
QUAD_USD_OK
```

## Provenance

`URDF_description/urdf/URDF_full_source.urdf` is a copy of the six-leg normalized
URDF used as the input source for `scripts/make_quad_urdf.py`. The active asset
for training is `URDF_quad.urdf` and the generated USD, not the full source copy.

---

# Dr. Eureka Bringup Guide For A New Session

Historical note: this section was written during bringup and includes some old
reward-shaping ideas. The checked-in code under `rl/` and the "Current State"
section above are authoritative for the current DirectRL setup.

This section is written for a fresh LLM/Codex session on a new Brev/Isaac box.
The goal is to make the new `insectoid_mini_quad` robot trainable with the same
Isaac Lab + Dr. Eureka pattern used for `/workspace/insectoid_mini`.

## Repos And Installs

Expected local paths:

- Isaac Lab launcher: `/workspace/isaaclab/isaaclab.sh`
- Dr. Eureka repo: `/workspace/GRAM_DrEureka`
- Quad side project: `/workspace/insectoid_mini_quad`

If the Dr. Eureka repo is missing, clone and install:

```bash
cd /workspace
git clone https://github.com/GRAM-Corporation/GRAM_DrEureka.git
cd /workspace/GRAM_DrEureka
/workspace/isaaclab/isaaclab.sh -p -m pip install -e source/isaaclab_eureka
```

The quad task is checked into this repo as its own Isaac Lab Python package:

```text
/workspace/insectoid_mini_quad/rl/
├── pyproject.toml
└── insectoid_mini_quad_rl/
    ├── __init__.py
    ├── articulation.py
    ├── direct_env.py
    └── agents/
        ├── __init__.py
        └── rsl_rl_ppo_cfg.py
```

Install editably:

```bash
/workspace/isaaclab/isaaclab.sh -p -m pip install -e /workspace/insectoid_mini_quad/rl
```

Then verify Gym registration:

```bash
/workspace/isaaclab/isaaclab.sh -p -c "import insectoid_mini_quad_rl, gymnasium as gym; print([k for k in gym.envs.registration.registry.keys() if 'InsectoidMiniQuad' in k])"
```

Recommended Gym IDs:

- `Isaac-InsectoidMiniQuad-Flat-Direct-v0`
- `Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0`

## Articulation Setup

Create `insectoid_mini_quad_rl/articulation.py` by adapting
`/workspace/insectoid_mini/insectoid_mini_rl/articulation.py`.

Use this USD:

```python
USD_PATH = "/workspace/insectoid_mini_quad/URDF_description/usd/insectoid_mini_quad.usd"
```

The actuator model should be exactly the same as the six-leg robot, but applied
only to the 12 active middle/rear walking joints:

```python
AK45_36_PEAK_TORQUE_NM = 24.0
AK45_36_RATED_TORQUE_NM = 8.0
AK45_36_CONTINUOUS_TORQUE_NM = AK45_36_RATED_TORQUE_NM
AK45_36_RATED_SPEED_RAD_PER_S = 40.0 * 2.0 * math.pi / 60.0
AK45_36_NO_LOAD_SPEED_RAD_PER_S = 52.0 * 2.0 * math.pi / 60.0
AK45_36_SIM_VELOCITY_LIMIT_RAD_PER_S = 3.0
AK45_36_GEAR_RATIO = 36.0
AK45_36_ROTOR_INERTIA_KG_M2 = 181.90e-7
AK45_36_REFLECTED_ROTOR_INERTIA_KG_M2 = AK45_36_ROTOR_INERTIA_KG_M2 * AK45_36_GEAR_RATIO**2
```

Actuator joint regex:

```python
joint_names_expr=[
    "ML_.*_joint", "MR_.*_joint",
    "BL_.*_joint", "BR_.*_joint",
]
```

Do not include any `FL_*_joint` or `FR_*_joint` in the actuator group; those are
fixed front-arm joints in the USD.

Suggested initial state:

```python
INSECTOID_MINI_QUAD_STANDING_POSE = {
    "ML_coxa_joint": math.radians(30.0),
    "MR_coxa_joint": math.radians(30.0),
    "BL_coxa_joint": math.radians(-20.0),
    "BR_coxa_joint": math.radians(-20.0),
    "ML_femur_joint": math.radians(-25.0),
    "MR_femur_joint": math.radians(-25.0),
    "BL_femur_joint": math.radians(-25.0),
    "BR_femur_joint": math.radians(-25.0),
    "ML_tibia_joint": math.radians(110.0),
    "MR_tibia_joint": math.radians(110.0),
    "BL_tibia_joint": math.radians(110.0),
    "BR_tibia_joint": math.radians(110.0),
}
```

Root pose:

```python
pos=(0.0, 0.0, 0.15)
rot=(1.0, 0.0, 0.0, 0.0)
```

Use `activate_contact_sensors=True`. Start with self-collisions disabled unless
there is a specific need to terminate on front-arm/body strikes; this quad USD
was generated with `self_collision: false`.

## DirectRL Environment Setup

Start from `/workspace/insectoid_mini/insectoid_mini_rl/direct_env.py`, but
change the morphology assumptions from 6 feet / 18 actions to 4 feet / 12
actions.

Recommended config:

```python
episode_length_s = 20.0
decimation = 4
action_scale = 0.35
action_target_filter_alpha = 1.0
action_space = 12
observation_space = 60  # see observation list below
state_space = 0
```

Sim:

```python
dt = 0.005
static_friction = 1.0
dynamic_friction = 1.0
restitution = 0.0
scene.num_envs = 1024
env_spacing = 1.5
```

Command convention should match the six-leg project:

```python
# body-frame [lin_vel_x, lin_vel_y, yaw_rate_z]
# +X lateral, +Y natural forward
lin_vel_x_range = (-0.25, 0.25)
lin_vel_y_range = (0.2, 0.3)
ang_vel_z_range = (-0.8, 0.8)
rel_standing_envs = 0.1
```

For the first quad runs, consider keeping `lin_vel_y_range = (0.15, 0.25)` if
it falls too often. Once it stands and steps, return to `(0.2, 0.3)`.

Contact bodies for rewards and success metrics should be only the four spherical
walking feet:

```python
FOOT_BODY_NAMES = ("BL_FOOT", "BR_FOOT", "ML_FOOT", "MR_FOOT")
FOOT_BODY_PATTERN = "BL_FOOT|BR_FOOT|ML_FOOT|MR_FOOT"
NON_FOOT_BODY_PATTERN = "base_link|.*_coxa_1|.*_femur_1|.*_tibia_1"
```

Important: the front tibias exist but are fixed suspended arms. Do not count
`FL_tibia_1` or `FR_tibia_1` as feet for walking contact rewards. Treat contact
on any tibia or front arm as undesired unless intentionally studying
payload-arm grounding.

Recommended observation vector:

```text
root_lin_vel_b:        3
root_ang_vel_b:        3
projected_gravity_b:   3
commands:              3
joint_pos_rel:        12
joint_vel:            12
actions:              12
foot_contacts:         4
stance_time_clipped:   4
swing_time_clipped:    4
total:                60
```

This keeps the simple ANYmal-C proprioceptive core and adds contact-aware
timing inputs for the four spherical feet. The contact and clipped stance/swing
timers help the policy distinguish planted support from swing and reduce rapid
tap contacts.

Termination conditions from the six-leg task are good starting points:

```python
base_contact = base contact force > 1.0 N
base_too_low = root_pos_w[:, 2] < 0.08
base_tilted = norm(projected_gravity_b[:, :2]) > 0.9
invalid_state = nonfinite root/joint state
time_out = episode_length_buf >= max_episode_length - 1
```

Initial reset noise from six-leg task:

- joint position noise: uniform `[-0.08, 0.08] rad`, clamped to soft limits
- joint velocity noise: uniform `[-0.05, 0.05] rad/s`
- root XY position noise: uniform `[-0.1, 0.1] m`
- random initial yaw
- root linear/angular velocity noise: uniform `[-0.05, 0.05]`

## PPO Setup

Reuse the current RSL-RL PPO config from
`/workspace/insectoid_mini/insectoid_mini_rl/agents/rsl_rl_ppo_cfg.py`.

Recommended file:
`insectoid_mini_quad_rl/agents/rsl_rl_ppo_cfg.py`

```python
seed = 42
num_steps_per_env = 24
max_iterations = 1000
save_interval = 50
experiment_name = "insectoid_mini_quad_flat"
obs_groups = {"policy": ["policy"], "critic": ["policy"]}
clip_actions = 1.0

policy.init_noise_std = 1.0
policy.actor_hidden_dims = [256, 128, 128]
policy.critic_hidden_dims = [256, 128, 128]
policy.activation = "elu"

algorithm.value_loss_coef = 1.0
algorithm.use_clipped_value_loss = True
algorithm.clip_param = 0.2
algorithm.entropy_coef = 0.005
algorithm.num_learning_epochs = 5
algorithm.num_mini_batches = 4
algorithm.learning_rate = 1.0e-3
algorithm.schedule = "adaptive"
algorithm.gamma = 0.99
algorithm.lam = 0.95
algorithm.desired_kl = 0.01
algorithm.max_grad_norm = 1.0
```

Smoke-test command pattern:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p scripts/train_rsl_rl.py \
  --task Isaac-InsectoidMiniQuad-Flat-Direct-v0 \
  --num_envs 1024 \
  --headless \
  --max_iterations 50
```

## Baseline Reward Structure

The six-leg direct env reward is a good baseline. For the quad, keep the same
categories but adapt all foot arrays from 6 to 4.

Recommended component names to log:

```text
track_lin_vel_xy_exp
track_ang_vel_z_exp
lin_vel_z_l2
ang_vel_xy_l2
dof_torques_l2
dof_torque_over_continuous_l2
dof_acc_l2
coxa_joint_vel_l2
action_rate_l2
action_saturation_l2
feet_air_time
foot_slip_l2
trot_contact
trot_swing_forward
trot_swing_drag
trot_swing_balance
stance_foot_velocity
coxa_stride
coxa_stride_underuse
coxa_stride_balance
stance_anchor_slip
step_count_balance
touchdown_rate
short_stance
short_swing
touchdown_stride
touchdown_short_stride
stride_length_underuse
stride_length_balance
left_right_stride_balance
undesired_contacts
flat_orientation_l2
survival
```

Initial scales copied from the current six-leg Iteration 10 setup:

```python
lin_vel_reward_scale = 1.5
lin_vel_tracking_sigma = 0.10
yaw_rate_reward_scale = 1.0
z_vel_reward_scale = -2.0
ang_vel_reward_scale = -0.08
joint_torque_reward_scale = -2.0e-5
joint_torque_over_continuous_reward_scale = -1.0e-4
joint_accel_reward_scale = -2.5e-7
coxa_joint_vel_reward_scale = 0.0
action_rate_reward_scale = -0.01
action_saturation_reward_scale = -0.02
feet_air_time_reward_scale = 0.12
foot_slip_reward_scale = -0.08
gait_cycle_s = 1.2
trot_contact_reward_scale = 0.18
trot_swing_forward_reward_scale = 0.16
trot_swing_drag_reward_scale = -0.12
trot_swing_balance_reward_scale = -0.06
coxa_stride_target_rad = 0.40
coxa_stride_reward_scale = 0.42
coxa_stride_underuse_reward_scale = -0.32
coxa_stride_balance_reward_scale = -0.14
stance_anchor_slip_tolerance_m = 0.015
stance_anchor_grace_s = 0.06
stance_anchor_ramp_s = 0.10
min_stance_s = 0.25
min_swing_s = 0.42
stride_length_target_m = 0.13
stride_length_balance_scale_m = 0.08
stride_length_ema_alpha = 0.2
stance_anchor_slip_reward_scale = -0.90
step_count_balance_reward_scale = -0.18
touchdown_rate_reward_scale = 0.0
short_stance_reward_scale = -1.0
short_swing_reward_scale = -4.0
touchdown_stride_reward_scale = 12.0
touchdown_short_stride_reward_scale = -12.0
stance_foot_velocity_scale_mps = 0.15
stance_foot_velocity_reward_scale = 0.0
stride_length_underuse_reward_scale = 0.0
stride_length_balance_reward_scale = -0.35
left_right_stride_balance_reward_scale = -0.45
undesired_contact_reward_scale = -1.0
flat_orientation_reward_scale = -3.5
survival_reward_scale = 0.2
```

For a four-leg robot, replace "tripod" with a diagonal trot gait prior:

- Foot order recommendation: `[BL, BR, ML, MR]`
- Diagonal pair A: `[BL, MR]`
- Diagonal pair B: `[BR, ML]`
- Alternating gait phase: pair A stance while pair B swings, then swap.

This is a soft gait prior only. Do not prescribe joint angles or overwrite
actions. The policy still decides the movement.

The most important physical reward concept from the six-leg tuning is stance
anchoring:

- When a walking tibia touches down, store its world XY location.
- While it remains in stance, penalize XY drift from that anchor after a short
  grace/ramp window.
- Use touchdown-to-touchdown distance to measure stride length.
- Penalize unequal step counts and unequal stride lengths across the four legs.

This addresses two common bad gaits:

- feet sliding while "standing" under the body,
- one leg taking many tiny steps while another leg takes fewer large steps.

## Key Reward Metrics To Watch

During PPO or Dr. Eureka training, watch these TensorBoard scalars:

```text
Episode_Reward/track_lin_vel_xy_exp
Episode_Reward/track_ang_vel_z_exp
Episode_Reward/flat_orientation_l2
Episode_Reward/dof_torque_over_continuous_l2
Episode_Reward/action_rate_l2
Episode_Reward/dof_acc_l2
Episode_Reward/stance_anchor_slip
Episode_Reward/step_count_balance
Episode_Reward/touchdown_stride
Episode_Reward/touchdown_short_stride
Episode_Reward/stride_length_balance
Episode_Reward/left_right_stride_balance
Episode_Termination/base_contact_or_fall
Episode_Termination/time_out
Eureka/success_metric
Eureka/eureka_total_rewards
Eureka/oracle_total_rewards
```

Healthy signs:

- `time_out` dominates `base_contact_or_fall`.
- velocity tracking increases without `flat_orientation_l2` becoming strongly
  negative.
- touchdown stride rises while touchdown short-stride penalty does not dominate.
- torque-over-continuous remains a diagnostic penalty, not the largest term.
- action-rate/joint-acc penalties do not dominate the reward.

Bad signs:

- `Eureka/eureka_total_rewards` rises while `Eureka/success_metric` is flat or
  declining. That means reward hacking.
- support/flatness/survival dominates and the robot becomes a static crouch.
- joint acceleration, action rate, or torque penalties are larger than the
  tracking terms by an order of magnitude.
- front fixed arms contact the ground and become an unintended support strategy.

## Dr. Eureka Integration

In `/workspace/GRAM_DrEureka`, edit:

```text
source/isaaclab_eureka/isaaclab_eureka/config/tasks.py
source/isaaclab_eureka/isaaclab_eureka/config/prompt_templates.py
source/isaaclab_eureka/isaaclab_eureka/managers/eureka_task_manager.py
```

### eureka_task_manager.py

Ensure the task manager imports the quad package before `gym.make()`:

```python
try:
    import insectoid_mini_quad_rl  # noqa: F401
except ModuleNotFoundError:
    pass
```

Keep the existing useful environment override:

```python
num_envs_override = os.environ.get("EUREKA_NUM_ENVS")
if num_envs_override and hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
    env_cfg.scene.num_envs = int(num_envs_override)
```

### tasks.py

Add a new `TASKS_CFG` entry:

```python
"Isaac-InsectoidMiniQuad-Flat-Direct-v0": {
    "description": INSECTOID_MINI_QUAD_DESC,
    "success_metric": INSECTOID_MINI_QUAD_METRIC,
    "success_metric_to_win": 1.0,
    "success_metric_tolerance": 0.05,
},
```

Suggested success metric, adapted from the six-leg mini metric:

```python
INSECTOID_MINI_QUAD_METRIC = (
    "(torch.exp(-torch.sum(torch.square("
    "self._commands[env_ids, :2] - self._robot.data.root_lin_vel_b[env_ids, :2]"
    "), dim=1) / 0.20) "
    "* torch.exp(-torch.square(self._commands[env_ids, 2] - self._robot.data.root_ang_vel_b[env_ids, 2]) / 0.25) "
    "* torch.clamp((self._robot.data.root_pos_w[env_ids, 2] - 0.08) / 0.07, 0.0, 1.0) "
    "* torch.exp(-torch.sum(torch.square(self._robot.data.projected_gravity_b[env_ids, :2]), dim=1) / 0.25) "
    "* (self.episode_length_buf[env_ids].float() / self.max_episode_length)"
    ").mean()"
)
```

Suggested task description:

```python
INSECTOID_MINI_QUAD_DESC = (
    "Task: make the insectoid_mini_quad robot track body-frame planar velocity and yaw-rate commands on flat terrain. "
    "This is a 23-body robot with four actuated walking legs, four fixed spherical foot links, and two fixed front legs. "
    "The front legs are full rigid chains swept forward by 60 degrees and should behave like suspended arms/payload structure, "
    "not walking supports. The middle and rear legs provide locomotion. A successful policy walks at the commanded velocity "
    "without base contact, excessive crouch, dragging the front arms, or a static standing solution.\\n\\n"
    "CURRENT ISAAC LAB TASK API:\\n"
    "- Preferred task ID is Isaac-InsectoidMiniQuad-Flat-Direct-v0, an Isaac Lab DirectRLEnv.\\n"
    "- Use self._robot for the articulation, self._commands for velocity commands, self._actions for current actions, "
    "and self._previous_actions for previous actions.\\n"
    "- self._commands: (num_envs, 3), body-frame [lin_vel_x, lin_vel_y, yaw_rate_z]. +X is lateral and +Y is natural forward.\\n"
    "- self._robot.data.root_lin_vel_b, root_ang_vel_b, root_pos_w, projected_gravity_b are available.\\n"
    "- self._robot.data.applied_torque, joint_acc, joint_pos, joint_vel are all (num_envs, 12) for the actuated walking joints.\\n"
    "- self._actions and self._previous_actions are (num_envs, 12).\\n"
    "- self._contact_sensor.data.net_forces_w_history is (num_envs, history_len, num_bodies, 3).\\n"
    "- self._feet_ids indexes only the four spherical walking feet [BL_FOOT, BR_FOOT, ML_FOOT, MR_FOOT].\\n"
    "- self._base_id indexes base_link.\\n\\n"
    "ROBOT SETUP:\\n"
    "- Active walking joints are ML/MR/BL/BR coxa/femur/tibia: 12 actions.\\n"
    "- Fixed front joints are FL/FR coxa/femur/tibia. Front coxas are baked at a 60 degree forward sweep.\\n"
    "- Walking standing pose: ML/MR coxa=+30 deg, BL/BR coxa=-20 deg, femur=-25 deg, tibia=+110 deg.\\n"
    "- Joint limits: coxa +/-60 deg, femur [-70, 0] deg, tibia [0, 150] deg.\\n"
    "- Actuator model: CubeMars AK45-36 KV80, peak torque 24 Nm, continuous threshold 8 Nm, runtime velocity cap 3 rad/s, "
    "reflected armature about 0.0236 kg*m^2, stiffness 40, damping 1.\\n\\n"
    "REWARD DESIGN PRIOR:\\n"
    "- Main objective: dense planar velocity tracking and yaw-rate tracking while alive and upright.\\n"
    "- Penalize vertical velocity, roll/pitch angular velocity, tilt, base contact, front-arm ground contact, excessive torque, joint acceleration, and action-rate spikes.\\n"
    "- Use the four spherical foot contacts for gait shaping. Do not include tibia links or fixed front arms as feet.\\n"
    "- A diagonal trot prior is useful: pair A [BL, MR], pair B [BR, ML]. Encourage one pair to stance/anchor while the other pair swings, then alternate.\\n"
    "- Keep gait timing soft. Never prescribe joint angles, overwrite actions, or reward clock phase alone.\\n"
    "- Primary footfall objective: stance anchoring. When a walking tibia touches down, its world XY position should stay nearly fixed until liftoff.\\n"
    "- Encourage comparable stride length and step count across the four walking legs.\\n"
    "- Penalize the fixed front arms if they touch or drag on the floor; they should remain suspended payload/arm structure.\\n"
    "- Keep every returned component shape exactly (self.num_envs,).\\n"
)
```

### prompt_templates.py

The current `PROGRAM_MD` in `prompt_templates.py` is already useful. Preserve
these standing orders:

- LLM may only edit `_get_rewards_eureka(self)`.
- It must return `(total_reward, dict[str, tensor])`.
- It must not change success metric, env class, observations, actions, or
  terminations.
- Prefer coefficient changes before large rewrites.
- Avoid sparse, discontinuous, unbounded rewards.
- Keep smoothness/effort secondary to command tracking.
- If shaped reward rises while success metric does not, treat it as reward
  hacking and reduce the terms that rose.

For the quad task, add or emphasize:

- This is not a hexapod tripod problem anymore.
- Use a diagonal four-leg trot prior: `[BL, MR]` vs `[BR, ML]`.
- The fixed front legs are arms/payload structure; do not treat their tibias as
  feet.
- Penalize front-arm ground contact/drag.
- Do not reward all six tibias touching; only four tibias are locomotion feet.

## Dr. Eureka Launch Commands

Store key locally without echoing it:

```bash
umask 077
printf '%s\n' "$OPENAI_API_KEY" > /workspace/.openai_key
chmod 600 /workspace/.openai_key
```

Cheap one-candidate smoke:

```bash
cd /workspace/GRAM_DrEureka
EUREKA_NUM_ENVS=512 OPENAI_API_KEY="$(cat /workspace/.openai_key)" \
TERM=xterm /workspace/isaaclab/isaaclab.sh -p scripts/train.py \
  --task=Isaac-InsectoidMiniQuad-Flat-Direct-v0 \
  --num_parallel_runs=1 \
  --max_eureka_iterations=1 \
  --max_training_iterations=80 \
  --gpt_model=gpt-5.4 \
  --rl_library=rsl_rl
```

Full run pattern:

```bash
cd /workspace/GRAM_DrEureka
mkdir -p logs/manual_runs
run_id="insectoid_mini_quad_dreureka_full_$(date -u +%Y%m%dT%H%M%SZ)"
log="/workspace/GRAM_DrEureka/logs/manual_runs/${run_id}.log"
setsid bash -lc 'cd /workspace/GRAM_DrEureka && export OPENAI_API_KEY="$(cat /workspace/.openai_key)" && export TERM=xterm && exec /workspace/isaaclab/isaaclab.sh -p scripts/train.py --task=Isaac-InsectoidMiniQuad-Flat-Direct-v0 --num_parallel_runs=4 --max_eureka_iterations=3 --max_training_iterations=1500 --gpt_model=gpt-5.4 --rl_library=rsl_rl' > "$log" 2>&1 < /dev/null &
echo "$log"
```

Monitor:

```bash
tail -f /workspace/GRAM_DrEureka/logs/manual_runs/<run>.log
```

## Playback And Diagnostics

Create quad equivalents of the six-leg helper scripts:

- `scripts/train_rsl_rl.py`
- `scripts/render_policy_video.py`
- `scripts/diagnose_policy_saturation.py`
- `scripts/diagnose_foot_drag.py`
- `scripts/diagnose_gait_quality.py`

The six-leg render pattern that worked:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p scripts/render_policy_video.py \
  --headless \
  --enable_cameras \
  --checkpoint <checkpoint.pt> \
  --command 0.0 0.25 0.0 \
  --video-length 500 \
  --video-subdir fixed_command_y025
```

For quad diagnostics, report:

- max torque per joint and percentage above `8 Nm`
- max joint velocity and whether it exceeds `3 rad/s`
- foot drag/contact for the four walking tibias only
- any contact of front fixed tibias/femurs with the ground
- stride length per walking leg
- step counts per walking leg
- body pitch/roll/yaw drift
- world path curvature / lateral drift

## Known Lessons From The Six-Leg Project

- Visual playback is mandatory. Scalar reward can look good while the robot
  drags feet, arcs sideways, or uses static support hacks.
- Smoothness penalties can easily overpower walking. Keep joint acceleration,
  torque, and action-rate terms bounded or very small.
- Alive/survival should not be a standalone route to high return.
- Reward components should be logged separately from the total reward.
- Footfall event rewards are useful, especially touchdown stride length and
  short-stride penalties, but they can over-pressure the gait if too large.
- Stance anchoring is the most physically important contact term: planted feet
  should stay fixed while the body moves over them.
- If the policy learns many rapid tiny steps, increase minimum swing time and
  stride target gradually; do not immediately add a harsh joint-velocity
  penalty.
- If the robot falls when action low-pass filtering is added, remove the filter
  and shape speed through reward/velocity caps instead.
