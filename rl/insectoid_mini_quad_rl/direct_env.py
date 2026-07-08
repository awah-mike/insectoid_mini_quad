from __future__ import annotations

import math

import gymnasium as gym
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .articulation import INSECTOID_MINI_QUAD_CFG


FOOT_BODY_NAMES = ("BL_FOOT", "BR_FOOT", "ML_FOOT", "MR_FOOT")
FOOT_BODY_PATTERN = "BL_FOOT|BR_FOOT|ML_FOOT|MR_FOOT"
NON_FOOT_BODY_PATTERN = "base_link|.*_coxa_1|.*_femur_1|.*_tibia_1"
BL_FOOT_INDEX = 0
BR_FOOT_INDEX = 1
BL_COXA_JOINT_NAME = "BL_coxa_joint"
BR_COXA_JOINT_NAME = "BR_coxa_joint"
BL_FEMUR_JOINT_NAME = "BL_femur_joint"
BR_FEMUR_JOINT_NAME = "BR_femur_joint"
BL_TIBIA_JOINT_NAME = "BL_tibia_joint"
BR_TIBIA_JOINT_NAME = "BR_tibia_joint"
BL_FEMUR_BODY_NAME = "BL_femur_1"
BR_FEMUR_BODY_NAME = "BR_femur_1"
BL_TIBIA_BODY_NAME = "BL_tibia_1"
BR_TIBIA_BODY_NAME = "BR_tibia_1"
DIAGONAL_PAIR_A = (0, 3)  # BL + MR
DIAGONAL_PAIR_B = (1, 2)  # BR + ML


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rpy_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=1)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )


@configclass
class InsectoidMiniQuadFlatEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_target_filter_alpha = 1.0
    action_space = 12
    observation_space = 60
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    events: EventCfg = EventCfg()

    robot = INSECTOID_MINI_QUAD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    lin_vel_x_range = (-0.03, 0.03)
    lin_vel_y_range = (0.20, 0.35)
    ang_vel_z_range = (-0.05, 0.05)
    rel_standing_envs = 0.0

    lin_vel_reward_scale = 1.0
    lin_vel_tracking_sigma = 0.25
    command_progress_reward_scale = 0.0
    command_progress_floor_reward_scale = 0.0
    min_command_progress_ratio = 0.0
    moving_command_threshold = 0.10
    yaw_rate_reward_scale = 0.5
    yaw_drift_reward_scale = 0.0
    yaw_drift_tolerance_rad = 0.20
    lateral_velocity_reward_scale = 0.0
    lateral_position_reward_scale = 0.0
    lateral_position_tolerance_m = 0.10
    command_overspeed_reward_scale = 0.0
    command_overspeed_tolerance = 0.10
    z_vel_reward_scale = -2.0
    base_height_floor_reward_scale = 0.0
    base_height_floor_m = 0.0
    ang_vel_reward_scale = -0.05
    ang_vel_z_reward_scale = 0.0
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 1.5
    feet_air_time_target = 0.50
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -8.0
    flat_orientation_pitch_weight = 1.0
    pitch_forward_reward_scale = -8.0
    front_up_target_reward_scale = 0.0
    front_up_target_projected_gravity_y = 0.0
    front_down_hinge_reward_scale = 0.0
    front_down_hinge_tolerance = 0.0
    front_up_progress_gate_ratio = 0.0
    front_up_progress_gate_min_weight = 1.0
    front_up_progress_bonus_reward_scale = 0.0
    front_up_progress_bonus_start_projected_gravity_y = -math.sin(math.radians(4.0))
    front_up_reset_deg = 0.0
    front_up_reset_root_z_offset = 0.0
    action_center_offsets = {}
    action_residual_limits = {}
    reset_joint_pos_offsets = {}
    front_up_posture_joint_offsets = {}
    front_up_posture_reward_scale = 0.0
    front_up_posture_command_reward_scale = 0.0
    front_up_posture_tolerance_rad = 0.16
    stance_foot_slip_reward_scale = -0.45
    low_swing_drag_reward_scale = -0.25
    low_swing_clearance_m = 0.025
    bl_low_swing_drag_reward_scale = 0.0
    bl_low_swing_time_reward_scale = 0.0
    swing_foot_speed_reward_scale = 0.0
    swing_foot_speed_target = 0.35
    support_distribution_reward_scale = -0.30
    contact_balance_reward_scale = 0.0
    diagonal_trot_reward_scale = 0.0
    diagonal_pair_sync_reward_scale = 0.0
    side_antiphase_reward_scale = 0.0
    rear_contact_antiphase_reward_scale = 0.0
    touchdown_cadence_reward_scale = 0.0
    touchdown_rate_balance_reward_scale = 0.0
    rapid_touchdown_reward_scale = 0.0
    touchdown_stride_reward_scale = 2.2
    touchdown_short_stride_reward_scale = 0.0
    stride_ema_shortfall_reward_scale = 0.0
    stride_ema_floor = 0.10
    stance_anchor_slip_reward_scale = 0.0
    rear_stride_balance_reward_scale = 0.0
    rear_stance_duration_balance_reward_scale = 0.0
    rear_touchdown_rate_balance_reward_scale = 0.0
    rear_contact_fraction_balance_reward_scale = 0.0
    rear_contact_fraction_target_reward_scale = 0.0
    bl_contact_fraction_excess_reward_scale = 0.0
    rear_stance_slip_balance_reward_scale = 0.0
    rear_left_stance_slip_excess_reward_scale = 0.0
    rear_lateral_mirror_reward_scale = 0.0
    rear_tibia_mean_symmetry_reward_scale = 0.0
    rear_tibia_mean_target_reward_scale = 0.0
    rear_tibia_mean_curl_floor_reward_scale = 0.0
    rear_tibia_command_symmetry_reward_scale = 0.0
    rear_tibia_command_target_reward_scale = 0.0
    rear_tibia_command_curl_floor_reward_scale = 0.0
    rear_tibia_tuck_envelope_reward_scale = 0.0
    bl_tibia_tuck_reward_scale = 0.0
    bl_tibia_pair_offset_reward_scale = 0.0
    bl_coxa_pair_offset_reward_scale = 0.0
    bl_femur_pair_offset_reward_scale = 0.0
    bl_coxa_command_pair_offset_reward_scale = 0.0
    bl_femur_command_pair_offset_reward_scale = 0.0
    bl_tibia_command_pair_offset_reward_scale = 0.0
    rear_coxa_joint_symmetry_reward_scale = 0.0
    rear_femur_joint_symmetry_reward_scale = 0.0
    rear_tibia_joint_symmetry_reward_scale = 0.0
    rear_coxa_command_symmetry_reward_scale = 0.0
    rear_femur_command_symmetry_reward_scale = 0.0
    rear_tibia_command_symmetry_abs_reward_scale = 0.0
    rear_femur_tibia_segment_mirror_reward_scale = 0.0
    rear_femur_tibia_z_mirror_reward_scale = 0.0
    rear_tibia_link_mirror_reward_scale = 0.0
    rear_tibia_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_axis_mirror_reward_scale = 0.0
    rear_tibia_primary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_primary_axis_mirror_reward_scale = 0.0
    rear_tibia_secondary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_secondary_axis_mirror_reward_scale = 0.0
    rear_tibia_mesh_long_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_mesh_long_axis_mirror_reward_scale = 0.0
    rear_tibia_mesh_secondary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_mesh_secondary_axis_mirror_reward_scale = 0.0
    rear_tibia_mesh_long_lateral_excess_reward_scale = 0.0
    bl_swing_tibia_mesh_long_lateral_excess_reward_scale = 0.0
    rear_tibia_mesh_secondary_lateral_excess_reward_scale = 0.0
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = 0.0
    rear_tibia_mesh_secondary_lateral_target_reward_scale = 0.0
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = 0.0
    br_swing_tibia_mesh_secondary_lateral_target_reward_scale = 0.0
    bl_fast_swing_rear_joint_pose_reward_scale = 0.0
    bl_fast_swing_rear_command_pose_reward_scale = 0.0
    bl_tibia_link_outward_vs_br_reward_scale = 0.0
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = 0.0
    bl_tibia_link_outward_floor_reward_scale = 0.0
    bl_lifted_swing_tibia_link_outward_floor_reward_scale = 0.0
    rear_tibia_segment_mirror_reward_scale = 0.0
    rear_tibia_segment_outward_reward_scale = 0.0
    rear_tibia_segment_outward_target_reward_scale = 0.0
    rear_swing_tibia_segment_outward_reward_scale = 0.0
    bl_swing_tibia_segment_outward_reward_scale = 0.0
    bl_lifted_swing_tibia_segment_outward_reward_scale = 0.0
    bl_phase_delayed_tibia_outward_reward_scale = 0.0
    bl_phase_delayed_tibia_link_outward_reward_scale = 0.0
    bl_lifted_phase_delayed_tibia_link_outward_reward_scale = 0.0
    bl_phase_delayed_tibia_link_mirror_reward_scale = 0.0
    bl_lifted_phase_delayed_tibia_link_mirror_reward_scale = 0.0
    bl_phase_delayed_femur_tibia_mirror_reward_scale = 0.0
    bl_lifted_phase_delayed_femur_tibia_mirror_reward_scale = 0.0
    bl_phase_delayed_tibia_primary_axis_mirror_reward_scale = 0.0
    bl_lifted_phase_delayed_tibia_primary_axis_mirror_reward_scale = 0.0
    bl_phase_delayed_tibia_secondary_axis_mirror_reward_scale = 0.0
    bl_lifted_phase_delayed_tibia_secondary_axis_mirror_reward_scale = 0.0
    bl_phase_delayed_rear_coxa_joint_symmetry_reward_scale = 0.0
    bl_lifted_phase_delayed_rear_coxa_joint_symmetry_reward_scale = 0.0
    bl_phase_delayed_rear_coxa_command_symmetry_reward_scale = 0.0
    bl_lifted_phase_delayed_rear_coxa_command_symmetry_reward_scale = 0.0
    bl_phase_delayed_rear_distal_joint_symmetry_reward_scale = 0.0
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_reward_scale = 0.0
    bl_phase_delayed_rear_distal_command_symmetry_reward_scale = 0.0
    bl_lifted_phase_delayed_rear_distal_command_symmetry_reward_scale = 0.0
    bl_phase_delayed_foot_height_reward_scale = 0.0
    rear_swing_outward_mean_balance_reward_scale = 0.0
    bl_swing_outward_mean_floor_reward_scale = 0.0
    bl_swing_outward_vs_br_reward_scale = 0.0
    bl_lifted_swing_outward_vs_br_reward_scale = 0.0
    rear_swing_femur_floor_reward_scale = 0.0
    bl_swing_femur_floor_reward_scale = 0.0
    rear_swing_femur_command_floor_reward_scale = 0.0
    bl_swing_femur_command_floor_reward_scale = 0.0
    # STEP HEIGHT TEST: kept reward for clearer swing lift.
    swing_height_reward_scale = 0.70
    bl_swing_height_reward_scale = 0.0
    short_stance_reward_scale = -0.35
    short_swing_reward_scale = -0.35
    min_touchdown_stride = 0.05
    touchdown_stride_target = 0.14
    touchdown_cadence_target_hz = 3.0
    stance_anchor_slip_tolerance_m = 0.015
    stance_anchor_grace_s = 0.04
    max_touchdown_rate_hz = 8.0
    rear_symmetry_ready_time_s = 1.0
    rear_contact_fraction_balance_tolerance = 0.20
    rear_contact_fraction_target = 0.52
    rear_contact_fraction_target_tolerance = 0.14
    bl_contact_fraction_ceiling = 0.54
    bl_contact_fraction_excess_tolerance = 0.08
    rear_stance_slip_balance_tolerance_mps = 0.08
    rear_left_stance_slip_tolerance_mps = 0.06
    rear_lateral_mirror_tolerance_m = 0.05
    rear_lateral_mirror_target_m = 0.0
    rear_tibia_mean_symmetry_tolerance_rad = 0.20
    rear_tibia_mean_target_rad = 2.05
    rear_tibia_mean_target_tolerance_rad = 0.22
    rear_tibia_curl_floor_rad = 1.90
    rear_tibia_curl_floor_tolerance_rad = 0.12
    rear_tibia_command_symmetry_tolerance_rad = 0.12
    rear_tibia_command_target_tolerance_rad = 0.18
    rear_tibia_tuck_limit_rad = 2.35
    rear_tibia_tuck_tolerance_rad = 0.20
    bl_tibia_tuck_limit_rad = 2.05
    bl_tibia_tuck_tolerance_rad = 0.15
    bl_tibia_pair_offset_limit_rad = 0.22
    bl_tibia_pair_offset_tolerance_rad = 0.18
    bl_coxa_pair_offset_limit_rad = 0.28
    bl_coxa_pair_offset_tolerance_rad = 0.14
    bl_femur_pair_offset_limit_rad = 0.18
    bl_femur_pair_offset_tolerance_rad = 0.14
    rear_coxa_joint_symmetry_tolerance_rad = 0.10
    rear_femur_joint_symmetry_tolerance_rad = 0.08
    rear_tibia_joint_symmetry_tolerance_rad = 0.08
    rear_coxa_command_symmetry_tolerance_rad = 0.10
    rear_femur_command_symmetry_tolerance_rad = 0.08
    rear_tibia_command_symmetry_abs_tolerance_rad = 0.08
    rear_femur_tibia_segment_mirror_tolerance_m = 0.014
    rear_femur_tibia_z_mirror_tolerance_m = 0.008
    rear_tibia_link_mirror_tolerance_m = 0.025
    rear_tibia_axis_mirror_tolerance = 0.060
    bl_lifted_tibia_axis_mirror_tolerance = 0.045
    rear_tibia_primary_axis_mirror_tolerance = 0.110
    bl_lifted_tibia_primary_axis_mirror_tolerance = 0.075
    rear_tibia_secondary_axis_mirror_tolerance = 0.125
    bl_lifted_tibia_secondary_axis_mirror_tolerance = 0.090
    rear_tibia_mesh_long_axis_mirror_tolerance = 0.090
    bl_lifted_tibia_mesh_long_axis_mirror_tolerance = 0.075
    rear_tibia_mesh_secondary_axis_mirror_tolerance = 0.075
    bl_lifted_tibia_mesh_secondary_axis_mirror_tolerance = 0.055
    rear_tibia_mesh_long_lateral_cap = 0.275
    bl_swing_tibia_mesh_long_lateral_cap = 0.265
    rear_tibia_mesh_secondary_lateral_cap = 0.375
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.360
    rear_tibia_mesh_secondary_lateral_target = 0.355
    bl_swing_tibia_mesh_secondary_lateral_target = 0.350
    br_swing_tibia_mesh_secondary_lateral_target = 0.385
    rear_tibia_mesh_long_lateral_tolerance = 0.035
    bl_swing_tibia_mesh_long_lateral_tolerance = 0.025
    rear_tibia_mesh_secondary_lateral_tolerance = 0.035
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.025
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.055
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.045
    br_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.045
    visual_cleanup_min_progress_ratio = 0.92
    bl_fast_swing_coxa_min_rad = 0.020
    bl_fast_swing_femur_min_rad = -0.345
    bl_fast_swing_tibia_max_rad = 1.560
    bl_fast_swing_coxa_tolerance_rad = 0.045
    bl_fast_swing_femur_tolerance_rad = 0.060
    bl_fast_swing_tibia_tolerance_rad = 0.060
    bl_tibia_link_outward_vs_br_tolerance_m = 0.004
    bl_lifted_swing_tibia_link_outward_vs_br_tolerance_m = 0.003
    bl_tibia_link_outward_floor_target_m = 0.042
    bl_tibia_link_outward_floor_tolerance_m = 0.010
    bl_lifted_swing_tibia_link_outward_floor_target_m = 0.045
    bl_lifted_swing_tibia_link_outward_floor_tolerance_m = 0.008
    rear_tibia_segment_mirror_tolerance_m = 0.015
    rear_tibia_segment_outward_min_m = 0.025
    rear_tibia_segment_outward_tolerance_m = 0.012
    rear_tibia_segment_outward_target_m = 0.034
    rear_tibia_segment_outward_target_tolerance_m = 0.010
    rear_swing_tibia_segment_outward_target_m = 0.043
    rear_swing_tibia_segment_outward_tolerance_m = 0.010
    bl_lifted_swing_tibia_segment_outward_target_m = 0.047
    bl_lifted_swing_tibia_segment_outward_tolerance_m = 0.007
    bl_lifted_swing_min_height_m = 0.035
    bl_lifted_swing_height_tolerance_m = 0.020
    rear_phase_symmetry_delay_steps = 12
    bl_phase_delayed_tibia_outward_tolerance_m = 0.007
    bl_phase_delayed_tibia_link_outward_tolerance_m = 0.004
    bl_lifted_phase_delayed_tibia_link_outward_tolerance_m = 0.003
    bl_phase_delayed_tibia_link_mirror_tolerance_m = 0.010
    bl_lifted_phase_delayed_tibia_link_mirror_tolerance_m = 0.009
    bl_phase_delayed_femur_tibia_mirror_tolerance_m = 0.010
    bl_lifted_phase_delayed_femur_tibia_mirror_tolerance_m = 0.009
    bl_phase_delayed_tibia_primary_axis_mirror_tolerance = 0.090
    bl_lifted_phase_delayed_tibia_primary_axis_mirror_tolerance = 0.070
    bl_phase_delayed_tibia_secondary_axis_mirror_tolerance = 0.100
    bl_lifted_phase_delayed_tibia_secondary_axis_mirror_tolerance = 0.075
    bl_phase_delayed_rear_coxa_joint_symmetry_tolerance_rad = 0.07
    bl_lifted_phase_delayed_rear_coxa_joint_symmetry_tolerance_rad = 0.055
    bl_phase_delayed_rear_coxa_command_symmetry_tolerance_rad = 0.08
    bl_lifted_phase_delayed_rear_coxa_command_symmetry_tolerance_rad = 0.065
    bl_phase_delayed_rear_distal_joint_symmetry_tolerance_rad = 0.10
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_tolerance_rad = 0.08
    bl_phase_delayed_rear_distal_command_symmetry_tolerance_rad = 0.10
    bl_lifted_phase_delayed_rear_distal_command_symmetry_tolerance_rad = 0.08
    bl_phase_delayed_foot_height_tolerance_m = 0.012
    bl_phase_delayed_foot_height_min_m = 0.040
    rear_swing_outward_mean_balance_tolerance_m = 0.004
    bl_swing_outward_mean_floor_target_m = 0.038
    bl_swing_outward_mean_floor_tolerance_m = 0.006
    bl_swing_outward_vs_br_tolerance_m = 0.004
    bl_lifted_swing_outward_vs_br_tolerance_m = 0.003
    rear_swing_femur_floor_rad = -0.43
    rear_swing_femur_floor_tolerance_rad = 0.06
    swing_height_min = 0.030
    swing_height_target = 0.075
    min_stance_time = 0.12
    min_swing_time = 0.10

    # Kept only for diagnostics and success metrics, not as direct reward terms.
    stride_length_ema_alpha = 0.2

    # Deployment observation mode. The real quad_test adapter does not have a
    # direct base linear velocity measurement, so deployment-tuned tasks replace
    # the privileged simulator velocity in obs[0:3] with a stance-foot kinematic
    # estimate computed from leg motion and IMU angular velocity.
    use_privileged_base_lin_vel_obs = True
    use_foot_kinematic_base_lin_vel_obs = False
    zero_base_lin_vel_z_obs = True
    base_lin_vel_estimator_smoothing = 0.82
    base_lin_vel_obs_max_abs = 1.0

    observation_noise_enabled = False
    use_rpy_attitude_obs = False
    rpy_attitude_obs_noise_std = 0.0
    base_lin_vel_obs_noise_std = 0.0
    angular_velocity_obs_noise_std = 0.0
    projected_gravity_obs_noise_std = 0.0
    joint_position_obs_noise_std = 0.0
    joint_velocity_obs_noise_std = 0.0
    action_obs_noise_std = 0.0
    contact_obs_dropout_prob = 0.0
    timer_obs_noise_std = 0.0
    zero_contact_obs = False
    zero_contact_timer_obs = False
    use_open_loop_gait_phase_obs = False
    open_loop_gait_period_s = 0.64
    open_loop_gait_duty_factor = 0.58


@configclass
class InsectoidMiniQuadFlatPlayEnvCfg(InsectoidMiniQuadFlatEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineEnvCfg(InsectoidMiniQuadFlatEnvCfg):
    action_target_filter_alpha = 0.60

    lin_vel_x_range = (-0.015, 0.015)
    lin_vel_y_range = (0.20, 0.28)
    ang_vel_z_range = (-0.025, 0.025)

    lin_vel_reward_scale = 1.25
    lin_vel_tracking_sigma = 0.18
    yaw_rate_reward_scale = 0.75
    yaw_drift_reward_scale = -0.35
    lateral_velocity_reward_scale = -0.30
    joint_accel_reward_scale = -6.0e-7
    action_rate_reward_scale = -0.035
    feet_air_time_reward_scale = 0.35
    feet_air_time_target = 0.20
    flat_orientation_reward_scale = -10.0
    pitch_forward_reward_scale = -10.0
    stance_foot_slip_reward_scale = -0.70
    low_swing_drag_reward_scale = -0.40
    support_distribution_reward_scale = -0.45
    contact_balance_reward_scale = -0.25
    diagonal_trot_reward_scale = 0.40
    diagonal_pair_sync_reward_scale = 0.0
    side_antiphase_reward_scale = 0.0
    touchdown_cadence_reward_scale = -0.45
    touchdown_rate_balance_reward_scale = -0.18
    rapid_touchdown_reward_scale = -0.55
    touchdown_stride_reward_scale = 4.0
    touchdown_short_stride_reward_scale = -2.8
    stance_anchor_slip_reward_scale = -0.85
    swing_height_reward_scale = 0.95
    short_stance_reward_scale = -1.15
    short_swing_reward_scale = -1.15
    min_touchdown_stride = 0.08
    touchdown_stride_target = 0.18
    touchdown_cadence_target_hz = 2.0
    stance_anchor_grace_s = 0.06
    max_touchdown_rate_hz = 3.0
    min_stance_time = 0.20
    min_swing_time = 0.16


@configclass
class InsectoidMiniQuadGaitRefinePlayEnvCfg(InsectoidMiniQuadGaitRefineEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceEnvCfg(InsectoidMiniQuadGaitRefineEnvCfg):
    # Slow the accepted forward gait by lengthening the step cycle, not by
    # shrinking stride. The command is about half of the 1798 evaluation speed.
    action_target_filter_alpha = 0.25

    lin_vel_y_range = (0.14, 0.18)
    lin_vel_tracking_sigma = 0.035
    lin_vel_reward_scale = 1.75
    command_progress_reward_scale = 0.0
    command_progress_floor_reward_scale = 0.0
    min_command_progress_ratio = 0.0
    command_overspeed_reward_scale = -2.50
    command_overspeed_tolerance = 0.030

    joint_accel_reward_scale = -1.2e-6
    action_rate_reward_scale = -0.070

    feet_air_time_reward_scale = 0.55
    feet_air_time_target = 0.36
    touchdown_cadence_reward_scale = -3.00
    touchdown_cadence_target_hz = 0.95
    rapid_touchdown_reward_scale = -4.00
    max_touchdown_rate_hz = 1.60
    short_stance_reward_scale = -5.00
    short_swing_reward_scale = -5.50
    min_stance_time = 0.36
    min_swing_time = 0.30

    touchdown_stride_reward_scale = 10.00
    touchdown_short_stride_reward_scale = -12.00
    touchdown_stride_target = 0.18
    min_touchdown_stride = 0.12
    stride_ema_shortfall_reward_scale = 0.0
    stride_ema_floor = 0.10
    stance_anchor_slip_reward_scale = -0.95
    low_swing_drag_reward_scale = -0.45
    swing_foot_speed_reward_scale = 0.0
    swing_foot_speed_target = 0.48
    swing_height_reward_scale = 1.00

    yaw_drift_reward_scale = -0.45
    lateral_velocity_reward_scale = -0.35
    contact_balance_reward_scale = -0.35
    touchdown_rate_balance_reward_scale = -0.25
    rear_stride_balance_reward_scale = -0.80
    rear_stance_duration_balance_reward_scale = -0.35
    rear_touchdown_rate_balance_reward_scale = -0.45


@configclass
class InsectoidMiniQuadGaitRefineSlowCadencePlayEnvCfg(InsectoidMiniQuadGaitRefineSlowCadenceEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceStrideLockEnvCfg(InsectoidMiniQuadGaitRefineSlowCadenceEnvCfg):
    # Follow-up from the v6 slow-cadence pass: tighten speed tracking while
    # making short-step solutions expensive enough that the policy should slow
    # cadence instead of just shrinking stride.
    lin_vel_y_range = (0.14, 0.16)
    lin_vel_tracking_sigma = 0.025
    lin_vel_reward_scale = 2.25
    command_overspeed_reward_scale = -3.35
    command_overspeed_tolerance = 0.015

    touchdown_cadence_reward_scale = -5.00
    touchdown_cadence_target_hz = 0.95
    rapid_touchdown_reward_scale = -6.50
    max_touchdown_rate_hz = 1.35
    short_stance_reward_scale = -6.00
    short_swing_reward_scale = -7.00
    min_stance_time = 0.38
    min_swing_time = 0.34

    touchdown_stride_reward_scale = 16.00
    touchdown_short_stride_reward_scale = -22.00
    touchdown_stride_target = 0.18
    min_touchdown_stride = 0.115
    stride_ema_shortfall_reward_scale = -4.00
    stride_ema_floor = 0.095

    stance_anchor_slip_reward_scale = -1.15
    low_swing_drag_reward_scale = -0.55
    action_rate_reward_scale = -0.080


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceStrideLockPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceStrideLockEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceStrideLockEnvCfg
):
    # Front/back attitude uses projected_gravity_b.y because this robot's
    # forward axis is body +Y. A positive front-up angle maps to -sin(angle).
    front_up_target_projected_gravity_y = -math.sin(math.radians(10.0))
    front_up_target_reward_scale = -75.0
    front_up_reset_deg = 8.0
    front_up_reset_root_z_offset = 0.015

    # Keep lateral roll flat, but stop the old flat/pitch terms from fighting
    # the intentional front-up posture.
    flat_orientation_pitch_weight = 0.0
    pitch_forward_reward_scale = 0.0
    flat_orientation_reward_scale = -14.0

    # Give the policy a little more room to rediscover balance around the
    # reclined body target without throwing away the slow-cadence objective.
    yaw_drift_reward_scale = -0.55
    contact_balance_reward_scale = -0.45
    action_rate_reward_scale = -0.070


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquireEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpEnvCfg
):
    # Posture acquisition phase with a hard progress floor: keep the body from
    # settling front-down without letting the policy "solve" posture by stopping.
    lin_vel_y_range = (0.12, 0.16)
    lin_vel_tracking_sigma = 0.030
    lin_vel_reward_scale = 3.00
    command_progress_floor_reward_scale = -3.00
    min_command_progress_ratio = 0.70
    command_overspeed_reward_scale = -2.50
    command_overspeed_tolerance = 0.030

    front_up_target_reward_scale = -110.0
    front_down_hinge_reward_scale = -260.0
    front_down_hinge_tolerance = 0.015
    front_up_reset_deg = 2.0
    front_up_reset_root_z_offset = 0.020

    touchdown_cadence_reward_scale = -4.00
    rapid_touchdown_reward_scale = -5.00
    max_touchdown_rate_hz = 1.55
    short_stance_reward_scale = -4.80
    short_swing_reward_scale = -5.50
    min_stance_time = 0.34
    min_swing_time = 0.30

    touchdown_stride_reward_scale = 13.00
    touchdown_short_stride_reward_scale = -17.00
    stride_ema_shortfall_reward_scale = -2.00

    yaw_drift_reward_scale = -0.70
    lateral_velocity_reward_scale = -0.45
    contact_balance_reward_scale = -0.45
    action_rate_reward_scale = -0.070


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquirePlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquireEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpStrongEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquireEnvCfg
):
    # Strong acquisition pass from clean model_1798: start episodes at the
    # requested 10 degree posture and make front-down settling much more
    # expensive while keeping the slow-cadence command alive.
    lin_vel_y_range = (0.12, 0.15)
    lin_vel_tracking_sigma = 0.035
    lin_vel_reward_scale = 3.25
    command_progress_floor_reward_scale = -3.50
    min_command_progress_ratio = 0.60
    command_overspeed_reward_scale = -2.20
    command_overspeed_tolerance = 0.040

    front_up_target_reward_scale = -520.0
    front_down_hinge_reward_scale = -900.0
    front_down_hinge_tolerance = 0.0
    front_up_reset_deg = 10.0
    front_up_reset_root_z_offset = 0.030

    flat_orientation_reward_scale = -8.0
    yaw_drift_reward_scale = -0.85
    lateral_velocity_reward_scale = -0.55
    contact_balance_reward_scale = -0.42
    action_rate_reward_scale = -0.060

    touchdown_cadence_reward_scale = -3.50
    rapid_touchdown_reward_scale = -4.50
    max_touchdown_rate_hz = 1.60
    touchdown_stride_reward_scale = 11.00
    touchdown_short_stride_reward_scale = -14.00
    stride_ema_shortfall_reward_scale = -1.50


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpStrongPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpStrongEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpStrongEnvCfg
):
    # Start from a mechanically plausible front-up support posture.  The
    # previous front-up runs reset the base pitch, but the default flat leg
    # geometry pulled the body back down before PPO found a useful basin.
    lin_vel_y_range = (0.10, 0.14)
    lin_vel_tracking_sigma = 0.040
    lin_vel_reward_scale = 2.75
    command_progress_floor_reward_scale = -2.50
    min_command_progress_ratio = 0.48
    command_overspeed_reward_scale = -1.80
    command_overspeed_tolerance = 0.045

    front_up_target_reward_scale = -720.0
    front_down_hinge_reward_scale = -1400.0
    front_down_hinge_tolerance = -0.020
    front_up_reset_deg = 10.0
    front_up_reset_root_z_offset = 0.055
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.18,
        "BR_femur_joint": -0.18,
        "ML_femur_joint": 0.28,
        "MR_femur_joint": 0.28,
    }

    flat_orientation_reward_scale = -16.0
    yaw_drift_reward_scale = -0.70
    lateral_velocity_reward_scale = -0.45
    contact_balance_reward_scale = -0.35
    action_rate_reward_scale = -0.055

    touchdown_cadence_reward_scale = -2.80
    rapid_touchdown_reward_scale = -3.40
    max_touchdown_rate_hz = 1.75
    touchdown_stride_reward_scale = 9.50
    touchdown_short_stride_reward_scale = -10.50
    stride_ema_shortfall_reward_scale = -1.00


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV2EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedEnvCfg
):
    # V1 learned to walk but relaxed back toward flat/front-down.  V2 keeps the
    # support geometry active in the reward so PPO cannot discard the pitched
    # leg posture immediately after reset.
    lin_vel_y_range = (0.08, 0.12)
    lin_vel_tracking_sigma = 0.045
    lin_vel_reward_scale = 2.35
    command_progress_floor_reward_scale = -2.10
    min_command_progress_ratio = 0.42
    command_overspeed_reward_scale = -1.35
    command_overspeed_tolerance = 0.050

    front_up_target_reward_scale = -980.0
    front_down_hinge_reward_scale = -2200.0
    front_down_hinge_tolerance = -0.035
    front_up_reset_deg = 10.0
    front_up_reset_root_z_offset = 0.075
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.32,
        "BR_femur_joint": -0.32,
        "ML_femur_joint": 0.48,
        "MR_femur_joint": 0.48,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.30,
        "BR_femur_joint": -0.30,
        "ML_femur_joint": 0.44,
        "MR_femur_joint": 0.44,
    }
    front_up_posture_reward_scale = -18.0
    front_up_posture_command_reward_scale = -10.0
    front_up_posture_tolerance_rad = 0.12

    flat_orientation_reward_scale = -11.0
    yaw_drift_reward_scale = -0.62
    lateral_velocity_reward_scale = -0.40
    contact_balance_reward_scale = -0.30
    action_rate_reward_scale = -0.050

    touchdown_cadence_reward_scale = -2.10
    rapid_touchdown_reward_scale = -2.55
    max_touchdown_rate_hz = 1.85
    touchdown_stride_reward_scale = 7.50
    touchdown_short_stride_reward_scale = -7.50
    min_touchdown_stride = 0.075
    stride_ema_shortfall_reward_scale = -0.60


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV2PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV2EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV3EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedEnvCfg
):
    # Static sweep on 2026-07-01 showed rear-femur support alone settles at
    # almost exactly 10 deg front-up.  Keep that simpler support basin instead
    # of the mixed rear/middle geometry that overshot statically and then
    # collapsed toward flat during PPO.
    lin_vel_y_range = (0.08, 0.12)
    lin_vel_tracking_sigma = 0.045
    lin_vel_reward_scale = 2.45
    command_progress_floor_reward_scale = -2.25
    min_command_progress_ratio = 0.45
    command_overspeed_reward_scale = -1.45
    command_overspeed_tolerance = 0.050

    front_up_target_reward_scale = -1050.0
    front_down_hinge_reward_scale = -2300.0
    front_down_hinge_tolerance = -0.040
    front_up_reset_deg = 10.0
    front_up_reset_root_z_offset = 0.080
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.50,
        "BR_femur_joint": -0.50,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.50,
        "BR_femur_joint": -0.50,
    }
    front_up_posture_reward_scale = -4.0
    front_up_posture_command_reward_scale = -5.0
    front_up_posture_tolerance_rad = 0.24

    flat_orientation_reward_scale = -9.0
    yaw_drift_reward_scale = -0.68
    lateral_velocity_reward_scale = -0.42
    contact_balance_reward_scale = -0.32
    action_rate_reward_scale = -0.052

    touchdown_cadence_reward_scale = -2.25
    rapid_touchdown_reward_scale = -2.75
    max_touchdown_rate_hz = 1.80
    touchdown_stride_reward_scale = 8.00
    touchdown_short_stride_reward_scale = -8.00
    min_touchdown_stride = 0.080
    stride_ema_shortfall_reward_scale = -0.70


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV3PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV3EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV4EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV3EnvCfg
):
    # V4 trains around a pitched action center.  This lets the resumed 1798
    # locomotion policy keep its learned action residuals while the neutral
    # rear-femur command already supports the target body attitude.
    action_center_offsets = {
        "BL_femur_joint": -0.50,
        "BR_femur_joint": -0.50,
    }
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.50,
        "BR_femur_joint": -0.50,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.50,
        "BR_femur_joint": -0.50,
    }
    front_up_target_reward_scale = -850.0
    front_down_hinge_reward_scale = -1800.0
    front_up_posture_reward_scale = -2.5
    front_up_posture_command_reward_scale = -3.0
    front_up_posture_tolerance_rad = 0.30
    command_overspeed_reward_scale = -1.85
    touchdown_cadence_reward_scale = -2.60
    rapid_touchdown_reward_scale = -3.10
    action_rate_reward_scale = -0.060


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV4PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV4EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV4EnvCfg
):
    # V5 backs off the action-center pitch from -0.50 to -0.40 rad.  The static
    # sweep measured this around 7.5 deg front-up, which should preserve more
    # rear-leg range while still biasing the policy toward the reclined body.
    lin_vel_y_range = (0.10, 0.14)
    lin_vel_tracking_sigma = 0.032
    lin_vel_reward_scale = 3.65
    command_progress_floor_reward_scale = -5.25
    min_command_progress_ratio = 0.68
    command_overspeed_reward_scale = -1.35
    command_overspeed_tolerance = 0.050

    action_center_offsets = {
        "BL_femur_joint": -0.40,
        "BR_femur_joint": -0.40,
    }
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.40,
        "BR_femur_joint": -0.40,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.40,
        "BR_femur_joint": -0.40,
    }
    front_up_target_reward_scale = -900.0
    front_down_hinge_reward_scale = -2000.0
    front_up_posture_reward_scale = -1.8
    front_up_posture_command_reward_scale = -2.2
    front_up_posture_tolerance_rad = 0.34

    flat_orientation_reward_scale = -8.0
    yaw_drift_reward_scale = -0.72
    lateral_velocity_reward_scale = -0.45
    contact_balance_reward_scale = -0.28
    action_rate_reward_scale = -0.052

    touchdown_cadence_reward_scale = -2.15
    rapid_touchdown_reward_scale = -2.50
    max_touchdown_rate_hz = 1.85
    touchdown_stride_reward_scale = 10.50
    touchdown_short_stride_reward_scale = -9.00
    min_touchdown_stride = 0.085
    stride_ema_shortfall_reward_scale = -0.80


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV6EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5EnvCfg
):
    # V6 pushes closer to the requested 10 deg front-up gait without returning
    # to the V4 stall mode.  The -0.45 rad rear-femur center sits between the
    # stable-but-underpitched V5 and the statically correct-but-crouched V4.
    lin_vel_y_range = (0.11, 0.15)
    lin_vel_tracking_sigma = 0.030
    lin_vel_reward_scale = 4.20
    command_progress_floor_reward_scale = -7.25
    min_command_progress_ratio = 0.72
    command_overspeed_reward_scale = -1.60
    command_overspeed_tolerance = 0.045

    action_center_offsets = {
        "BL_femur_joint": -0.45,
        "BR_femur_joint": -0.45,
    }
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.45,
        "BR_femur_joint": -0.45,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.45,
        "BR_femur_joint": -0.45,
    }
    front_up_target_reward_scale = -1350.0
    front_down_hinge_reward_scale = -3000.0
    front_down_hinge_tolerance = -0.070
    front_up_posture_reward_scale = -2.4
    front_up_posture_command_reward_scale = -2.8
    front_up_posture_tolerance_rad = 0.36

    flat_orientation_reward_scale = -7.0
    yaw_drift_reward_scale = -0.86
    lateral_velocity_reward_scale = -0.58
    contact_balance_reward_scale = -0.48
    support_distribution_reward_scale = -0.28
    action_rate_reward_scale = -0.050

    touchdown_cadence_reward_scale = -2.00
    rapid_touchdown_reward_scale = -2.35
    max_touchdown_rate_hz = 1.85
    touchdown_stride_reward_scale = 11.50
    touchdown_short_stride_reward_scale = -10.00
    min_touchdown_stride = 0.088
    stride_ema_shortfall_reward_scale = -0.85


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV6PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV6EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV7EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5EnvCfg
):
    # V7 uses the mixed rear/middle support shape from the static sweep instead
    # of rear-only pitch bias.  That pose measured around 8 deg front-up while
    # leaving the rear pair less crouched than V4/V6, so PPO has a cleaner
    # basin to adapt from the accepted model_1798 gait.
    lin_vel_y_range = (0.10, 0.14)
    lin_vel_tracking_sigma = 0.034
    lin_vel_reward_scale = 3.85
    command_progress_floor_reward_scale = -5.75
    min_command_progress_ratio = 0.66
    command_overspeed_reward_scale = -1.35
    command_overspeed_tolerance = 0.050

    action_center_offsets = {
        "BL_femur_joint": -0.18,
        "BR_femur_joint": -0.18,
        "ML_femur_joint": 0.28,
        "MR_femur_joint": 0.28,
    }
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.18,
        "BR_femur_joint": -0.18,
        "ML_femur_joint": 0.28,
        "MR_femur_joint": 0.28,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.18,
        "BR_femur_joint": -0.18,
        "ML_femur_joint": 0.28,
        "MR_femur_joint": 0.28,
    }
    front_up_target_reward_scale = -1180.0
    front_down_hinge_reward_scale = -2500.0
    front_down_hinge_tolerance = -0.055
    front_up_posture_reward_scale = -2.2
    front_up_posture_command_reward_scale = -2.6
    front_up_posture_tolerance_rad = 0.32

    flat_orientation_reward_scale = -7.5
    yaw_drift_reward_scale = -0.80
    lateral_velocity_reward_scale = -0.52
    contact_balance_reward_scale = -0.42
    support_distribution_reward_scale = -0.24
    action_rate_reward_scale = -0.052

    touchdown_cadence_reward_scale = -2.10
    rapid_touchdown_reward_scale = -2.45
    max_touchdown_rate_hz = 1.85
    touchdown_stride_reward_scale = 10.75
    touchdown_short_stride_reward_scale = -9.50
    min_touchdown_stride = 0.085
    stride_ema_shortfall_reward_scale = -0.80


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV7PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV7EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV8EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5EnvCfg
):
    # V8 deliberately lowers the command speed while enforcing the front-up
    # posture.  V6/V7 showed that the faster command basin makes the policy
    # flatten the torso; this branch asks for the slow 10 deg gait directly.
    lin_vel_y_range = (0.055, 0.085)
    lin_vel_tracking_sigma = 0.022
    lin_vel_reward_scale = 3.80
    command_progress_floor_reward_scale = -5.40
    min_command_progress_ratio = 0.62
    moving_command_threshold = 0.04
    command_overspeed_reward_scale = -1.10
    command_overspeed_tolerance = 0.045

    action_center_offsets = {
        "BL_femur_joint": -0.50,
        "BR_femur_joint": -0.50,
    }
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.50,
        "BR_femur_joint": -0.50,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.50,
        "BR_femur_joint": -0.50,
    }
    front_up_target_reward_scale = -1650.0
    front_down_hinge_reward_scale = -4200.0
    front_down_hinge_tolerance = -0.105
    front_up_posture_reward_scale = -2.4
    front_up_posture_command_reward_scale = -2.8
    front_up_posture_tolerance_rad = 0.36

    flat_orientation_reward_scale = -6.5
    yaw_drift_reward_scale = -0.70
    lateral_velocity_reward_scale = -0.45
    contact_balance_reward_scale = -0.42
    support_distribution_reward_scale = -0.25
    action_rate_reward_scale = -0.048

    touchdown_cadence_target_hz = 0.75
    touchdown_cadence_reward_scale = -2.00
    rapid_touchdown_reward_scale = -2.35
    max_touchdown_rate_hz = 1.45
    touchdown_stride_reward_scale = 9.00
    touchdown_short_stride_reward_scale = -7.50
    min_touchdown_stride = 0.060
    stride_ema_shortfall_reward_scale = -0.65
    stride_ema_floor = 0.060
    min_stance_time = 0.42
    min_swing_time = 0.36


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV8PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV8EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV9EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV8EnvCfg
):
    # V9 keeps the V8 slow command gates, but clips rear-femur residual actions
    # so the policy cannot cancel the pitched support posture.  Limits are in
    # normalized action units before the global action_scale is applied.
    action_residual_limits = {
        "BL_femur_joint": (-0.80, 0.20),
        "BR_femur_joint": (-0.80, 0.20),
    }
    lin_vel_reward_scale = 4.10
    command_progress_floor_reward_scale = -6.25
    min_command_progress_ratio = 0.58
    front_up_target_reward_scale = -1350.0
    front_down_hinge_reward_scale = -3600.0
    contact_balance_reward_scale = -0.50
    support_distribution_reward_scale = -0.30


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV9PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV9EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV10EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV9EnvCfg
):
    # V9 found a 7-8 deg front-up posture, but did it by crouching to ~0.125 m
    # and generating non-foot contacts.  V10 keeps the pitched rear support
    # basin but adds a base-height floor and relaxes the residual femur clamp
    # enough that the policy can step instead of sitting on the body.
    action_residual_limits = {
        "BL_femur_joint": (-0.80, 0.45),
        "BR_femur_joint": (-0.80, 0.45),
    }
    front_up_reset_root_z_offset = 0.12
    base_height_floor_m = 0.18
    base_height_floor_reward_scale = -160.0
    undesired_contact_reward_scale = -3.25

    lin_vel_reward_scale = 4.25
    command_progress_floor_reward_scale = -6.80
    min_command_progress_ratio = 0.60
    command_overspeed_reward_scale = -1.20
    command_overspeed_tolerance = 0.050

    front_up_target_reward_scale = -1250.0
    front_down_hinge_reward_scale = -3200.0
    front_down_hinge_tolerance = -0.060
    front_up_posture_reward_scale = -2.0
    front_up_posture_command_reward_scale = -2.4
    front_up_posture_tolerance_rad = 0.34

    flat_orientation_reward_scale = -5.8
    yaw_drift_reward_scale = -0.78
    lateral_velocity_reward_scale = -0.48
    contact_balance_reward_scale = -0.55
    support_distribution_reward_scale = -0.32
    action_rate_reward_scale = -0.050


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV10PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV10EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV11EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5EnvCfg
):
    # V11 resumes from the best walking front-up branch (V5/model_1955) and
    # nudges attitude upward from ~6.5 deg without re-entering the crouched
    # -0.50 rear-femur basin that broke V9/V10 locomotion.
    lin_vel_y_range = (0.09, 0.13)
    lin_vel_tracking_sigma = 0.034
    lin_vel_reward_scale = 3.90
    command_progress_floor_reward_scale = -5.80
    min_command_progress_ratio = 0.64
    moving_command_threshold = 0.05
    command_overspeed_reward_scale = -1.25
    command_overspeed_tolerance = 0.055

    front_up_reset_root_z_offset = 0.10
    base_height_floor_m = 0.19
    base_height_floor_reward_scale = -95.0
    undesired_contact_reward_scale = -2.25

    front_up_target_reward_scale = -1250.0
    front_down_hinge_reward_scale = -2300.0
    front_down_hinge_tolerance = -0.125
    front_up_posture_reward_scale = -1.4
    front_up_posture_command_reward_scale = -1.8
    front_up_posture_tolerance_rad = 0.38

    flat_orientation_reward_scale = -6.8
    yaw_drift_reward_scale = -0.78
    lateral_velocity_reward_scale = -0.48
    contact_balance_reward_scale = -0.36
    support_distribution_reward_scale = -0.24
    action_rate_reward_scale = -0.050


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV11PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV11EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV12EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV11EnvCfg
):
    # V12 fixes the observed BL-foot drag without a render-time/post-policy
    # correction. V11's late checkpoints bought extra pitch by keeping BL in
    # stance too long, so this branch makes rear contact timing and BL swing
    # clearance explicit while preserving the slow-cadence/front-up objective.
    lin_vel_y_range = (0.10, 0.13)
    lin_vel_reward_scale = 4.25
    command_progress_floor_reward_scale = -7.20
    min_command_progress_ratio = 0.72
    command_overspeed_reward_scale = -1.35

    contact_balance_reward_scale = -0.48
    rear_contact_antiphase_reward_scale = 0.55
    rear_contact_fraction_balance_reward_scale = -0.95
    rear_contact_fraction_balance_tolerance = 0.11
    rear_stance_duration_balance_reward_scale = -0.38
    rear_touchdown_rate_balance_reward_scale = -0.34
    rear_stride_balance_reward_scale = -0.30
    rear_stance_slip_balance_reward_scale = -0.48
    rear_left_stance_slip_excess_reward_scale = -0.42
    rear_symmetry_ready_time_s = 0.45

    low_swing_drag_reward_scale = -0.72
    bl_swing_height_reward_scale = 0.80
    bl_lifted_swing_min_height_m = 0.038
    bl_lifted_swing_height_tolerance_m = 0.018

    bl_swing_femur_floor_reward_scale = -0.20
    bl_swing_femur_command_floor_reward_scale = -0.26
    rear_swing_femur_floor_rad = -0.43
    rear_swing_femur_floor_tolerance_rad = 0.070

    rear_tibia_link_mirror_reward_scale = -0.20
    rear_tibia_segment_mirror_reward_scale = -0.28
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -0.34
    bl_lifted_swing_tibia_segment_outward_reward_scale = -0.38
    bl_lifted_swing_tibia_link_outward_vs_br_tolerance_m = 0.004
    bl_lifted_swing_tibia_segment_outward_target_m = 0.044
    bl_lifted_swing_tibia_segment_outward_tolerance_m = 0.008


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV12PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV12EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV13EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV12EnvCfg
):
    # V12 showed that a generic rear-contact balance penalty is not enough:
    # the policy simply pays it while keeping BL planted. V13 adds a direct
    # ceiling on BL contact fraction and a soft rear contact target, and backs
    # off the visual mirror terms that were large but not changing contact time.
    rear_contact_fraction_balance_reward_scale = -1.35
    rear_contact_fraction_balance_tolerance = 0.10
    rear_contact_fraction_target_reward_scale = -1.20
    rear_contact_fraction_target = 0.52
    rear_contact_fraction_target_tolerance = 0.12
    bl_contact_fraction_excess_reward_scale = -4.50
    bl_contact_fraction_ceiling = 0.53
    bl_contact_fraction_excess_tolerance = 0.060

    rear_stance_duration_balance_reward_scale = -0.45
    rear_touchdown_rate_balance_reward_scale = -0.42
    rear_stride_balance_reward_scale = -0.36
    rear_stance_slip_balance_reward_scale = -0.55
    rear_contact_antiphase_reward_scale = 0.70

    bl_swing_height_reward_scale = 1.15
    bl_swing_femur_floor_reward_scale = -0.34
    bl_swing_femur_command_floor_reward_scale = -0.40
    low_swing_drag_reward_scale = -0.82

    rear_tibia_link_mirror_reward_scale = 0.0
    rear_tibia_segment_mirror_reward_scale = 0.0
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -0.18
    bl_lifted_swing_tibia_segment_outward_reward_scale = -0.18

    lin_vel_reward_scale = 4.55
    command_progress_floor_reward_scale = -8.80
    min_command_progress_ratio = 0.74
    front_up_target_reward_scale = -1150.0
    front_down_hinge_reward_scale = -2050.0


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV13PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV13EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5EnvCfg
):
    # V14 restarts from clean model_1798 instead of the V11/V12/V13 branches.
    # The goal is a 10 deg front-up cat-style diagonal gait: keep the proven V5
    # rear support basin, push the body attitude upward, and reward diagonal
    # pair contact timing rather than direct BL contact suppression.
    lin_vel_y_range = (0.10, 0.13)
    lin_vel_tracking_sigma = 0.034
    lin_vel_reward_scale = 4.15
    command_progress_floor_reward_scale = -6.80
    min_command_progress_ratio = 0.70
    moving_command_threshold = 0.05
    command_overspeed_reward_scale = -1.35
    command_overspeed_tolerance = 0.050

    action_center_offsets = {
        "BL_femur_joint": -0.42,
        "BR_femur_joint": -0.42,
    }
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.42,
        "BR_femur_joint": -0.42,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.42,
        "BR_femur_joint": -0.42,
    }
    front_up_reset_root_z_offset = 0.105
    base_height_floor_m = 0.19
    base_height_floor_reward_scale = -90.0
    undesired_contact_reward_scale = -2.40

    front_up_target_reward_scale = -1425.0
    front_down_hinge_reward_scale = -2700.0
    front_down_hinge_tolerance = -0.115
    front_up_posture_reward_scale = -1.5
    front_up_posture_command_reward_scale = -1.9
    front_up_posture_tolerance_rad = 0.38

    flat_orientation_reward_scale = -6.3
    yaw_drift_reward_scale = -0.92
    lateral_velocity_reward_scale = -0.58
    contact_balance_reward_scale = -0.42
    support_distribution_reward_scale = -0.24
    action_rate_reward_scale = -0.052
    joint_accel_reward_scale = -1.4e-6

    diagonal_trot_reward_scale = 1.35
    diagonal_pair_sync_reward_scale = -0.85
    side_antiphase_reward_scale = 0.55
    rear_contact_antiphase_reward_scale = 0.75

    touchdown_cadence_target_hz = 0.95
    touchdown_cadence_reward_scale = -2.30
    rapid_touchdown_reward_scale = -2.80
    max_touchdown_rate_hz = 1.75
    min_stance_time = 0.34
    min_swing_time = 0.30
    short_stance_reward_scale = -4.60
    short_swing_reward_scale = -5.00

    touchdown_stride_reward_scale = 10.50
    touchdown_short_stride_reward_scale = -9.50
    min_touchdown_stride = 0.082
    touchdown_stride_target = 0.17
    stride_ema_shortfall_reward_scale = -0.75
    stride_ema_floor = 0.075

    stance_anchor_slip_reward_scale = -1.05
    low_swing_drag_reward_scale = -0.60
    swing_height_reward_scale = 1.05
    bl_swing_height_reward_scale = 0.28
    bl_lifted_swing_min_height_m = 0.034
    bl_lifted_swing_height_tolerance_m = 0.020

    rear_stride_balance_reward_scale = -0.42
    rear_stance_duration_balance_reward_scale = -0.24
    rear_touchdown_rate_balance_reward_scale = -0.26
    rear_contact_fraction_balance_reward_scale = -0.32
    rear_contact_fraction_balance_tolerance = 0.16
    rear_stance_slip_balance_reward_scale = -0.34
    rear_symmetry_ready_time_s = 0.55


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV15EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14EnvCfg
):
    # V14 reached a healthier front-up/high-body basin, but it got there with
    # short rapid steps: ~7.5 deg front-up, ~0.036 m stride, and ~2.5 Hz
    # per-foot touchdowns. V15 pushes the support posture slightly closer to
    # 10 deg while making the rapid-tapping solution much more expensive.
    lin_vel_y_range = (0.09, 0.115)
    lin_vel_tracking_sigma = 0.026
    lin_vel_reward_scale = 4.60
    command_progress_floor_reward_scale = -7.00
    min_command_progress_ratio = 0.66
    command_overspeed_reward_scale = -1.75
    command_overspeed_tolerance = 0.045

    action_center_offsets = {
        "BL_femur_joint": -0.46,
        "BR_femur_joint": -0.46,
    }
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.46,
        "BR_femur_joint": -0.46,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.46,
        "BR_femur_joint": -0.46,
    }
    front_up_reset_root_z_offset = 0.118
    base_height_floor_m = 0.205
    base_height_floor_reward_scale = -130.0
    undesired_contact_reward_scale = -3.00

    front_up_target_reward_scale = -2200.0
    front_down_hinge_reward_scale = -4400.0
    front_down_hinge_tolerance = -0.155
    front_up_posture_reward_scale = -2.0
    front_up_posture_command_reward_scale = -2.5
    front_up_posture_tolerance_rad = 0.34

    flat_orientation_reward_scale = -5.8
    yaw_drift_reward_scale = -1.05
    lateral_velocity_reward_scale = -0.75
    contact_balance_reward_scale = -0.50
    support_distribution_reward_scale = -0.28
    action_rate_reward_scale = -0.070
    joint_accel_reward_scale = -2.0e-6

    diagonal_trot_reward_scale = 0.55
    diagonal_pair_sync_reward_scale = -0.60
    side_antiphase_reward_scale = 0.30
    rear_contact_antiphase_reward_scale = 0.35

    feet_air_time_reward_scale = 0.45
    feet_air_time_target = 0.34
    touchdown_cadence_target_hz = 0.85
    touchdown_cadence_reward_scale = -6.00
    rapid_touchdown_reward_scale = -8.00
    max_touchdown_rate_hz = 1.35
    min_stance_time = 0.42
    min_swing_time = 0.36
    short_stance_reward_scale = -7.00
    short_swing_reward_scale = -8.00

    touchdown_stride_reward_scale = 16.00
    touchdown_short_stride_reward_scale = -22.00
    min_touchdown_stride = 0.105
    touchdown_stride_target = 0.18
    stride_ema_shortfall_reward_scale = -4.00
    stride_ema_floor = 0.095

    stance_anchor_slip_reward_scale = -1.35
    low_swing_drag_reward_scale = -0.70
    swing_height_reward_scale = 0.90
    bl_swing_height_reward_scale = 0.25
    bl_lifted_swing_min_height_m = 0.035
    bl_lifted_swing_height_tolerance_m = 0.020

    rear_stride_balance_reward_scale = -0.55
    rear_stance_duration_balance_reward_scale = -0.32
    rear_touchdown_rate_balance_reward_scale = -0.36
    rear_contact_fraction_balance_reward_scale = -0.38
    rear_contact_fraction_balance_tolerance = 0.16
    rear_stance_slip_balance_reward_scale = -0.42
    rear_symmetry_ready_time_s = 0.55


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV15PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV15EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV16EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14EnvCfg
):
    # V15 over-tightened the 10 deg/cadence/height objective and collapsed into
    # a low short-step gait. V16 keeps V14's cleaner walking basin from
    # model_1798, adds progress-gated pitch pressure, and uses forgiving
    # cadence/stride targets so forward locomotion stays the primary solution.
    lin_vel_y_range = (0.115, 0.145)
    lin_vel_tracking_sigma = 0.040
    lin_vel_reward_scale = 5.00
    command_progress_floor_reward_scale = -8.20
    min_command_progress_ratio = 0.76
    command_overspeed_reward_scale = -1.20
    command_overspeed_tolerance = 0.065

    action_center_offsets = {
        "BL_femur_joint": -0.44,
        "BR_femur_joint": -0.44,
    }
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.44,
        "BR_femur_joint": -0.44,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.44,
        "BR_femur_joint": -0.44,
    }
    front_up_reset_root_z_offset = 0.110
    base_height_floor_m = 0.190
    base_height_floor_reward_scale = -80.0
    undesired_contact_reward_scale = -2.25

    front_up_target_reward_scale = -1650.0
    front_down_hinge_reward_scale = -3200.0
    front_down_hinge_tolerance = -0.135
    front_up_progress_gate_ratio = 0.62
    front_up_progress_gate_min_weight = 0.35
    front_up_posture_reward_scale = -1.55
    front_up_posture_command_reward_scale = -1.95
    front_up_posture_tolerance_rad = 0.38

    flat_orientation_reward_scale = -5.8
    yaw_drift_reward_scale = -0.95
    lateral_velocity_reward_scale = -0.60
    contact_balance_reward_scale = -0.38
    support_distribution_reward_scale = -0.22
    action_rate_reward_scale = -0.056
    joint_accel_reward_scale = -1.4e-6

    diagonal_trot_reward_scale = 1.15
    diagonal_pair_sync_reward_scale = -0.75
    side_antiphase_reward_scale = 0.48
    rear_contact_antiphase_reward_scale = 0.62

    feet_air_time_reward_scale = 0.45
    feet_air_time_target = 0.28
    touchdown_cadence_target_hz = 1.45
    touchdown_cadence_reward_scale = -2.60
    rapid_touchdown_reward_scale = -3.40
    max_touchdown_rate_hz = 2.45
    min_stance_time = 0.24
    min_swing_time = 0.20
    short_stance_reward_scale = -3.70
    short_swing_reward_scale = -4.00

    touchdown_stride_reward_scale = 12.00
    touchdown_short_stride_reward_scale = -8.00
    min_touchdown_stride = 0.060
    touchdown_stride_target = 0.135
    stride_ema_shortfall_reward_scale = -0.90
    stride_ema_floor = 0.060

    stance_anchor_slip_reward_scale = -1.00
    low_swing_drag_reward_scale = -0.62
    swing_height_reward_scale = 1.00
    bl_swing_height_reward_scale = 0.30
    bl_lifted_swing_min_height_m = 0.034
    bl_lifted_swing_height_tolerance_m = 0.020

    rear_stride_balance_reward_scale = -0.40
    rear_stance_duration_balance_reward_scale = -0.24
    rear_touchdown_rate_balance_reward_scale = -0.26
    rear_contact_fraction_balance_reward_scale = -0.30
    rear_contact_fraction_balance_tolerance = 0.17
    rear_stance_slip_balance_reward_scale = -0.34
    rear_symmetry_ready_time_s = 0.55


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV16PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV16EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV17EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5EnvCfg
):
    # V17 restarts from clean model_1798 and keeps the successful V5 support
    # basin, but changes the objective shape: locomotion is rewarded strongly,
    # 10 deg front-up is encouraged with a capped progress-gated bonus, and the
    # pitch penalties stay mild enough that the policy should not crouch/stop to
    # reduce loss. This is still a 10 deg target, not an 8 deg acquisition task.
    lin_vel_y_range = (0.125, 0.165)
    lin_vel_tracking_sigma = 0.045
    lin_vel_reward_scale = 5.35
    command_progress_floor_reward_scale = -9.25
    min_command_progress_ratio = 0.74
    moving_command_threshold = 0.05
    command_overspeed_reward_scale = -0.85
    command_overspeed_tolerance = 0.080

    action_center_offsets = {
        "BL_femur_joint": -0.40,
        "BR_femur_joint": -0.40,
    }
    reset_joint_pos_offsets = {
        "BL_femur_joint": -0.40,
        "BR_femur_joint": -0.40,
    }
    front_up_posture_joint_offsets = {
        "BL_femur_joint": -0.40,
        "BR_femur_joint": -0.40,
    }
    front_up_reset_root_z_offset = 0.095
    base_height_floor_m = 0.185
    base_height_floor_reward_scale = -55.0
    undesired_contact_reward_scale = -2.05

    front_up_target_reward_scale = -620.0
    front_down_hinge_reward_scale = -1050.0
    front_down_hinge_tolerance = -0.095
    front_up_progress_gate_ratio = 0.66
    front_up_progress_gate_min_weight = 0.18
    front_up_progress_bonus_reward_scale = 2.85
    front_up_progress_bonus_start_projected_gravity_y = -math.sin(math.radians(5.0))
    front_up_posture_reward_scale = -1.15
    front_up_posture_command_reward_scale = -1.35
    front_up_posture_tolerance_rad = 0.42

    flat_orientation_reward_scale = -5.25
    yaw_drift_reward_scale = -0.78
    lateral_velocity_reward_scale = -0.48
    contact_balance_reward_scale = -0.24
    support_distribution_reward_scale = -0.18
    action_rate_reward_scale = -0.044
    joint_accel_reward_scale = -1.0e-6

    diagonal_trot_reward_scale = 1.05
    diagonal_pair_sync_reward_scale = -0.62
    side_antiphase_reward_scale = 0.45
    rear_contact_antiphase_reward_scale = 0.58

    feet_air_time_reward_scale = 0.42
    feet_air_time_target = 0.26
    touchdown_cadence_target_hz = 1.75
    touchdown_cadence_reward_scale = -1.30
    rapid_touchdown_reward_scale = -1.75
    max_touchdown_rate_hz = 3.20
    min_stance_time = 0.20
    min_swing_time = 0.16
    short_stance_reward_scale = -2.00
    short_swing_reward_scale = -2.30

    touchdown_stride_reward_scale = 10.50
    touchdown_short_stride_reward_scale = -5.25
    min_touchdown_stride = 0.052
    touchdown_stride_target = 0.12
    stride_ema_shortfall_reward_scale = -0.32
    stride_ema_floor = 0.050

    stance_anchor_slip_reward_scale = -0.85
    low_swing_drag_reward_scale = -0.55
    swing_height_reward_scale = 1.00
    bl_swing_height_reward_scale = 0.28
    bl_lifted_swing_min_height_m = 0.033
    bl_lifted_swing_height_tolerance_m = 0.020

    rear_stride_balance_reward_scale = -0.34
    rear_stance_duration_balance_reward_scale = -0.20
    rear_touchdown_rate_balance_reward_scale = -0.22
    rear_contact_fraction_balance_reward_scale = -0.24
    rear_contact_fraction_balance_tolerance = 0.18
    rear_stance_slip_balance_reward_scale = -0.28
    rear_symmetry_ready_time_s = 0.55


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV17PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV17EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV18EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV17EnvCfg
):
    # V18 is a cleanup continuation from the near-10-degree V17 checkpoint.
    # V17 reached the target pitch, but eval showed yaw drift and rear-contact
    # imbalance. Keep the same pitch/velocity basin and tighten the gait quality
    # terms instead of re-acquiring the posture from scratch.
    lin_vel_y_range = (0.135, 0.165)
    lin_vel_tracking_sigma = 0.040
    lin_vel_reward_scale = 5.60
    command_progress_floor_reward_scale = -9.75
    min_command_progress_ratio = 0.78
    command_overspeed_reward_scale = -0.95
    command_overspeed_tolerance = 0.070

    front_up_target_reward_scale = -700.0
    front_down_hinge_reward_scale = -1125.0
    front_up_progress_gate_ratio = 0.70
    front_up_progress_gate_min_weight = 0.16
    front_up_progress_bonus_reward_scale = 2.55

    yaw_drift_reward_scale = -1.55
    yaw_drift_tolerance_rad = 0.105
    lateral_velocity_reward_scale = -0.80
    lateral_position_reward_scale = -0.35
    lateral_position_tolerance_m = 0.09
    flat_orientation_reward_scale = -6.00

    contact_balance_reward_scale = -0.32
    support_distribution_reward_scale = -0.22
    diagonal_trot_reward_scale = 1.00
    diagonal_pair_sync_reward_scale = -0.74
    side_antiphase_reward_scale = 0.52
    rear_contact_antiphase_reward_scale = 0.70

    touchdown_cadence_target_hz = 1.55
    touchdown_cadence_reward_scale = -1.75
    rapid_touchdown_reward_scale = -2.20
    max_touchdown_rate_hz = 3.05
    short_stance_reward_scale = -2.20
    short_swing_reward_scale = -2.55

    touchdown_stride_reward_scale = 10.00
    touchdown_short_stride_reward_scale = -5.75
    min_touchdown_stride = 0.055
    stride_ema_shortfall_reward_scale = -0.42
    stride_ema_floor = 0.052

    stance_anchor_slip_reward_scale = -1.15
    low_swing_drag_reward_scale = -0.70
    swing_height_reward_scale = 0.95
    bl_swing_height_reward_scale = 0.34

    rear_stride_balance_reward_scale = -0.55
    rear_stance_duration_balance_reward_scale = -0.32
    rear_touchdown_rate_balance_reward_scale = -0.46
    rear_contact_fraction_balance_reward_scale = -0.52
    rear_contact_fraction_balance_tolerance = 0.13
    rear_contact_fraction_target_reward_scale = -0.24
    rear_contact_fraction_target = 0.50
    rear_contact_fraction_target_tolerance = 0.14
    rear_stance_slip_balance_reward_scale = -0.50

    # Light visual guards only; strong tibia equalization previously fought the
    # walking cycle. These are small enough to clean the rear posture without
    # overriding the learned gait.
    rear_tibia_link_mirror_reward_scale = -0.12
    rear_tibia_segment_mirror_reward_scale = -0.16
    rear_tibia_link_mirror_tolerance_m = 0.035
    rear_tibia_segment_mirror_tolerance_m = 0.026


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV18PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV18EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV19EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV18EnvCfg
):
    # V19 directly addresses the remaining visible BL scrape from V18. The
    # previous cleanup improved yaw/posture, but BL still traveled too quickly
    # while below the clearance band. Keep V18's front-up basin and add a
    # foot-specific low-swing penalty plus a stronger BL swing-clearance signal.
    low_swing_drag_reward_scale = -0.95
    bl_low_swing_drag_reward_scale = -2.80
    swing_height_reward_scale = 1.05
    bl_swing_height_reward_scale = 1.20
    swing_height_min = 0.032
    swing_height_target = 0.082
    bl_lifted_swing_min_height_m = 0.040
    bl_lifted_swing_height_tolerance_m = 0.016

    min_stance_time = 0.18
    min_swing_time = 0.18
    short_stance_reward_scale = -2.70
    short_swing_reward_scale = -3.30
    touchdown_cadence_target_hz = 1.50
    touchdown_cadence_reward_scale = -1.95
    rapid_touchdown_reward_scale = -2.55
    max_touchdown_rate_hz = 2.90

    rear_stride_balance_reward_scale = -0.68
    rear_stance_duration_balance_reward_scale = -0.54
    rear_touchdown_rate_balance_reward_scale = -0.72
    rear_contact_fraction_balance_reward_scale = -0.78
    rear_contact_fraction_balance_tolerance = 0.11
    rear_contact_fraction_target_reward_scale = -0.34
    rear_contact_fraction_target = 0.50
    rear_contact_fraction_target_tolerance = 0.12
    rear_stance_slip_balance_reward_scale = -0.56

    bl_swing_femur_floor_reward_scale = -0.16
    bl_swing_femur_command_floor_reward_scale = -0.20
    rear_swing_femur_floor_rad = -0.43
    rear_swing_femur_floor_tolerance_rad = 0.075

    rear_tibia_link_mirror_reward_scale = -0.10
    rear_tibia_segment_mirror_reward_scale = -0.12
    rear_tibia_link_mirror_tolerance_m = 0.034
    rear_tibia_segment_mirror_tolerance_m = 0.026
    bl_lifted_swing_tibia_segment_outward_reward_scale = -0.16
    bl_lifted_swing_tibia_segment_outward_target_m = 0.044
    bl_lifted_swing_tibia_segment_outward_tolerance_m = 0.008

    yaw_drift_reward_scale = -1.70
    yaw_drift_tolerance_rad = 0.100
    lateral_velocity_reward_scale = -0.85


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV19PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV19EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV20EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV19EnvCfg
):
    # V19 made yaw/contact balance better but BL still skimmed through swing.
    # This branch adds a direct penalty for BL time spent below the clearance
    # band, not just a speed penalty when it is already scraping.
    low_swing_clearance_m = 0.035
    low_swing_drag_reward_scale = -1.10
    bl_low_swing_drag_reward_scale = -3.60
    bl_low_swing_time_reward_scale = -1.55
    swing_height_reward_scale = 1.15
    bl_swing_height_reward_scale = 1.75
    swing_height_min = 0.035
    swing_height_target = 0.092
    bl_lifted_swing_min_height_m = 0.045
    bl_lifted_swing_height_tolerance_m = 0.014

    base_height_floor_m = 0.200
    base_height_floor_reward_scale = -78.0

    feet_air_time_reward_scale = 0.55
    feet_air_time_target = 0.28
    min_stance_time = 0.17
    min_swing_time = 0.20
    short_stance_reward_scale = -2.80
    short_swing_reward_scale = -3.75
    touchdown_cadence_target_hz = 1.42
    touchdown_cadence_reward_scale = -2.15
    rapid_touchdown_reward_scale = -2.95
    max_touchdown_rate_hz = 2.65

    rear_contact_fraction_balance_reward_scale = -0.90
    rear_touchdown_rate_balance_reward_scale = -0.86
    rear_stance_duration_balance_reward_scale = -0.60
    rear_stride_balance_reward_scale = -0.72
    rear_contact_fraction_balance_tolerance = 0.10
    rear_stance_slip_balance_reward_scale = -0.60

    bl_swing_femur_floor_reward_scale = -0.22
    bl_swing_femur_command_floor_reward_scale = -0.26
    rear_swing_femur_floor_tolerance_rad = 0.070

    yaw_drift_reward_scale = -1.75
    lateral_velocity_reward_scale = -0.90


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV20PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV20EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV20EnvCfg
):
    # V20 still lets BL skim through swing. This branch gives BL a phase-delayed
    # rear-right foot-height reference, so the learned policy must clear the
    # floor where BR already demonstrates clean clearance.
    low_swing_clearance_m = 0.037
    bl_low_swing_drag_reward_scale = -3.40
    bl_low_swing_time_reward_scale = -1.35
    swing_height_reward_scale = 1.18
    bl_swing_height_reward_scale = 1.95
    swing_height_min = 0.036
    swing_height_target = 0.094
    bl_lifted_swing_min_height_m = 0.046
    bl_lifted_swing_height_tolerance_m = 0.013

    bl_phase_delayed_foot_height_reward_scale = -2.60
    bl_phase_delayed_foot_height_tolerance_m = 0.012
    bl_phase_delayed_foot_height_min_m = 0.046

    bl_phase_delayed_rear_coxa_joint_symmetry_reward_scale = -0.14
    bl_phase_delayed_rear_coxa_command_symmetry_reward_scale = -0.10
    bl_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.25
    bl_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.18

    rear_contact_fraction_balance_reward_scale = -0.94
    rear_touchdown_rate_balance_reward_scale = -0.90
    rear_stance_duration_balance_reward_scale = -0.64
    rear_stride_balance_reward_scale = -0.76
    rear_stance_slip_balance_reward_scale = -0.64

    yaw_drift_reward_scale = -1.80
    lateral_velocity_reward_scale = -0.92


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV22EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21EnvCfg
):
    # V21 fixed BL floor drag, but the rear-left tibia still curls inward more
    # than BR. Keep the clearance rewards and add direct, signed tibia curl caps.
    bl_tibia_tuck_reward_scale = -1.15
    bl_tibia_tuck_limit_rad = 2.18
    bl_tibia_tuck_tolerance_rad = 0.16
    bl_tibia_pair_offset_reward_scale = -1.85
    bl_tibia_command_pair_offset_reward_scale = -1.25
    bl_tibia_pair_offset_limit_rad = 0.34
    bl_tibia_pair_offset_tolerance_rad = 0.14

    rear_tibia_mean_target_reward_scale = -0.22
    rear_tibia_mean_target_rad = 1.95
    rear_tibia_mean_target_tolerance_rad = 0.25

    rear_tibia_link_mirror_reward_scale = -0.16
    rear_tibia_segment_mirror_reward_scale = -0.22
    rear_tibia_link_mirror_tolerance_m = 0.030
    rear_tibia_segment_mirror_tolerance_m = 0.022

    bl_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.34
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.28
    bl_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.24
    bl_lifted_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.20

    bl_low_swing_drag_reward_scale = -3.55
    bl_low_swing_time_reward_scale = -1.42
    bl_phase_delayed_foot_height_reward_scale = -2.45
    yaw_drift_reward_scale = -1.86
    lateral_velocity_reward_scale = -0.94


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV22PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV22EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21EnvCfg
):
    # V22 proved that the tibia-offset term can make the rear legs symmetric by
    # curling BR inward. This branch returns to V21 and instead caps BL curl
    # directly while keeping both rear tibias inside a shared tuck envelope.
    bl_tibia_tuck_reward_scale = -3.00
    bl_tibia_tuck_limit_rad = 2.18
    bl_tibia_tuck_tolerance_rad = 0.12

    rear_tibia_tuck_envelope_reward_scale = -1.35
    rear_tibia_tuck_limit_rad = 2.20
    rear_tibia_tuck_tolerance_rad = 0.18

    rear_tibia_mean_target_reward_scale = -1.20
    rear_tibia_command_target_reward_scale = -0.80
    rear_tibia_mean_target_rad = 1.95
    rear_tibia_mean_target_tolerance_rad = 0.22
    rear_tibia_command_target_tolerance_rad = 0.22

    bl_tibia_pair_offset_reward_scale = -0.35
    bl_tibia_command_pair_offset_reward_scale = -0.25
    bl_tibia_pair_offset_limit_rad = 0.42
    bl_tibia_pair_offset_tolerance_rad = 0.16

    rear_tibia_link_mirror_reward_scale = -0.12
    rear_tibia_segment_mirror_reward_scale = -0.18
    rear_tibia_link_mirror_tolerance_m = 0.033
    rear_tibia_segment_mirror_tolerance_m = 0.025

    bl_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.26
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.22
    bl_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.19
    bl_lifted_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.16

    bl_low_swing_drag_reward_scale = -3.45
    bl_low_swing_time_reward_scale = -1.36
    bl_phase_delayed_foot_height_reward_scale = -2.50
    yaw_drift_reward_scale = -1.90
    lateral_velocity_reward_scale = -0.96


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23NoVelDeployEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23EnvCfg
):
    # Real-robot deployment variant for the accepted V23 front-up gait. The
    # actor keeps the same 60D observation layout for checkpoint compatibility,
    # but obs[0:3] is always zero because the hardware does not measure torso
    # linear velocity. Rewards may still use privileged simulator velocity.
    use_privileged_base_lin_vel_obs = False
    use_foot_kinematic_base_lin_vel_obs = False
    base_lin_vel_obs_noise_std = 0.0

    # IMU/joint robustness. The deployment adapter can compute projected gravity
    # from roll/pitch/yaw; angular velocity can come from gyro or filtered RPY
    # finite differences. These noises force adaptation away from clean sim
    # attitude and motor state.
    observation_noise_enabled = True
    angular_velocity_obs_noise_std = 0.025
    projected_gravity_obs_noise_std = 0.015
    joint_position_obs_noise_std = 0.008
    joint_velocity_obs_noise_std = 0.050
    action_obs_noise_std = 0.004
    contact_obs_dropout_prob = 0.10
    timer_obs_noise_std = 0.020


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23NoVelDeployPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23NoVelDeployEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeployEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23NoVelDeployEnvCfg
):
    # Strict torso-sensing deployment variant. The actor receives no measured or
    # estimated body linear velocity and no body angular velocity. The old
    # angular-velocity observation block, obs[3:6], is replaced with relative
    # roll, pitch, yaw in radians so the torso sensor contract is attitude-only.
    use_rpy_attitude_obs = True
    ang_vel_z_range = (0.0, 0.0)
    angular_velocity_obs_noise_std = 0.0
    rpy_attitude_obs_noise_std = 0.012
    projected_gravity_obs_noise_std = 0.018
    joint_position_obs_noise_std = 0.010
    joint_velocity_obs_noise_std = 0.060
    action_obs_noise_std = 0.004
    contact_obs_dropout_prob = 0.12
    timer_obs_noise_std = 0.025

    # The real robot should not need torso velocity sensing, but the simulator can
    # still use privileged angular velocity in the reward to train a quieter body.
    yaw_rate_reward_scale = 4.80
    yaw_drift_reward_scale = -16.00
    yaw_drift_tolerance_rad = 0.026
    ang_vel_reward_scale = -0.75
    ang_vel_z_reward_scale = -8.00
    lateral_velocity_reward_scale = -2.35
    lateral_position_reward_scale = -3.20
    lateral_position_tolerance_m = 0.050


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeployPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeployEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeployEnvCfg
):
    # Continuation branch from the accepted RPY-only deploy checkpoint. The
    # policy/observation contract stays identical to V23 RPY deploy; only the
    # rewards are tightened to reduce torso shake, foot slip, and contact chatter.
    action_target_filter_alpha = 0.20

    command_overspeed_reward_scale = -3.25
    command_overspeed_tolerance = 0.025

    z_vel_reward_scale = -5.00
    ang_vel_reward_scale = -1.15
    ang_vel_z_reward_scale = -9.50
    yaw_rate_reward_scale = 5.20
    yaw_drift_reward_scale = -18.00
    yaw_drift_tolerance_rad = 0.022
    lateral_velocity_reward_scale = -2.55
    lateral_position_reward_scale = -3.40
    lateral_position_tolerance_m = 0.045
    flat_orientation_reward_scale = -8.50

    joint_accel_reward_scale = -2.4e-6
    action_rate_reward_scale = -0.105

    stance_foot_slip_reward_scale = -1.60
    stance_anchor_slip_reward_scale = -1.75
    stance_anchor_slip_tolerance_m = 0.010
    low_swing_clearance_m = 0.040
    low_swing_drag_reward_scale = -1.30
    bl_low_swing_drag_reward_scale = -3.75
    bl_low_swing_time_reward_scale = -1.50

    support_distribution_reward_scale = -0.40
    contact_balance_reward_scale = -0.62
    diagonal_trot_reward_scale = 0.18
    diagonal_pair_sync_reward_scale = -0.12
    rear_contact_antiphase_reward_scale = 0.78
    touchdown_cadence_reward_scale = -2.40
    rapid_touchdown_reward_scale = -3.90
    max_touchdown_rate_hz = 1.50
    short_stance_reward_scale = -3.20
    short_swing_reward_scale = -3.40
    min_stance_time = 0.34
    min_swing_time = 0.28

    rear_contact_fraction_balance_reward_scale = -1.20
    rear_contact_fraction_balance_tolerance = 0.085
    rear_contact_fraction_target_reward_scale = -0.42
    rear_contact_fraction_target_tolerance = 0.11
    rear_touchdown_rate_balance_reward_scale = -1.05
    rear_stance_duration_balance_reward_scale = -0.80
    rear_stance_slip_balance_reward_scale = -0.85
    rear_stance_slip_balance_tolerance_mps = 0.060


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactV2EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactEnvCfg
):
    # V1 improved stance slip early, but its cadence target and rapid-tap ceiling
    # conflicted. V2 aligns those terms and focuses on cleaner swing clearance.
    action_target_filter_alpha = 0.18

    low_swing_clearance_m = 0.046
    low_swing_drag_reward_scale = -1.65
    bl_low_swing_drag_reward_scale = -4.25
    bl_low_swing_time_reward_scale = -1.85
    swing_height_reward_scale = 1.28
    bl_swing_height_reward_scale = 2.20
    swing_height_target = 0.102

    touchdown_cadence_target_hz = 2.35
    touchdown_cadence_reward_scale = -1.20
    max_touchdown_rate_hz = 2.85
    rapid_touchdown_reward_scale = -1.65
    touchdown_stride_reward_scale = 10.75
    touchdown_short_stride_reward_scale = -6.00

    z_vel_reward_scale = -5.40
    ang_vel_reward_scale = -1.30
    ang_vel_z_reward_scale = -10.20
    yaw_rate_reward_scale = 5.40
    yaw_drift_reward_scale = -19.50
    lateral_velocity_reward_scale = -2.75

    stance_foot_slip_reward_scale = -1.80
    stance_anchor_slip_reward_scale = -1.95
    rear_stance_slip_balance_reward_scale = -0.95
    contact_balance_reward_scale = -0.68
    rear_touchdown_rate_balance_reward_scale = -1.12


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactV2PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactV2EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactEnvCfg
):
    # Continuation from the best measured smooth-contact checkpoint. Focus on
    # reducing side-to-side torso motion without reintroducing the V2 tapping
    # and low-clearance regression.
    action_target_filter_alpha = 0.18

    lateral_velocity_reward_scale = -4.20
    lateral_position_reward_scale = -5.20
    lateral_position_tolerance_m = 0.030
    flat_orientation_reward_scale = -13.50
    flat_orientation_pitch_weight = 0.0
    ang_vel_reward_scale = -1.75
    ang_vel_z_reward_scale = -10.80
    yaw_drift_reward_scale = -20.00
    yaw_rate_reward_scale = 5.35

    contact_balance_reward_scale = -0.86
    touchdown_rate_balance_reward_scale = -0.42
    rear_contact_fraction_balance_reward_scale = -1.42
    rear_touchdown_rate_balance_reward_scale = -1.18
    rear_stance_duration_balance_reward_scale = -0.90
    rear_stance_slip_balance_reward_scale = -0.92

    stance_foot_slip_reward_scale = -1.65
    stance_anchor_slip_reward_scale = -1.85
    low_swing_drag_reward_scale = -1.35
    bl_low_swing_drag_reward_scale = -3.85
    bl_low_swing_time_reward_scale = -1.55

    touchdown_cadence_target_hz = 2.55
    touchdown_cadence_reward_scale = -1.10
    max_touchdown_rate_hz = 3.05
    rapid_touchdown_reward_scale = -1.40
    touchdown_stride_reward_scale = 10.50
    touchdown_short_stride_reward_scale = -5.40


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayEnvCfg
):
    # Real-hardware adaptation branch for the anti-sway checkpoint. Keep the
    # 60D actor layout checkpoint-compatible, but make unavailable sensors
    # uninformative: no torso linear velocity, no foot-contact bits, and no
    # contact-derived timing. The policy must lean on command, IMU attitude, joint
    # state, and action history, matching the current robot deployment contract.
    use_privileged_base_lin_vel_obs = False
    use_foot_kinematic_base_lin_vel_obs = False
    base_lin_vel_obs_noise_std = 0.0
    use_rpy_attitude_obs = True
    observation_noise_enabled = True

    # IMU and encoder noise. RPY is about 1.1 deg std; projected gravity and
    # joint channels are intentionally a little noisy to improve sim-to-real
    # tolerance without forcing the gait to relearn from scratch.
    rpy_attitude_obs_noise_std = 0.020
    projected_gravity_obs_noise_std = 0.025
    joint_position_obs_noise_std = 0.015
    joint_velocity_obs_noise_std = 0.080
    action_obs_noise_std = 0.006
    contact_obs_dropout_prob = 1.0
    timer_obs_noise_std = 0.0
    zero_contact_obs = True
    zero_contact_timer_obs = True

    # Slightly relax the drift terms under noisier observations, while keeping
    # the anti-sway body-shape rewards that made model_3000 useful.
    lateral_position_tolerance_m = 0.040
    yaw_drift_tolerance_rad = 0.030
    action_target_filter_alpha = 0.20


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPhaseEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotEnvCfg
):
    # Strict no-contact observations made the policy recover by tapping too
    # quickly. This variant remains real-robot deployable by replacing the
    # contact/timer channels with a controller-owned diagonal-pair phase clock.
    # No measured contact or torso velocity is required on hardware.
    zero_contact_obs = False
    zero_contact_timer_obs = False
    contact_obs_dropout_prob = 0.0
    use_open_loop_gait_phase_obs = True
    open_loop_gait_period_s = 0.68
    open_loop_gait_duty_factor = 0.60

    # Keep robustness noise, but slightly reduce joint-velocity noise because
    # the phase clock already handles the largest observation shift.
    joint_velocity_obs_noise_std = 0.060
    action_target_filter_alpha = 0.20


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPhasePlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPhaseEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV24EnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21EnvCfg
):
    # V23 only partially reduced BL tibia curl. V24 adds direct visual geometry
    # guards for the BL tibia link/segment so the learned gait must keep the
    # lifted rear-left tibia outward, not just numerically closer to BR.
    bl_tibia_tuck_reward_scale = -6.50
    bl_tibia_tuck_limit_rad = 2.08
    bl_tibia_tuck_tolerance_rad = 0.14

    rear_tibia_tuck_envelope_reward_scale = -1.75
    rear_tibia_tuck_limit_rad = 2.18
    rear_tibia_tuck_tolerance_rad = 0.18

    rear_tibia_mean_target_reward_scale = -0.90
    rear_tibia_command_target_reward_scale = -0.65
    rear_tibia_mean_target_rad = 1.95
    rear_tibia_mean_target_tolerance_rad = 0.24
    rear_tibia_command_target_tolerance_rad = 0.24

    bl_tibia_pair_offset_reward_scale = -0.20
    bl_tibia_command_pair_offset_reward_scale = -0.15
    bl_tibia_pair_offset_limit_rad = 0.48
    bl_tibia_pair_offset_tolerance_rad = 0.18

    bl_tibia_link_outward_vs_br_reward_scale = -0.65
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -0.55
    bl_tibia_link_outward_floor_reward_scale = -0.18
    bl_lifted_swing_tibia_link_outward_floor_reward_scale = -0.50
    bl_tibia_link_outward_floor_target_m = 0.294
    bl_tibia_link_outward_floor_tolerance_m = 0.014
    bl_lifted_swing_tibia_link_outward_floor_target_m = 0.302
    bl_lifted_swing_tibia_link_outward_floor_tolerance_m = 0.010

    bl_lifted_swing_tibia_segment_outward_reward_scale = -0.70
    bl_lifted_swing_tibia_segment_outward_target_m = 0.047
    bl_lifted_swing_tibia_segment_outward_tolerance_m = 0.007

    rear_tibia_link_mirror_reward_scale = -0.10
    rear_tibia_segment_mirror_reward_scale = -0.16
    rear_tibia_link_mirror_tolerance_m = 0.034
    rear_tibia_segment_mirror_tolerance_m = 0.026

    bl_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.22
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.18
    bl_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.16
    bl_lifted_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.14

    bl_low_swing_drag_reward_scale = -3.45
    bl_low_swing_time_reward_scale = -1.36
    bl_phase_delayed_foot_height_reward_scale = -2.50
    yaw_drift_reward_scale = -1.92
    lateral_velocity_reward_scale = -0.97


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV24PlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV24EnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpRearMirrorEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquireEnvCfg
):
    # Continue from the accepted slow/front-up gait, but make the rear-left leg
    # visually match the rear-right leg instead of adding a render-time fix.
    rear_tibia_mean_symmetry_reward_scale = -2.00
    rear_tibia_command_symmetry_reward_scale = -1.00
    rear_coxa_joint_symmetry_reward_scale = -0.18
    rear_femur_joint_symmetry_reward_scale = -0.18
    rear_tibia_joint_symmetry_reward_scale = -0.35
    rear_femur_tibia_segment_mirror_reward_scale = -0.45
    rear_tibia_link_mirror_reward_scale = -0.55
    rear_tibia_segment_mirror_reward_scale = -0.60

    # Match the same phase of the cleaner rear-right motion during BL swing
    # instead of forcing both rear legs to be identical while one is planted.
    bl_phase_delayed_rear_coxa_joint_symmetry_reward_scale = -0.25
    bl_lifted_phase_delayed_rear_coxa_joint_symmetry_reward_scale = -0.35
    bl_phase_delayed_rear_coxa_command_symmetry_reward_scale = -0.18
    bl_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.35
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.45
    bl_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.25

    rear_stride_balance_reward_scale = -0.25
    rear_stance_duration_balance_reward_scale = -0.18
    rear_touchdown_rate_balance_reward_scale = -0.18
    rear_contact_fraction_balance_reward_scale = -0.12
    rear_stance_slip_balance_reward_scale = -0.25
    rear_symmetry_ready_time_s = 0.6

    rear_tibia_mean_symmetry_tolerance_rad = 0.12
    rear_tibia_command_symmetry_tolerance_rad = 0.12
    rear_coxa_joint_symmetry_tolerance_rad = 0.10
    rear_femur_joint_symmetry_tolerance_rad = 0.09
    rear_tibia_joint_symmetry_tolerance_rad = 0.09
    rear_femur_tibia_segment_mirror_tolerance_m = 0.018
    rear_tibia_link_mirror_tolerance_m = 0.024
    rear_tibia_segment_mirror_tolerance_m = 0.018


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpRearMirrorPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpRearMirrorEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPhaseMirrorEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquireEnvCfg
):
    # The rear legs are phase-shifted in the accepted gait, so same-time BL/BR
    # equality fights the locomotion cycle. Train BL to match the cleaner BR
    # geometry at the corresponding delayed swing phase instead.
    bl_phase_delayed_tibia_link_mirror_reward_scale = -0.45
    bl_lifted_phase_delayed_tibia_link_mirror_reward_scale = -0.65
    bl_phase_delayed_femur_tibia_mirror_reward_scale = -0.35
    bl_lifted_phase_delayed_femur_tibia_mirror_reward_scale = -0.50
    bl_phase_delayed_tibia_primary_axis_mirror_reward_scale = -0.20
    bl_lifted_phase_delayed_tibia_primary_axis_mirror_reward_scale = -0.30
    bl_phase_delayed_tibia_secondary_axis_mirror_reward_scale = -0.18
    bl_lifted_phase_delayed_tibia_secondary_axis_mirror_reward_scale = -0.26

    bl_phase_delayed_tibia_outward_reward_scale = -0.18
    bl_phase_delayed_tibia_link_outward_reward_scale = -0.24
    bl_lifted_phase_delayed_tibia_link_outward_reward_scale = -0.35
    bl_phase_delayed_rear_coxa_joint_symmetry_reward_scale = -0.10
    bl_lifted_phase_delayed_rear_coxa_joint_symmetry_reward_scale = -0.16
    bl_phase_delayed_rear_coxa_command_symmetry_reward_scale = -0.08
    bl_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.16
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.22
    bl_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.12

    # Keep only light timing/contact balance, not posture equality.
    rear_stride_balance_reward_scale = -0.08
    rear_stance_duration_balance_reward_scale = -0.06
    rear_touchdown_rate_balance_reward_scale = -0.06
    rear_contact_fraction_balance_reward_scale = -0.04
    rear_stance_slip_balance_reward_scale = -0.08
    rear_symmetry_ready_time_s = 0.6

    bl_phase_delayed_tibia_link_mirror_tolerance_m = 0.014
    bl_lifted_phase_delayed_tibia_link_mirror_tolerance_m = 0.012
    bl_phase_delayed_femur_tibia_mirror_tolerance_m = 0.014
    bl_lifted_phase_delayed_femur_tibia_mirror_tolerance_m = 0.012
    bl_phase_delayed_tibia_link_outward_tolerance_m = 0.006
    bl_lifted_phase_delayed_tibia_link_outward_tolerance_m = 0.005
    bl_phase_delayed_rear_coxa_joint_symmetry_tolerance_rad = 0.08
    bl_lifted_phase_delayed_rear_coxa_joint_symmetry_tolerance_rad = 0.07
    bl_phase_delayed_rear_distal_joint_symmetry_tolerance_rad = 0.12
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_tolerance_rad = 0.10
    rear_phase_symmetry_delay_steps = 12


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPhaseMirrorPlayEnvCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPhaseMirrorEnvCfg
):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineDeployEnvCfg(InsectoidMiniQuadGaitRefineEnvCfg):
    use_privileged_base_lin_vel_obs = False
    use_foot_kinematic_base_lin_vel_obs = True

    observation_noise_enabled = True
    base_lin_vel_obs_noise_std = 0.025
    angular_velocity_obs_noise_std = 0.015
    projected_gravity_obs_noise_std = 0.008
    joint_position_obs_noise_std = 0.006
    joint_velocity_obs_noise_std = 0.035
    action_obs_noise_std = 0.002
    contact_obs_dropout_prob = 0.015
    timer_obs_noise_std = 0.015

    # Deployment adaptation should not rediscover a new visual gait. These
    # light guards preserve the accepted model_1798 rear-leg shape while the
    # policy adapts to estimated/noisy velocity observations.
    rear_tibia_mean_symmetry_reward_scale = -0.65
    rear_tibia_command_symmetry_reward_scale = -0.35
    rear_tibia_joint_symmetry_reward_scale = -0.18
    rear_tibia_mean_symmetry_tolerance_rad = 0.18
    rear_tibia_command_symmetry_tolerance_rad = 0.16


@configclass
class InsectoidMiniQuadGaitRefineDeployPlayEnvCfg(InsectoidMiniQuadGaitRefineDeployEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadGaitRefineNoVelDeployEnvCfg(InsectoidMiniQuadGaitRefineDeployEnvCfg):
    # Control deployment contract: no true/estimated base linear velocity enters
    # the policy. This matches the current quad_test ROS adapter more closely
    # than the stance-foot estimator and lets PPO adapt without relying on a
    # state estimate that may not be available on hardware.
    use_privileged_base_lin_vel_obs = False
    use_foot_kinematic_base_lin_vel_obs = False
    base_lin_vel_obs_noise_std = 0.004


@configclass
class InsectoidMiniQuadGaitRefineNoVelDeployPlayEnvCfg(InsectoidMiniQuadGaitRefineNoVelDeployEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardEnvCfg(InsectoidMiniQuadGaitRefineEnvCfg):
    # Reverse primitive stage: force signed rearward progress first, then refine
    # gait quality once the policy is no longer choosing the stand-still local optimum.
    lin_vel_y_range = (-0.20, -0.12)
    ang_vel_z_range = (-0.02, 0.02)

    lin_vel_reward_scale = 1.50
    lin_vel_tracking_sigma = 0.045
    command_progress_reward_scale = 2.50
    command_progress_floor_reward_scale = -3.00
    min_command_progress_ratio = 0.55
    flat_orientation_reward_scale = -6.0
    pitch_forward_reward_scale = -2.0

    touchdown_stride_reward_scale = 3.5
    touchdown_short_stride_reward_scale = -2.2
    touchdown_stride_target = 0.14
    min_touchdown_stride = 0.06
    touchdown_cadence_reward_scale = -0.35
    rapid_touchdown_reward_scale = -0.45
    stance_anchor_slip_reward_scale = -0.75
    max_touchdown_rate_hz = 4.0


@configclass
class InsectoidMiniQuadBackwardPlayEnvCfg(InsectoidMiniQuadBackwardEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardStabilizeEnvCfg(InsectoidMiniQuadBackwardEnvCfg):
    # Keep the strict signed-progress objective from the backward discovery
    # stage, but restore forward-style attitude and phase discipline.
    lin_vel_y_range = (-0.20, -0.14)
    ang_vel_z_range = (-0.01, 0.01)

    yaw_rate_reward_scale = 1.25
    yaw_drift_reward_scale = -1.60
    yaw_drift_tolerance_rad = 0.12
    lateral_velocity_reward_scale = -1.00
    ang_vel_reward_scale = -0.12
    flat_orientation_reward_scale = -14.0
    pitch_forward_reward_scale = -10.0

    contact_balance_reward_scale = -0.70
    diagonal_trot_reward_scale = 0.80
    diagonal_pair_sync_reward_scale = -0.80
    side_antiphase_reward_scale = 0.60
    touchdown_rate_balance_reward_scale = -0.35
    touchdown_cadence_reward_scale = -0.70
    rapid_touchdown_reward_scale = -1.00
    stance_anchor_slip_reward_scale = -1.05
    max_touchdown_rate_hz = 5.0

    lin_vel_reward_scale = 1.25
    command_progress_reward_scale = 2.25
    command_progress_floor_reward_scale = -3.00
    min_command_progress_ratio = 0.55


@configclass
class InsectoidMiniQuadBackwardStabilizePlayEnvCfg(InsectoidMiniQuadBackwardStabilizeEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardStabilizeV2EnvCfg(InsectoidMiniQuadBackwardStabilizeEnvCfg):
    # V1 reduced yaw rate, but still allowed diagonal backward drift. V2 makes
    # straight-line reverse tracking stricter while keeping the diagonal gait
    # phase discipline learned in V1.
    lin_vel_y_range = (-0.18, -0.14)
    lin_vel_tracking_sigma = 0.030

    lin_vel_reward_scale = 1.55
    command_progress_reward_scale = 1.70
    command_progress_floor_reward_scale = -3.20
    command_overspeed_reward_scale = -1.60
    command_overspeed_tolerance = 0.06

    yaw_rate_reward_scale = 1.80
    yaw_drift_reward_scale = -2.60
    yaw_drift_tolerance_rad = 0.08
    lateral_velocity_reward_scale = -2.25
    lateral_position_reward_scale = -2.20
    lateral_position_tolerance_m = 0.07
    ang_vel_reward_scale = -0.18
    flat_orientation_reward_scale = -18.0
    pitch_forward_reward_scale = -12.0

    contact_balance_reward_scale = -0.85
    diagonal_pair_sync_reward_scale = -1.20
    side_antiphase_reward_scale = 0.90
    touchdown_rate_balance_reward_scale = -0.45
    stance_anchor_slip_reward_scale = -1.15


@configclass
class InsectoidMiniQuadBackwardStabilizeV2PlayEnvCfg(InsectoidMiniQuadBackwardStabilizeV2EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardSymmetryEnvCfg(InsectoidMiniQuadBackwardStabilizeEnvCfg):
    # Directly target the yaw source: asymmetric rear-leg posture/timing. This
    # starts from the faster V1 backward gait and nudges the two rear legs
    # toward matching motion instead of only penalizing body yaw afterward.
    lin_vel_y_range = (-0.18, -0.14)
    lin_vel_tracking_sigma = 0.035

    lin_vel_reward_scale = 1.35
    command_progress_reward_scale = 1.90
    command_progress_floor_reward_scale = -3.00
    command_overspeed_reward_scale = -0.80
    command_overspeed_tolerance = 0.08

    yaw_rate_reward_scale = 1.65
    yaw_drift_reward_scale = -2.20
    yaw_drift_tolerance_rad = 0.09
    lateral_velocity_reward_scale = -1.50
    lateral_position_reward_scale = -0.85
    lateral_position_tolerance_m = 0.09
    flat_orientation_reward_scale = -16.0
    pitch_forward_reward_scale = -11.0

    diagonal_pair_sync_reward_scale = -1.10
    side_antiphase_reward_scale = 0.90
    rear_stride_balance_reward_scale = -0.25
    rear_stance_duration_balance_reward_scale = -0.15
    rear_touchdown_rate_balance_reward_scale = -0.18
    rear_contact_fraction_balance_reward_scale = -0.12
    rear_stance_slip_balance_reward_scale = -0.30
    rear_tibia_mean_symmetry_reward_scale = -2.20
    rear_tibia_mean_target_reward_scale = -0.80
    rear_tibia_command_symmetry_reward_scale = -1.60
    rear_tibia_command_target_reward_scale = -0.90
    rear_tibia_tuck_envelope_reward_scale = -0.25
    rear_tibia_mean_symmetry_tolerance_rad = 0.14
    rear_tibia_mean_target_rad = 1.90
    rear_tibia_mean_target_tolerance_rad = 0.28
    rear_tibia_command_symmetry_tolerance_rad = 0.14
    rear_tibia_command_target_tolerance_rad = 0.25
    rear_tibia_tuck_limit_rad = 2.28
    rear_tibia_tuck_tolerance_rad = 0.24


@configclass
class InsectoidMiniQuadBackwardSymmetryPlayEnvCfg(InsectoidMiniQuadBackwardSymmetryEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardAntiInwardEnvCfg(InsectoidMiniQuadBackwardStabilizeEnvCfg):
    # Recovery stage for baked BL cleanup. Keep the backward gait moving, but
    # explicitly prevent the back-left leg from curling inward relative to the
    # back-right leg.
    lin_vel_y_range = (-0.18, -0.14)
    lin_vel_tracking_sigma = 0.040

    lin_vel_reward_scale = 2.25
    command_progress_reward_scale = 3.60
    command_progress_floor_reward_scale = -8.00
    min_command_progress_ratio = 0.84
    command_overspeed_reward_scale = -0.70
    command_overspeed_tolerance = 0.10

    yaw_rate_reward_scale = 2.20
    yaw_drift_reward_scale = -3.20
    yaw_drift_tolerance_rad = 0.07
    lateral_velocity_reward_scale = -1.60
    lateral_position_reward_scale = -0.75
    lateral_position_tolerance_m = 0.08
    flat_orientation_reward_scale = -18.0
    pitch_forward_reward_scale = -12.0

    diagonal_pair_sync_reward_scale = -1.45
    side_antiphase_reward_scale = 1.20
    rear_contact_antiphase_reward_scale = 0.75
    touchdown_rate_balance_reward_scale = -0.45
    stance_anchor_slip_reward_scale = -1.15

    bl_coxa_pair_offset_reward_scale = -1.10
    bl_femur_pair_offset_reward_scale = -1.35
    bl_tibia_pair_offset_reward_scale = -4.50
    rear_tibia_mean_symmetry_reward_scale = -2.50
    rear_tibia_mean_target_reward_scale = -0.70
    rear_tibia_command_target_reward_scale = -0.45
    bl_coxa_command_pair_offset_reward_scale = -0.85
    bl_femur_command_pair_offset_reward_scale = -1.10
    bl_tibia_command_pair_offset_reward_scale = -3.50
    rear_coxa_joint_symmetry_reward_scale = -0.30
    rear_femur_joint_symmetry_reward_scale = -0.25
    rear_tibia_joint_symmetry_reward_scale = -0.35
    rear_coxa_command_symmetry_reward_scale = -0.12
    rear_femur_command_symmetry_reward_scale = -0.12
    rear_tibia_command_symmetry_abs_reward_scale = -0.16
    rear_femur_tibia_segment_mirror_reward_scale = -1.10
    rear_femur_tibia_z_mirror_reward_scale = 0.0
    rear_tibia_link_mirror_reward_scale = -1.20
    rear_tibia_segment_mirror_reward_scale = -1.80
    rear_tibia_segment_outward_reward_scale = -0.80
    rear_tibia_segment_outward_target_reward_scale = -0.55
    rear_swing_tibia_segment_outward_reward_scale = -0.65
    bl_swing_tibia_segment_outward_reward_scale = -1.80
    bl_lifted_swing_tibia_segment_outward_reward_scale = 0.0
    bl_phase_delayed_tibia_outward_reward_scale = 0.0
    rear_swing_outward_mean_balance_reward_scale = -2.20
    bl_swing_outward_mean_floor_reward_scale = -1.15
    bl_swing_outward_vs_br_reward_scale = 0.0
    bl_lifted_swing_outward_vs_br_reward_scale = 0.0
    rear_swing_femur_floor_reward_scale = 0.0
    bl_swing_femur_floor_reward_scale = 0.0
    rear_swing_femur_command_floor_reward_scale = 0.0
    bl_swing_femur_command_floor_reward_scale = 0.0
    bl_swing_height_reward_scale = 0.0
    bl_coxa_pair_offset_limit_rad = 0.32
    bl_coxa_pair_offset_tolerance_rad = 0.12
    bl_femur_pair_offset_limit_rad = 0.20
    bl_femur_pair_offset_tolerance_rad = 0.12
    bl_tibia_pair_offset_limit_rad = 0.35
    bl_tibia_pair_offset_tolerance_rad = 0.14
    rear_tibia_mean_symmetry_tolerance_rad = 0.035
    rear_tibia_mean_target_rad = 1.58
    rear_tibia_mean_target_tolerance_rad = 0.14
    rear_tibia_command_target_tolerance_rad = 0.22
    rear_coxa_joint_symmetry_tolerance_rad = 0.090
    rear_femur_joint_symmetry_tolerance_rad = 0.080
    rear_tibia_joint_symmetry_tolerance_rad = 0.080
    rear_coxa_command_symmetry_tolerance_rad = 0.100
    rear_femur_command_symmetry_tolerance_rad = 0.100
    rear_tibia_command_symmetry_abs_tolerance_rad = 0.100
    rear_femur_tibia_segment_mirror_tolerance_m = 0.012
    rear_femur_tibia_z_mirror_tolerance_m = 0.007
    rear_tibia_link_mirror_tolerance_m = 0.020
    rear_tibia_segment_mirror_tolerance_m = 0.010
    rear_tibia_segment_outward_min_m = 0.028
    rear_tibia_segment_outward_tolerance_m = 0.012
    rear_tibia_segment_outward_target_m = 0.034
    rear_tibia_segment_outward_target_tolerance_m = 0.010
    rear_swing_tibia_segment_outward_target_m = 0.043
    rear_swing_tibia_segment_outward_tolerance_m = 0.010
    bl_lifted_swing_tibia_segment_outward_target_m = 0.047
    bl_lifted_swing_tibia_segment_outward_tolerance_m = 0.007
    bl_lifted_swing_min_height_m = 0.035
    bl_lifted_swing_height_tolerance_m = 0.020
    rear_phase_symmetry_delay_steps = 12
    bl_phase_delayed_tibia_outward_tolerance_m = 0.007
    rear_swing_outward_mean_balance_tolerance_m = 0.004
    bl_swing_outward_mean_floor_target_m = 0.038
    bl_swing_outward_mean_floor_tolerance_m = 0.006
    bl_swing_outward_vs_br_tolerance_m = 0.004
    bl_lifted_swing_outward_vs_br_tolerance_m = 0.003
    rear_swing_femur_floor_rad = -0.43
    rear_swing_femur_floor_tolerance_rad = 0.06


@configclass
class InsectoidMiniQuadBackwardAntiInwardPlayEnvCfg(InsectoidMiniQuadBackwardAntiInwardEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLNoCurlEnvCfg(InsectoidMiniQuadBackwardAntiInwardEnvCfg):
    # Fine-tune stage from the baked v2 backward policy. Keep the useful
    # backward gait, but make BL swing/lift outward geometry the dominant
    # cleanup target instead of forcing direct femur equality.
    lin_vel_y_range = (-0.18, -0.15)
    lin_vel_tracking_sigma = 0.035

    lin_vel_reward_scale = 2.45
    command_progress_reward_scale = 4.00
    command_progress_floor_reward_scale = -9.00
    min_command_progress_ratio = 0.84
    command_overspeed_reward_scale = -0.85

    yaw_rate_reward_scale = 2.45
    yaw_drift_reward_scale = -4.00
    yaw_drift_tolerance_rad = 0.055
    lateral_velocity_reward_scale = -2.00
    lateral_position_reward_scale = -1.00
    lateral_position_tolerance_m = 0.070
    flat_orientation_reward_scale = -18.0
    pitch_forward_reward_scale = -12.0

    diagonal_pair_sync_reward_scale = -1.45
    side_antiphase_reward_scale = 1.20
    rear_contact_antiphase_reward_scale = 0.75
    touchdown_rate_balance_reward_scale = -0.45
    stance_anchor_slip_reward_scale = -1.15

    bl_coxa_pair_offset_reward_scale = -0.80
    bl_femur_pair_offset_reward_scale = -0.45
    bl_tibia_pair_offset_reward_scale = -5.00
    rear_tibia_mean_symmetry_reward_scale = -2.80
    rear_tibia_mean_target_reward_scale = -0.55
    rear_tibia_command_target_reward_scale = -0.45
    bl_coxa_command_pair_offset_reward_scale = -0.60
    bl_femur_command_pair_offset_reward_scale = -0.35
    bl_tibia_command_pair_offset_reward_scale = -4.00
    rear_coxa_joint_symmetry_reward_scale = -0.20
    rear_femur_joint_symmetry_reward_scale = -0.10
    rear_tibia_joint_symmetry_reward_scale = -0.40
    rear_coxa_command_symmetry_reward_scale = -0.08
    rear_femur_command_symmetry_reward_scale = -0.05
    rear_tibia_command_symmetry_abs_reward_scale = -0.20

    rear_femur_tibia_segment_mirror_reward_scale = -1.25
    rear_tibia_link_mirror_reward_scale = -1.45
    rear_tibia_segment_mirror_reward_scale = -2.25
    rear_tibia_segment_outward_reward_scale = -0.85
    rear_tibia_segment_outward_target_reward_scale = -0.70
    rear_swing_tibia_segment_outward_reward_scale = -0.95
    bl_swing_tibia_segment_outward_reward_scale = -3.40
    bl_lifted_swing_tibia_segment_outward_reward_scale = -2.40
    bl_phase_delayed_tibia_outward_reward_scale = -0.35
    rear_swing_outward_mean_balance_reward_scale = -3.00
    bl_swing_outward_mean_floor_reward_scale = -2.10
    bl_swing_outward_vs_br_reward_scale = -1.20
    bl_lifted_swing_outward_vs_br_reward_scale = -1.60
    bl_swing_height_reward_scale = 0.35

    rear_tibia_mean_symmetry_tolerance_rad = 0.032
    rear_tibia_mean_target_rad = 1.56
    rear_tibia_mean_target_tolerance_rad = 0.13
    rear_tibia_command_target_tolerance_rad = 0.20
    rear_tibia_segment_mirror_tolerance_m = 0.009
    rear_tibia_segment_outward_min_m = 0.030
    rear_tibia_segment_outward_target_m = 0.038
    rear_swing_tibia_segment_outward_target_m = 0.046
    rear_swing_tibia_segment_outward_tolerance_m = 0.008
    bl_lifted_swing_tibia_segment_outward_target_m = 0.048
    bl_lifted_swing_tibia_segment_outward_tolerance_m = 0.006
    rear_swing_outward_mean_balance_tolerance_m = 0.003
    bl_swing_outward_mean_floor_target_m = 0.043
    bl_swing_outward_mean_floor_tolerance_m = 0.005
    bl_swing_outward_vs_br_tolerance_m = 0.003
    bl_lifted_swing_outward_vs_br_tolerance_m = 0.0025


@configclass
class InsectoidMiniQuadBackwardBLNoCurlPlayEnvCfg(InsectoidMiniQuadBackwardBLNoCurlEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLNoCurlBalancedEnvCfg(InsectoidMiniQuadBackwardAntiInwardEnvCfg):
    # Conservative follow-up from baked v2: reduce BL inward curl while
    # explicitly preserving rear contact timing and straight backward travel.
    lin_vel_y_range = (-0.18, -0.15)
    lin_vel_tracking_sigma = 0.032

    lin_vel_reward_scale = 2.65
    command_progress_reward_scale = 4.20
    command_progress_floor_reward_scale = -9.00
    min_command_progress_ratio = 0.86
    command_overspeed_reward_scale = -0.90

    yaw_rate_reward_scale = 2.70
    yaw_drift_reward_scale = -6.00
    yaw_drift_tolerance_rad = 0.045
    lateral_velocity_reward_scale = -2.60
    lateral_position_reward_scale = -2.40
    lateral_position_tolerance_m = 0.055
    flat_orientation_reward_scale = -18.0
    pitch_forward_reward_scale = -12.0

    contact_balance_reward_scale = -1.10
    diagonal_pair_sync_reward_scale = -1.55
    side_antiphase_reward_scale = 1.20
    rear_contact_antiphase_reward_scale = 0.75
    touchdown_rate_balance_reward_scale = -0.55
    rear_touchdown_rate_balance_reward_scale = -0.45
    rear_contact_fraction_balance_reward_scale = -0.60
    rear_stance_duration_balance_reward_scale = -0.25
    rear_stance_slip_balance_reward_scale = -0.35
    stance_anchor_slip_reward_scale = -1.25

    bl_coxa_pair_offset_reward_scale = -0.70
    bl_femur_pair_offset_reward_scale = -0.25
    bl_tibia_pair_offset_reward_scale = -3.80
    rear_tibia_mean_symmetry_reward_scale = -2.00
    rear_tibia_mean_target_reward_scale = -0.45
    rear_tibia_command_target_reward_scale = -0.40
    bl_coxa_command_pair_offset_reward_scale = -0.55
    bl_femur_command_pair_offset_reward_scale = -0.20
    bl_tibia_command_pair_offset_reward_scale = -3.20
    rear_coxa_joint_symmetry_reward_scale = -0.18
    rear_femur_joint_symmetry_reward_scale = -0.08
    rear_tibia_joint_symmetry_reward_scale = -0.28
    rear_coxa_command_symmetry_reward_scale = -0.08
    rear_femur_command_symmetry_reward_scale = -0.04
    rear_tibia_command_symmetry_abs_reward_scale = -0.14

    rear_femur_tibia_segment_mirror_reward_scale = -1.00
    rear_tibia_link_mirror_reward_scale = -1.20
    rear_tibia_segment_mirror_reward_scale = -1.45
    rear_tibia_segment_outward_reward_scale = -0.70
    rear_tibia_segment_outward_target_reward_scale = -0.45
    rear_swing_tibia_segment_outward_reward_scale = -0.70
    bl_swing_tibia_segment_outward_reward_scale = -1.35
    bl_lifted_swing_tibia_segment_outward_reward_scale = -0.85
    bl_phase_delayed_tibia_outward_reward_scale = -0.15
    rear_swing_outward_mean_balance_reward_scale = -1.40
    bl_swing_outward_mean_floor_reward_scale = -0.95
    bl_swing_outward_vs_br_reward_scale = -0.45
    bl_lifted_swing_outward_vs_br_reward_scale = -0.55
    bl_swing_height_reward_scale = 0.20

    rear_tibia_mean_symmetry_tolerance_rad = 0.035
    rear_tibia_mean_target_rad = 1.56
    rear_tibia_mean_target_tolerance_rad = 0.13
    rear_tibia_command_target_tolerance_rad = 0.20
    rear_tibia_segment_mirror_tolerance_m = 0.010
    rear_tibia_segment_outward_min_m = 0.029
    rear_tibia_segment_outward_target_m = 0.036
    rear_swing_tibia_segment_outward_target_m = 0.042
    rear_swing_tibia_segment_outward_tolerance_m = 0.008
    bl_lifted_swing_tibia_segment_outward_target_m = 0.044
    bl_lifted_swing_tibia_segment_outward_tolerance_m = 0.006
    rear_swing_outward_mean_balance_tolerance_m = 0.004
    bl_swing_outward_mean_floor_target_m = 0.041
    bl_swing_outward_mean_floor_tolerance_m = 0.006
    bl_swing_outward_vs_br_tolerance_m = 0.004
    bl_lifted_swing_outward_vs_br_tolerance_m = 0.003


@configclass
class InsectoidMiniQuadBackwardBLNoCurlBalancedPlayEnvCfg(InsectoidMiniQuadBackwardBLNoCurlBalancedEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkEnvCfg(InsectoidMiniQuadBackwardBLNoCurlBalancedEnvCfg):
    # Follow-up from the balanced run. This directly penalizes the visual issue:
    # the BL tibia link sitting inward of the mirrored BR tibia link.
    yaw_rate_reward_scale = 2.85
    yaw_drift_reward_scale = -6.50
    lateral_velocity_reward_scale = -2.80
    lateral_position_reward_scale = -2.70

    contact_balance_reward_scale = -1.20
    rear_touchdown_rate_balance_reward_scale = -0.52
    rear_contact_fraction_balance_reward_scale = -0.72
    rear_stance_slip_balance_reward_scale = -0.42

    rear_tibia_link_mirror_reward_scale = -1.55
    bl_tibia_link_outward_vs_br_reward_scale = -3.25
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -2.25
    rear_tibia_segment_mirror_reward_scale = -1.55
    bl_swing_tibia_segment_outward_reward_scale = -1.20
    bl_lifted_swing_tibia_segment_outward_reward_scale = -0.70
    rear_swing_outward_mean_balance_reward_scale = -1.25
    bl_swing_outward_mean_floor_reward_scale = -0.85
    bl_swing_outward_vs_br_reward_scale = -0.35
    bl_lifted_swing_outward_vs_br_reward_scale = -0.45

    rear_tibia_joint_symmetry_reward_scale = -0.42
    rear_femur_joint_symmetry_reward_scale = -0.12
    rear_tibia_command_symmetry_abs_reward_scale = -0.18

    rear_tibia_link_mirror_tolerance_m = 0.012
    bl_tibia_link_outward_vs_br_tolerance_m = 0.0025
    bl_lifted_swing_tibia_link_outward_vs_br_tolerance_m = 0.0020
    rear_tibia_segment_mirror_tolerance_m = 0.010


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkPlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkEnvCfg):
    # Recovery pass from the cleanest tibia-link checkpoint. Keep a soft
    # one-sided BL tibia guard, but make gait timing and attitude dominant.
    lin_vel_tracking_sigma = 0.030
    lin_vel_reward_scale = 2.85
    command_progress_reward_scale = 4.60
    command_progress_floor_reward_scale = -10.00
    min_command_progress_ratio = 0.88
    command_overspeed_reward_scale = -0.95

    yaw_rate_reward_scale = 3.20
    yaw_drift_reward_scale = -8.50
    yaw_drift_tolerance_rad = 0.040
    lateral_velocity_reward_scale = -3.20
    lateral_position_reward_scale = -3.80
    lateral_position_tolerance_m = 0.050

    contact_balance_reward_scale = -1.55
    diagonal_pair_sync_reward_scale = -1.75
    rear_touchdown_rate_balance_reward_scale = -0.85
    rear_contact_fraction_balance_reward_scale = -1.35
    rear_stance_duration_balance_reward_scale = -0.55
    rear_stance_slip_balance_reward_scale = -0.65
    stance_anchor_slip_reward_scale = -1.35

    rear_tibia_link_mirror_reward_scale = -1.00
    bl_tibia_link_outward_vs_br_reward_scale = -1.35
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -0.90
    rear_tibia_segment_mirror_reward_scale = -1.00
    bl_swing_tibia_segment_outward_reward_scale = -0.80
    bl_lifted_swing_tibia_segment_outward_reward_scale = -0.45
    rear_swing_outward_mean_balance_reward_scale = -1.10
    bl_swing_outward_mean_floor_reward_scale = -0.55
    bl_swing_outward_vs_br_reward_scale = -0.30
    bl_lifted_swing_outward_vs_br_reward_scale = -0.30

    rear_tibia_joint_symmetry_reward_scale = -0.55
    rear_femur_joint_symmetry_reward_scale = -0.20
    rear_tibia_command_symmetry_abs_reward_scale = -0.22

    bl_tibia_link_outward_vs_br_tolerance_m = 0.0040
    bl_lifted_swing_tibia_link_outward_vs_br_tolerance_m = 0.0035
    rear_contact_fraction_balance_tolerance = 0.12


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalancePlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV2EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceEnvCfg):
    # V2 keeps the useful outward BL tibia-link bias from the 2595 candidate,
    # but tightens cadence/contact and straight-line terms so the policy cannot
    # buy visual symmetry by slowing down or drifting.
    lin_vel_tracking_sigma = 0.026
    lin_vel_reward_scale = 3.05
    command_progress_reward_scale = 5.10
    command_progress_floor_reward_scale = -12.50
    min_command_progress_ratio = 0.90
    command_overspeed_reward_scale = -1.10

    yaw_rate_reward_scale = 3.45
    yaw_drift_reward_scale = -9.00
    yaw_drift_tolerance_rad = 0.038
    lateral_velocity_reward_scale = -3.45
    lateral_position_reward_scale = -4.25
    lateral_position_tolerance_m = 0.048

    contact_balance_reward_scale = -1.75
    diagonal_pair_sync_reward_scale = -1.95
    rear_touchdown_rate_balance_reward_scale = -1.05
    rear_contact_fraction_balance_reward_scale = -1.75
    rear_stance_duration_balance_reward_scale = -0.70
    rear_stance_slip_balance_reward_scale = -0.78
    stance_anchor_slip_reward_scale = -1.55

    rear_tibia_link_mirror_reward_scale = -1.15
    bl_tibia_link_outward_vs_br_reward_scale = -1.85
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -1.25
    rear_tibia_segment_mirror_reward_scale = -1.15
    bl_swing_tibia_segment_outward_reward_scale = -0.90
    bl_lifted_swing_tibia_segment_outward_reward_scale = -0.55
    rear_swing_outward_mean_balance_reward_scale = -1.20
    bl_swing_outward_mean_floor_reward_scale = -0.62
    bl_swing_outward_vs_br_reward_scale = -0.32
    bl_lifted_swing_outward_vs_br_reward_scale = -0.34

    rear_tibia_joint_symmetry_reward_scale = -0.60
    rear_femur_joint_symmetry_reward_scale = -0.23
    rear_tibia_command_symmetry_abs_reward_scale = -0.24

    bl_tibia_link_outward_vs_br_tolerance_m = 0.0032
    bl_lifted_swing_tibia_link_outward_vs_br_tolerance_m = 0.0028
    rear_contact_fraction_balance_tolerance = 0.10


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV2PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV2EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV2EnvCfg):
    # V3 starts from the best V2 checkpoint and adds an absolute BL tibia-link
    # outward floor. This targets the visual inward-curl defect directly while
    # keeping V2's contact/cadence constraints strong enough to preserve gait.
    bl_tibia_link_outward_floor_reward_scale = -1.25
    bl_lifted_swing_tibia_link_outward_floor_reward_scale = -0.85
    bl_tibia_link_outward_floor_target_m = 0.041
    bl_tibia_link_outward_floor_tolerance_m = 0.008
    bl_lifted_swing_tibia_link_outward_floor_target_m = 0.044
    bl_lifted_swing_tibia_link_outward_floor_tolerance_m = 0.006

    bl_tibia_link_outward_vs_br_reward_scale = -1.65
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -1.10
    rear_contact_fraction_balance_reward_scale = -1.90
    rear_touchdown_rate_balance_reward_scale = -1.12
    rear_stance_slip_balance_reward_scale = -0.84


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardDeployEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    use_privileged_base_lin_vel_obs = False
    use_foot_kinematic_base_lin_vel_obs = True

    observation_noise_enabled = True
    base_lin_vel_obs_noise_std = 0.025
    angular_velocity_obs_noise_std = 0.015
    projected_gravity_obs_noise_std = 0.008
    joint_position_obs_noise_std = 0.006
    joint_velocity_obs_noise_std = 0.035
    action_obs_noise_std = 0.002
    contact_obs_dropout_prob = 0.015
    timer_obs_noise_std = 0.015


@configclass
class InsectoidMiniQuadBackwardDeployPlayEnvCfg(InsectoidMiniQuadBackwardDeployEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardNoVelDeployEnvCfg(InsectoidMiniQuadBackwardDeployEnvCfg):
    use_privileged_base_lin_vel_obs = False
    use_foot_kinematic_base_lin_vel_obs = False
    base_lin_vel_obs_noise_std = 0.004


@configclass
class InsectoidMiniQuadBackwardNoVelDeployPlayEnvCfg(InsectoidMiniQuadBackwardNoVelDeployEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV4EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V4 corrects the V3 absolute floor target. The tibia-link body-frame
    # outward distance is about 0.30 m, not 0.04 m, so V3's floor rarely
    # activated. Keep this as a low-tail guard instead of a hard outward splay.
    bl_tibia_link_outward_floor_reward_scale = -0.55
    bl_lifted_swing_tibia_link_outward_floor_reward_scale = -0.70
    bl_tibia_link_outward_floor_target_m = 0.300
    bl_tibia_link_outward_floor_tolerance_m = 0.012
    bl_lifted_swing_tibia_link_outward_floor_target_m = 0.302
    bl_lifted_swing_tibia_link_outward_floor_tolerance_m = 0.010

    bl_tibia_link_outward_vs_br_reward_scale = -2.20
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -1.65
    bl_tibia_link_outward_vs_br_tolerance_m = 0.0026
    bl_lifted_swing_tibia_link_outward_vs_br_tolerance_m = 0.0022

    rear_tibia_link_mirror_reward_scale = -1.25
    rear_tibia_segment_mirror_reward_scale = -1.25
    rear_contact_fraction_balance_reward_scale = -2.05
    rear_touchdown_rate_balance_reward_scale = -1.20
    rear_stance_slip_balance_reward_scale = -0.90


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV4PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV4EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV5EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V5 backs off V4's all-frame floor and focuses on the lifted swing phase,
    # where the inward tibia curl is visually obvious and less tied to support.
    bl_tibia_link_outward_floor_reward_scale = -0.10
    bl_tibia_link_outward_floor_target_m = 0.294
    bl_tibia_link_outward_floor_tolerance_m = 0.010
    bl_lifted_swing_tibia_link_outward_floor_reward_scale = -0.45
    bl_lifted_swing_tibia_link_outward_floor_target_m = 0.307
    bl_lifted_swing_tibia_link_outward_floor_tolerance_m = 0.010

    bl_tibia_link_outward_vs_br_reward_scale = -1.70
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -1.20
    rear_contact_fraction_balance_reward_scale = -1.95
    rear_touchdown_rate_balance_reward_scale = -1.15
    rear_stance_slip_balance_reward_scale = -0.86


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV5PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV5EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV6EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V6 compares the BL tibia link against BR at the matched swing phase,
    # instead of tightening same-frame or absolute floors that disturbed gait.
    bl_tibia_link_outward_floor_reward_scale = 0.0
    bl_lifted_swing_tibia_link_outward_floor_reward_scale = 0.0
    bl_tibia_link_outward_vs_br_reward_scale = -1.25
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -0.75

    bl_phase_delayed_tibia_link_outward_reward_scale = -0.85
    bl_lifted_phase_delayed_tibia_link_outward_reward_scale = -0.70
    bl_phase_delayed_tibia_link_outward_tolerance_m = 0.0035
    bl_lifted_phase_delayed_tibia_link_outward_tolerance_m = 0.0025

    rear_contact_fraction_balance_reward_scale = -1.95
    rear_touchdown_rate_balance_reward_scale = -1.15
    rear_stance_slip_balance_reward_scale = -0.86


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV6PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV6EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV7EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V7 keeps the successful V3 gait intact and applies only a light
    # phase-matched BL tibia-link correction. V6 was too strong and disturbed
    # rear contact balance, so this version is intended as a small policy nudge.
    bl_phase_delayed_tibia_link_outward_reward_scale = -0.18
    bl_lifted_phase_delayed_tibia_link_outward_reward_scale = -0.12
    bl_phase_delayed_tibia_link_outward_tolerance_m = 0.0050
    bl_lifted_phase_delayed_tibia_link_outward_tolerance_m = 0.0040

    rear_contact_fraction_balance_reward_scale = -1.95
    rear_touchdown_rate_balance_reward_scale = -1.15
    rear_stance_slip_balance_reward_scale = -0.86


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV7PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV7EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV8EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V8 targets the observed visual defect as a phase-matched mirror problem:
    # compare BL's tibia-link shape against BR's delayed matching gait phase,
    # instead of pushing BL outward in the current frame.
    rear_tibia_link_mirror_reward_scale = -0.65
    rear_femur_tibia_segment_mirror_reward_scale = -0.75
    rear_tibia_segment_mirror_reward_scale = -0.80

    bl_tibia_link_outward_vs_br_reward_scale = -1.45
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -0.95
    bl_phase_delayed_tibia_link_outward_reward_scale = -0.06
    bl_lifted_phase_delayed_tibia_link_outward_reward_scale = -0.04

    bl_phase_delayed_tibia_link_mirror_reward_scale = -0.32
    bl_lifted_phase_delayed_tibia_link_mirror_reward_scale = -0.20
    bl_phase_delayed_femur_tibia_mirror_reward_scale = -0.24
    bl_lifted_phase_delayed_femur_tibia_mirror_reward_scale = -0.16
    bl_phase_delayed_tibia_link_mirror_tolerance_m = 0.012
    bl_lifted_phase_delayed_tibia_link_mirror_tolerance_m = 0.010
    bl_phase_delayed_femur_tibia_mirror_tolerance_m = 0.011
    bl_lifted_phase_delayed_femur_tibia_mirror_tolerance_m = 0.009

    rear_contact_fraction_balance_reward_scale = -1.95
    rear_touchdown_rate_balance_reward_scale = -1.15
    rear_stance_slip_balance_reward_scale = -0.86


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV8PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV8EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV9EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V9 keeps the V3 geometry guard but makes the main correction phase-aware:
    # BL femur/tibia should resemble BR femur/tibia at the matching delayed
    # rear-leg swing phase. This avoids same-frame symmetry forcing, which is
    # wrong for an out-of-phase gait.
    yaw_rate_reward_scale = 3.80
    yaw_drift_reward_scale = -10.00
    yaw_drift_tolerance_rad = 0.035
    lateral_velocity_reward_scale = -3.70
    lateral_position_reward_scale = -4.60
    lateral_position_tolerance_m = 0.045
    flat_orientation_reward_scale = -20.0
    pitch_forward_reward_scale = -14.0

    contact_balance_reward_scale = -1.85
    diagonal_pair_sync_reward_scale = -2.20
    side_antiphase_reward_scale = 1.35
    rear_contact_antiphase_reward_scale = 1.40
    rear_touchdown_rate_balance_reward_scale = -1.35
    rear_contact_fraction_balance_reward_scale = -2.25
    rear_stance_duration_balance_reward_scale = -0.70
    rear_stance_slip_balance_reward_scale = -0.95
    stance_anchor_slip_reward_scale = -1.55

    bl_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.45
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.30
    bl_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.30
    bl_lifted_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.20
    bl_phase_delayed_rear_distal_joint_symmetry_tolerance_rad = 0.13
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_tolerance_rad = 0.11
    bl_phase_delayed_rear_distal_command_symmetry_tolerance_rad = 0.12
    bl_lifted_phase_delayed_rear_distal_command_symmetry_tolerance_rad = 0.10

    bl_tibia_link_outward_vs_br_reward_scale = -1.35
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -0.85
    bl_tibia_link_outward_floor_reward_scale = -0.18
    bl_lifted_swing_tibia_link_outward_floor_reward_scale = -0.38
    bl_tibia_link_outward_floor_target_m = 0.296
    bl_tibia_link_outward_floor_tolerance_m = 0.014
    bl_lifted_swing_tibia_link_outward_floor_target_m = 0.304
    bl_lifted_swing_tibia_link_outward_floor_tolerance_m = 0.010
    rear_tibia_link_mirror_reward_scale = -0.60
    rear_tibia_segment_mirror_reward_scale = -0.70
    rear_femur_tibia_segment_mirror_reward_scale = -0.55
    rear_tibia_joint_symmetry_reward_scale = -0.38
    rear_femur_joint_symmetry_reward_scale = -0.16
    rear_tibia_command_symmetry_abs_reward_scale = -0.16


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV9PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV9EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV10EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V10 targets the visual tibia-link rotation defect directly. The V3 link
    # center metrics were mostly clean; the bad lifted frames show up as a
    # mirrored tibia local-axis error, so this is narrower than V9's joint
    # imitation.
    yaw_rate_reward_scale = 3.55
    yaw_drift_reward_scale = -9.50
    yaw_drift_tolerance_rad = 0.038
    lateral_velocity_reward_scale = -3.45
    lateral_position_reward_scale = -4.20
    lateral_position_tolerance_m = 0.047

    contact_balance_reward_scale = -1.65
    diagonal_pair_sync_reward_scale = -1.95
    side_antiphase_reward_scale = 1.25
    rear_contact_antiphase_reward_scale = 1.05
    rear_touchdown_rate_balance_reward_scale = -1.20
    rear_contact_fraction_balance_reward_scale = -2.05
    rear_stance_duration_balance_reward_scale = -0.62
    rear_stance_slip_balance_reward_scale = -0.88

    rear_tibia_axis_mirror_reward_scale = -0.32
    bl_lifted_tibia_axis_mirror_reward_scale = -0.62
    rear_tibia_axis_mirror_tolerance = 0.060
    bl_lifted_tibia_axis_mirror_tolerance = 0.045

    bl_tibia_link_outward_vs_br_reward_scale = -1.35
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -0.90
    bl_tibia_link_outward_floor_reward_scale = -0.12
    bl_lifted_swing_tibia_link_outward_floor_reward_scale = -0.28
    bl_tibia_link_outward_floor_target_m = 0.294
    bl_tibia_link_outward_floor_tolerance_m = 0.014
    bl_lifted_swing_tibia_link_outward_floor_target_m = 0.302
    bl_lifted_swing_tibia_link_outward_floor_tolerance_m = 0.010


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV10PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV10EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV11EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V11 keeps the successful V3 gait and adds a narrow tibia-link primary-axis
    # mirror term. V10's full-axis term over-constrained USD-local frame twist,
    # so this version only penalizes the stable tibia long-axis mismatch.
    yaw_rate_reward_scale = 3.60
    yaw_drift_reward_scale = -9.80
    yaw_drift_tolerance_rad = 0.038
    lateral_velocity_reward_scale = -3.55
    lateral_position_reward_scale = -4.35
    lateral_position_tolerance_m = 0.045
    flat_orientation_reward_scale = -20.0
    pitch_forward_reward_scale = -14.0

    contact_balance_reward_scale = -1.75
    diagonal_pair_sync_reward_scale = -2.05
    side_antiphase_reward_scale = 1.30
    rear_contact_antiphase_reward_scale = 1.25
    rear_touchdown_rate_balance_reward_scale = -1.24
    rear_contact_fraction_balance_reward_scale = -2.10
    rear_stance_duration_balance_reward_scale = -0.66
    rear_stance_slip_balance_reward_scale = -0.90

    rear_tibia_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_axis_mirror_reward_scale = 0.0
    rear_tibia_primary_axis_mirror_reward_scale = -0.14
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.40
    rear_tibia_primary_axis_mirror_tolerance = 0.110
    bl_lifted_tibia_primary_axis_mirror_tolerance = 0.075

    bl_tibia_link_outward_vs_br_reward_scale = -1.35
    bl_lifted_swing_tibia_link_outward_vs_br_reward_scale = -0.90
    bl_tibia_link_outward_floor_reward_scale = -0.12
    bl_lifted_swing_tibia_link_outward_floor_reward_scale = -0.28
    bl_tibia_link_outward_floor_target_m = 0.294
    bl_tibia_link_outward_floor_tolerance_m = 0.014
    bl_lifted_swing_tibia_link_outward_floor_target_m = 0.302
    bl_lifted_swing_tibia_link_outward_floor_tolerance_m = 0.010


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV11PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV11EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV12EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V12 is a minimal policy nudge: keep V3's reward stack and only add a
    # light lifted-swing primary-axis mirror penalty. V11 improved that metric
    # but disturbed yaw, so this version avoids additional attitude/contact
    # overrides and relies on a much smaller PPO update.
    rear_tibia_primary_axis_mirror_reward_scale = -0.04
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.16
    rear_tibia_primary_axis_mirror_tolerance = 0.115
    bl_lifted_tibia_primary_axis_mirror_tolerance = 0.085


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV12PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV12EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV13EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V13 keeps V3's gait rewards but uses the V11 primary-axis pressure. This
    # tests whether the axis correction itself is useful when decoupled from
    # V11's heavier attitude/contact reward changes.
    rear_tibia_primary_axis_mirror_reward_scale = -0.14
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.40
    rear_tibia_primary_axis_mirror_tolerance = 0.110
    bl_lifted_tibia_primary_axis_mirror_tolerance = 0.075


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV13PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV13EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV14EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV13EnvCfg):
    # V14 keeps V13's narrow tibia-primary-axis visual symmetry pressure, but
    # tightens only straight-line preservation. V11 showed that changing the
    # contact/cadence stack can hurt the good V3 gait, while V13 mostly failed
    # by adding yaw/lateral drift.
    yaw_rate_reward_scale = 3.75
    yaw_drift_reward_scale = -12.00
    yaw_drift_tolerance_rad = 0.030
    lateral_velocity_reward_scale = -3.80
    lateral_position_reward_scale = -4.90
    lateral_position_tolerance_m = 0.040


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV14PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV14EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV13EnvCfg):
    # V15 keeps the same visible tibia-primary-axis cleanup pressure as V13,
    # but adds a stronger progress floor so the policy cannot improve symmetry
    # by slowing down, which was the main V14 failure mode.
    lin_vel_tracking_sigma = 0.024
    lin_vel_reward_scale = 3.35
    command_progress_reward_scale = 6.40
    command_progress_floor_reward_scale = -18.00
    min_command_progress_ratio = 0.94
    command_overspeed_reward_scale = -1.35

    yaw_rate_reward_scale = 3.70
    yaw_drift_reward_scale = -10.50
    yaw_drift_tolerance_rad = 0.035
    lateral_velocity_reward_scale = -3.60
    lateral_position_reward_scale = -4.50
    lateral_position_tolerance_m = 0.044


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV16EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15EnvCfg):
    # V16 targets the joint-space asymmetry that best matches the visible rear
    # leg mismatch: BL and BR coxa angles differ much more than femur/tibia in
    # the benchmark trajectory. Keep the speed-preserving V15 stack and make
    # coxa symmetry the dominant cleanup term.
    rear_coxa_joint_symmetry_reward_scale = -1.10
    rear_coxa_command_symmetry_reward_scale = -0.75
    rear_coxa_joint_symmetry_tolerance_rad = 0.040
    rear_coxa_command_symmetry_tolerance_rad = 0.045

    rear_tibia_primary_axis_mirror_reward_scale = -0.10
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.30


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV16PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV16EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV17EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15EnvCfg):
    # V17 targets the visible BL curl in the part of the gait where it is most
    # noticeable: rear swing/lift. V16's same-time coxa clamp disturbed cadence,
    # so this uses a short phase-delayed BR coxa reference instead.
    rear_phase_symmetry_delay_steps = 2

    rear_tibia_primary_axis_mirror_reward_scale = -0.08
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.22

    bl_phase_delayed_rear_coxa_joint_symmetry_reward_scale = -0.55
    bl_lifted_phase_delayed_rear_coxa_joint_symmetry_reward_scale = -0.85
    bl_phase_delayed_rear_coxa_command_symmetry_reward_scale = -0.30
    bl_lifted_phase_delayed_rear_coxa_command_symmetry_reward_scale = -0.45
    bl_phase_delayed_rear_coxa_joint_symmetry_tolerance_rad = 0.060
    bl_lifted_phase_delayed_rear_coxa_joint_symmetry_tolerance_rad = 0.045
    bl_phase_delayed_rear_coxa_command_symmetry_tolerance_rad = 0.070
    bl_lifted_phase_delayed_rear_coxa_command_symmetry_tolerance_rad = 0.055

    bl_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.10
    bl_lifted_phase_delayed_rear_distal_joint_symmetry_reward_scale = -0.14
    bl_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.06
    bl_lifted_phase_delayed_rear_distal_command_symmetry_reward_scale = -0.08


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV17PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV17EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV18EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV17EnvCfg):
    # V18 keeps V17's cleaner lifted tibia-axis shape but pushes progress harder
    # so the policy cannot buy symmetry by slowing the backward gait.
    lin_vel_tracking_sigma = 0.020
    lin_vel_reward_scale = 4.25
    command_progress_reward_scale = 9.20
    command_progress_floor_reward_scale = -30.00
    min_command_progress_ratio = 0.97
    command_overspeed_reward_scale = -1.90

    yaw_rate_reward_scale = 3.90
    yaw_drift_reward_scale = -11.50
    lateral_velocity_reward_scale = -3.80
    lateral_position_reward_scale = -4.70

    touchdown_stride_reward_scale = 5.25
    touchdown_short_stride_reward_scale = -4.10
    rapid_touchdown_reward_scale = -1.05

    rear_tibia_primary_axis_mirror_reward_scale = -0.10
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.26
    bl_phase_delayed_rear_coxa_joint_symmetry_reward_scale = -0.38
    bl_lifted_phase_delayed_rear_coxa_joint_symmetry_reward_scale = -0.62
    bl_phase_delayed_rear_coxa_command_symmetry_reward_scale = -0.22
    bl_lifted_phase_delayed_rear_coxa_command_symmetry_reward_scale = -0.34


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV18PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV18EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV19EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15EnvCfg):
    # V19 targets the remaining visual twist/curl directly. V17 improved the
    # primary tibia axis but slowed down; V18 over-corrected cadence. This branch
    # keeps V15 speed preservation and adds only light lifted-swing visual-axis
    # pressure on the tibia frame.
    rear_tibia_primary_axis_mirror_reward_scale = -0.07
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.18
    rear_tibia_primary_axis_mirror_tolerance = 0.115
    bl_lifted_tibia_primary_axis_mirror_tolerance = 0.085

    rear_tibia_secondary_axis_mirror_reward_scale = -0.08
    bl_lifted_tibia_secondary_axis_mirror_reward_scale = -0.30
    rear_tibia_secondary_axis_mirror_tolerance = 0.125
    bl_lifted_tibia_secondary_axis_mirror_tolerance = 0.090


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV19PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV19EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV20EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15EnvCfg):
    # V19 proved the secondary tibia axis is the right visual handle, but the
    # always-on term let the policy slow down and reorganize the gait. V20 keeps
    # the correction limited to the lifted BL swing phase and raises the speed
    # floor so the policy cannot buy symmetry by stopping.
    lin_vel_tracking_sigma = 0.021
    lin_vel_reward_scale = 4.10
    command_progress_reward_scale = 8.60
    command_progress_floor_reward_scale = -28.00
    min_command_progress_ratio = 0.965
    command_overspeed_reward_scale = -1.65

    yaw_rate_reward_scale = 3.95
    yaw_drift_reward_scale = -11.50
    yaw_drift_tolerance_rad = 0.032
    lateral_velocity_reward_scale = -3.90
    lateral_position_reward_scale = -4.85
    lateral_position_tolerance_m = 0.040

    rear_tibia_primary_axis_mirror_reward_scale = -0.035
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.095
    rear_tibia_primary_axis_mirror_tolerance = 0.125
    bl_lifted_tibia_primary_axis_mirror_tolerance = 0.095

    rear_tibia_secondary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_secondary_axis_mirror_reward_scale = -0.155
    rear_tibia_secondary_axis_mirror_tolerance = 0.135
    bl_lifted_tibia_secondary_axis_mirror_tolerance = 0.100


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV20PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV20EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV21EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15EnvCfg):
    # V21 uses a phase-delayed BR tibia-frame reference instead of same-time
    # mirroring. The rear legs are out of phase, so this gives PPO a cleaner
    # target for BL's lifted swing without forcing both rear legs into the same
    # instant of the cycle.
    rear_phase_symmetry_delay_steps = 2

    lin_vel_tracking_sigma = 0.021
    lin_vel_reward_scale = 4.10
    command_progress_reward_scale = 8.60
    command_progress_floor_reward_scale = -28.00
    min_command_progress_ratio = 0.965
    command_overspeed_reward_scale = -1.65

    yaw_rate_reward_scale = 3.95
    yaw_drift_reward_scale = -11.50
    yaw_drift_tolerance_rad = 0.032
    lateral_velocity_reward_scale = -3.90
    lateral_position_reward_scale = -4.85
    lateral_position_tolerance_m = 0.040

    rear_tibia_primary_axis_mirror_reward_scale = -0.025
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.060
    rear_tibia_primary_axis_mirror_tolerance = 0.130
    bl_lifted_tibia_primary_axis_mirror_tolerance = 0.100
    rear_tibia_secondary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_secondary_axis_mirror_reward_scale = 0.0

    bl_phase_delayed_tibia_primary_axis_mirror_reward_scale = -0.10
    bl_lifted_phase_delayed_tibia_primary_axis_mirror_reward_scale = -0.18
    bl_phase_delayed_tibia_secondary_axis_mirror_reward_scale = -0.14
    bl_lifted_phase_delayed_tibia_secondary_axis_mirror_reward_scale = -0.28
    bl_phase_delayed_tibia_primary_axis_mirror_tolerance = 0.095
    bl_lifted_phase_delayed_tibia_primary_axis_mirror_tolerance = 0.075
    bl_phase_delayed_tibia_secondary_axis_mirror_tolerance = 0.105
    bl_lifted_phase_delayed_tibia_secondary_axis_mirror_tolerance = 0.080


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV21PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV21EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV22EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV19EnvCfg):
    # V22 starts from V19's cleaner tibia-frame orientation and recovers speed
    # with a stricter progress floor. This tests whether the clean visual axis
    # can be preserved when backward velocity is no longer optional.
    lin_vel_tracking_sigma = 0.019
    lin_vel_reward_scale = 4.55
    command_progress_reward_scale = 10.20
    command_progress_floor_reward_scale = -36.00
    min_command_progress_ratio = 0.980
    command_overspeed_reward_scale = -1.85

    yaw_rate_reward_scale = 4.05
    yaw_drift_reward_scale = -12.25
    yaw_drift_tolerance_rad = 0.030
    lateral_velocity_reward_scale = -4.10
    lateral_position_reward_scale = -5.15
    lateral_position_tolerance_m = 0.038

    touchdown_stride_reward_scale = 5.20
    touchdown_short_stride_reward_scale = -3.80
    rapid_touchdown_reward_scale = -1.05

    rear_tibia_primary_axis_mirror_reward_scale = -0.055
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.145
    rear_tibia_secondary_axis_mirror_reward_scale = -0.055
    bl_lifted_tibia_secondary_axis_mirror_reward_scale = -0.220


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV22PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV22EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV23EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15EnvCfg):
    # V23 targets the rendered tibia mesh axes directly. The visible tibia mesh
    # is not aligned with local link Y, so V19's X/Y axis cleanup missed the
    # mostly-local-Z secondary visual axis that contributes to the inward curl.
    lin_vel_tracking_sigma = 0.022
    lin_vel_reward_scale = 3.85
    command_progress_reward_scale = 7.60
    command_progress_floor_reward_scale = -24.00
    min_command_progress_ratio = 0.955
    command_overspeed_reward_scale = -1.55

    yaw_rate_reward_scale = 3.85
    yaw_drift_reward_scale = -11.00
    yaw_drift_tolerance_rad = 0.032
    lateral_velocity_reward_scale = -3.80
    lateral_position_reward_scale = -4.75
    lateral_position_tolerance_m = 0.040

    rear_tibia_primary_axis_mirror_reward_scale = -0.020
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.050
    rear_tibia_primary_axis_mirror_tolerance = 0.130
    bl_lifted_tibia_primary_axis_mirror_tolerance = 0.100
    rear_tibia_secondary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_secondary_axis_mirror_reward_scale = 0.0

    rear_tibia_mesh_long_axis_mirror_reward_scale = -0.055
    bl_lifted_tibia_mesh_long_axis_mirror_reward_scale = -0.160
    rear_tibia_mesh_secondary_axis_mirror_reward_scale = -0.045
    bl_lifted_tibia_mesh_secondary_axis_mirror_reward_scale = -0.130
    rear_tibia_mesh_long_axis_mirror_tolerance = 0.085
    bl_lifted_tibia_mesh_long_axis_mirror_tolerance = 0.070
    rear_tibia_mesh_secondary_axis_mirror_tolerance = 0.070
    bl_lifted_tibia_mesh_secondary_axis_mirror_tolerance = 0.052


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV23PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV23EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV24EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V24 targets the visible rear tibia twist that remained in the accepted
    # 2615 gait. The cleaner V19 candidate reduced the lateral component of the
    # rear tibia mesh axes on both rear legs, but slowed down badly. This keeps
    # V3's gait stack and adds only a small visual-axis cap with a hard progress
    # floor so PPO cannot buy cleanliness by stopping.
    lin_vel_tracking_sigma = 0.020
    lin_vel_reward_scale = 4.30
    command_progress_reward_scale = 9.50
    command_progress_floor_reward_scale = -34.00
    min_command_progress_ratio = 0.970
    command_overspeed_reward_scale = -1.75

    yaw_rate_reward_scale = 4.00
    yaw_drift_reward_scale = -12.50
    yaw_drift_tolerance_rad = 0.030
    lateral_velocity_reward_scale = -4.00
    lateral_position_reward_scale = -5.20
    lateral_position_tolerance_m = 0.038

    rear_tibia_primary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_primary_axis_mirror_reward_scale = 0.0
    rear_tibia_secondary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_secondary_axis_mirror_reward_scale = 0.0
    rear_tibia_mesh_long_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_mesh_long_axis_mirror_reward_scale = 0.0
    rear_tibia_mesh_secondary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_mesh_secondary_axis_mirror_reward_scale = 0.0

    rear_tibia_mesh_long_lateral_excess_reward_scale = -0.045
    bl_swing_tibia_mesh_long_lateral_excess_reward_scale = -0.110
    rear_tibia_mesh_secondary_lateral_excess_reward_scale = -0.075
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = -0.190

    rear_tibia_mesh_long_lateral_cap = 0.275
    bl_swing_tibia_mesh_long_lateral_cap = 0.265
    rear_tibia_mesh_secondary_lateral_cap = 0.375
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.360
    rear_tibia_mesh_long_lateral_tolerance = 0.035
    bl_swing_tibia_mesh_long_lateral_tolerance = 0.025
    rear_tibia_mesh_secondary_lateral_tolerance = 0.035
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.025


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV24PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV24EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV25EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V25 replaces V24's one-sided visual cap with a bounded target for the
    # rear tibia mesh secondary-axis lateral component. The target is between
    # the accepted 2615 shape and the clean-but-slow V19 shape, so over-folded
    # stalled policies are penalized as well as inward visual curl.
    lin_vel_tracking_sigma = 0.020
    lin_vel_reward_scale = 4.35
    command_progress_reward_scale = 9.80
    command_progress_floor_reward_scale = -36.00
    min_command_progress_ratio = 0.972
    command_overspeed_reward_scale = -1.80

    yaw_rate_reward_scale = 4.05
    yaw_drift_reward_scale = -13.00
    yaw_drift_tolerance_rad = 0.030
    lateral_velocity_reward_scale = -4.05
    lateral_position_reward_scale = -5.35
    lateral_position_tolerance_m = 0.038

    rear_tibia_primary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_primary_axis_mirror_reward_scale = 0.0
    rear_tibia_secondary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_secondary_axis_mirror_reward_scale = 0.0
    rear_tibia_mesh_long_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_mesh_long_axis_mirror_reward_scale = 0.0
    rear_tibia_mesh_secondary_axis_mirror_reward_scale = 0.0
    bl_lifted_tibia_mesh_secondary_axis_mirror_reward_scale = 0.0

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.055
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.145
    rear_tibia_mesh_secondary_lateral_target = 0.355
    bl_swing_tibia_mesh_secondary_lateral_target = 0.350
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.055
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.045


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV25PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV25EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV26EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg):
    # V26 targets the actuator-space cause of the visible BL tibia curl. In the
    # accepted 2615 rollout, BL tibia mesh lateral curl is almost perfectly
    # anti-correlated with BL coxa angle during BL swing, with smaller femur and
    # tibia contributions. Apply this only when backward progress is already
    # near target speed so PPO cannot buy the cleanup by stopping.
    lin_vel_tracking_sigma = 0.020
    lin_vel_reward_scale = 4.50
    command_progress_reward_scale = 10.50
    command_progress_floor_reward_scale = -42.00
    min_command_progress_ratio = 0.978
    command_overspeed_reward_scale = -1.85

    yaw_rate_reward_scale = 4.10
    yaw_drift_reward_scale = -13.50
    yaw_drift_tolerance_rad = 0.030
    lateral_velocity_reward_scale = -4.15
    lateral_position_reward_scale = -5.45
    lateral_position_tolerance_m = 0.038

    bl_fast_swing_rear_joint_pose_reward_scale = -0.28
    visual_cleanup_min_progress_ratio = 0.92
    bl_fast_swing_coxa_min_rad = 0.020
    bl_fast_swing_femur_min_rad = -0.345
    bl_fast_swing_tibia_max_rad = 1.560
    bl_fast_swing_coxa_tolerance_rad = 0.045
    bl_fast_swing_femur_tolerance_rad = 0.060
    bl_fast_swing_tibia_tolerance_rad = 0.060

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.025
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.060
    rear_tibia_mesh_secondary_lateral_target = 0.370
    bl_swing_tibia_mesh_secondary_lateral_target = 0.360
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.070
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.060


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV26PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV26EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV27EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV26EnvCfg):
    # V27 makes the V26 correction actionable by applying the same
    # progress-gated BL swing target to the commanded joint positions. This is a
    # stronger but still localized nudge toward a less inward BL rear tibia.
    bl_fast_swing_rear_joint_pose_reward_scale = -0.55
    bl_fast_swing_rear_command_pose_reward_scale = -0.85
    visual_cleanup_min_progress_ratio = 0.90
    bl_fast_swing_coxa_min_rad = 0.030
    bl_fast_swing_femur_min_rad = -0.340
    bl_fast_swing_tibia_max_rad = 1.550
    bl_fast_swing_coxa_tolerance_rad = 0.040
    bl_fast_swing_femur_tolerance_rad = 0.055
    bl_fast_swing_tibia_tolerance_rad = 0.055

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.015
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.040
    rear_tibia_mesh_secondary_lateral_target = 0.365
    bl_swing_tibia_mesh_secondary_lateral_target = 0.355


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV27PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV27EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV28EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV27EnvCfg):
    # V28 recovers speed from the clean V27 2616 checkpoint. The visual cleanup
    # target is kept as a light guard while progress and straight backward travel
    # dominate the update.
    lin_vel_tracking_sigma = 0.018
    lin_vel_reward_scale = 5.20
    command_progress_reward_scale = 13.00
    command_progress_floor_reward_scale = -62.00
    min_command_progress_ratio = 0.985
    command_overspeed_reward_scale = -2.10

    yaw_rate_reward_scale = 4.45
    yaw_drift_reward_scale = -15.00
    yaw_drift_tolerance_rad = 0.028
    lateral_velocity_reward_scale = -4.50
    lateral_position_reward_scale = -5.80
    lateral_position_tolerance_m = 0.035

    bl_fast_swing_rear_joint_pose_reward_scale = -0.14
    bl_fast_swing_rear_command_pose_reward_scale = -0.22
    visual_cleanup_min_progress_ratio = 0.86
    bl_fast_swing_coxa_min_rad = 0.008
    bl_fast_swing_femur_min_rad = -0.355
    bl_fast_swing_tibia_max_rad = 1.570
    bl_fast_swing_coxa_tolerance_rad = 0.055
    bl_fast_swing_femur_tolerance_rad = 0.070
    bl_fast_swing_tibia_tolerance_rad = 0.070

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.008
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.018
    rear_tibia_mesh_secondary_lateral_target = 0.370
    bl_swing_tibia_mesh_secondary_lateral_target = 0.360


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV28PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV28EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV29EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV16EnvCfg):
    # V29 starts from the fast/symmetric V16 policy, then lowers the absolute
    # rear tibia mesh-secondary lateral component for both rear legs. This tests
    # whether symmetry can be preserved while making the pair look less inward.
    lin_vel_tracking_sigma = 0.019
    lin_vel_reward_scale = 4.80
    command_progress_reward_scale = 11.50
    command_progress_floor_reward_scale = -48.00
    min_command_progress_ratio = 0.978
    command_overspeed_reward_scale = -1.90

    yaw_rate_reward_scale = 4.20
    yaw_drift_reward_scale = -13.50
    yaw_drift_tolerance_rad = 0.030
    lateral_velocity_reward_scale = -4.20
    lateral_position_reward_scale = -5.45
    lateral_position_tolerance_m = 0.038

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.050
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.095
    rear_tibia_mesh_secondary_lateral_target = 0.390
    bl_swing_tibia_mesh_secondary_lateral_target = 0.380
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.060
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.055
    visual_cleanup_min_progress_ratio = 0.88

    rear_tibia_primary_axis_mirror_reward_scale = -0.020
    bl_lifted_tibia_primary_axis_mirror_reward_scale = -0.050
    rear_tibia_secondary_axis_mirror_reward_scale = -0.030
    bl_lifted_tibia_secondary_axis_mirror_reward_scale = -0.080


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV29PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV29EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV30EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV29EnvCfg):
    # V30 continues from V29's fast symmetric gait, but makes the absolute
    # inward visual cap stronger. This tests whether we can keep the fast
    # symmetric gait while moving both rear tibias away from the inward pose.
    lin_vel_tracking_sigma = 0.018
    lin_vel_reward_scale = 5.10
    command_progress_reward_scale = 12.75
    command_progress_floor_reward_scale = -58.00
    min_command_progress_ratio = 0.982

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.120
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.220
    rear_tibia_mesh_secondary_lateral_target = 0.385
    bl_swing_tibia_mesh_secondary_lateral_target = 0.375
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.050
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.045

    rear_tibia_mesh_secondary_lateral_excess_reward_scale = -0.080
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = -0.160
    rear_tibia_mesh_secondary_lateral_cap = 0.410
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.400
    rear_tibia_mesh_secondary_lateral_tolerance = 0.035
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.030


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV30PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV30EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV31EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV30EnvCfg):
    # V31 starts from V30's recovered-speed policy and adds a localized BL
    # swing-pose cleanup. The cleanup is progress-gated so the policy cannot
    # satisfy the visual target by slowing down.
    lin_vel_tracking_sigma = 0.017
    lin_vel_reward_scale = 5.45
    command_progress_reward_scale = 15.25
    command_progress_floor_reward_scale = -74.00
    min_command_progress_ratio = 0.988
    command_overspeed_reward_scale = -2.20

    yaw_rate_reward_scale = 4.55
    yaw_drift_reward_scale = -15.50
    yaw_drift_tolerance_rad = 0.027
    lateral_velocity_reward_scale = -4.70
    lateral_position_reward_scale = -6.00
    lateral_position_tolerance_m = 0.034

    bl_fast_swing_rear_joint_pose_reward_scale = -0.36
    bl_fast_swing_rear_command_pose_reward_scale = -0.62
    visual_cleanup_min_progress_ratio = 0.94
    bl_fast_swing_coxa_min_rad = 0.018
    bl_fast_swing_femur_min_rad = -0.335
    bl_fast_swing_tibia_max_rad = 1.545
    bl_fast_swing_coxa_tolerance_rad = 0.038
    bl_fast_swing_femur_tolerance_rad = 0.052
    bl_fast_swing_tibia_tolerance_rad = 0.052

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.105
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.260
    rear_tibia_mesh_secondary_lateral_target = 0.380
    bl_swing_tibia_mesh_secondary_lateral_target = 0.365
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.050
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.040

    rear_tibia_mesh_secondary_lateral_excess_reward_scale = -0.060
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = -0.190
    rear_tibia_mesh_secondary_lateral_cap = 0.405
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.390
    rear_tibia_mesh_secondary_lateral_tolerance = 0.035
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.028


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV31PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV31EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV32EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV31EnvCfg):
    # V32 recovers speed from V31's cleaner BL posture. Visual constraints are
    # kept as guards, while progress and straight travel dominate the update.
    lin_vel_tracking_sigma = 0.016
    lin_vel_reward_scale = 6.10
    command_progress_reward_scale = 18.50
    command_progress_floor_reward_scale = -96.00
    min_command_progress_ratio = 0.993
    command_overspeed_reward_scale = -2.55

    yaw_rate_reward_scale = 4.80
    yaw_drift_reward_scale = -16.50
    yaw_drift_tolerance_rad = 0.026
    lateral_velocity_reward_scale = -4.95
    lateral_position_reward_scale = -6.35
    lateral_position_tolerance_m = 0.032

    bl_fast_swing_rear_joint_pose_reward_scale = -0.18
    bl_fast_swing_rear_command_pose_reward_scale = -0.30
    visual_cleanup_min_progress_ratio = 0.90
    bl_fast_swing_coxa_min_rad = -0.005
    bl_fast_swing_femur_min_rad = -0.350
    bl_fast_swing_tibia_max_rad = 1.570
    bl_fast_swing_coxa_tolerance_rad = 0.052
    bl_fast_swing_femur_tolerance_rad = 0.065
    bl_fast_swing_tibia_tolerance_rad = 0.065

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.050
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.105
    rear_tibia_mesh_secondary_lateral_target = 0.392
    bl_swing_tibia_mesh_secondary_lateral_target = 0.382
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.060
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.052

    rear_tibia_mesh_secondary_lateral_excess_reward_scale = -0.035
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = -0.085
    rear_tibia_mesh_secondary_lateral_cap = 0.420
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.405
    rear_tibia_mesh_secondary_lateral_tolerance = 0.045
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.038


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV32PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV32EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV33EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV32EnvCfg):
    # V33 is a speed-retention pass from V32. The V32 posture correction fades
    # late in deterministic rollouts, so keep a light BL guard while pushing
    # progress, yaw, and lateral stability harder.
    lin_vel_tracking_sigma = 0.014
    lin_vel_reward_scale = 7.20
    command_progress_reward_scale = 23.00
    command_progress_floor_reward_scale = -132.00
    min_command_progress_ratio = 0.996
    command_overspeed_reward_scale = -2.90

    yaw_rate_reward_scale = 5.20
    yaw_drift_reward_scale = -20.00
    yaw_drift_tolerance_rad = 0.024
    lateral_velocity_reward_scale = -5.40
    lateral_position_reward_scale = -7.40
    lateral_position_tolerance_m = 0.029

    bl_fast_swing_rear_joint_pose_reward_scale = -0.10
    bl_fast_swing_rear_command_pose_reward_scale = -0.17
    visual_cleanup_min_progress_ratio = 0.86
    bl_fast_swing_coxa_min_rad = -0.018
    bl_fast_swing_femur_min_rad = -0.360
    bl_fast_swing_tibia_max_rad = 1.580
    bl_fast_swing_coxa_tolerance_rad = 0.065
    bl_fast_swing_femur_tolerance_rad = 0.078
    bl_fast_swing_tibia_tolerance_rad = 0.078

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.026
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.055
    rear_tibia_mesh_secondary_lateral_target = 0.398
    bl_swing_tibia_mesh_secondary_lateral_target = 0.388
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.075
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.065

    rear_tibia_mesh_secondary_lateral_excess_reward_scale = -0.020
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = -0.052
    rear_tibia_mesh_secondary_lateral_cap = 0.425
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.412
    rear_tibia_mesh_secondary_lateral_tolerance = 0.055
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.048


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV33PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV33EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV34EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV33EnvCfg):
    # V34 starts from V33's clean midpoint and targets the specific remaining
    # failure: BL stays clean but spends too little time in rear support, causing
    # a mid-rollout speed dip. Preserve the clean BL swing posture while
    # tightening rear contact balance and progress retention.
    lin_vel_tracking_sigma = 0.015
    lin_vel_reward_scale = 6.90
    command_progress_reward_scale = 21.00
    command_progress_floor_reward_scale = -112.00
    min_command_progress_ratio = 0.992
    command_overspeed_reward_scale = -2.70

    yaw_rate_reward_scale = 5.05
    yaw_drift_reward_scale = -18.50
    yaw_drift_tolerance_rad = 0.025
    lateral_velocity_reward_scale = -5.20
    lateral_position_reward_scale = -7.00
    lateral_position_tolerance_m = 0.030

    rear_contact_fraction_balance_reward_scale = -2.65
    rear_touchdown_rate_balance_reward_scale = -1.65
    rear_stance_duration_balance_reward_scale = -0.90
    rear_stance_slip_balance_reward_scale = -1.05
    rear_contact_fraction_balance_tolerance = 0.080

    bl_fast_swing_rear_joint_pose_reward_scale = -0.18
    bl_fast_swing_rear_command_pose_reward_scale = -0.34
    visual_cleanup_min_progress_ratio = 0.90
    bl_fast_swing_coxa_min_rad = -0.010
    bl_fast_swing_femur_min_rad = -0.345
    bl_fast_swing_tibia_max_rad = 1.560
    bl_fast_swing_coxa_tolerance_rad = 0.050
    bl_fast_swing_femur_tolerance_rad = 0.060
    bl_fast_swing_tibia_tolerance_rad = 0.060

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.070
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.160
    rear_tibia_mesh_secondary_lateral_target = 0.390
    bl_swing_tibia_mesh_secondary_lateral_target = 0.380
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.055
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.046

    rear_tibia_mesh_secondary_lateral_excess_reward_scale = -0.035
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = -0.090
    rear_tibia_mesh_secondary_lateral_cap = 0.415
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.400
    rear_tibia_mesh_secondary_lateral_tolerance = 0.042
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.034


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV34PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV34EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV35EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV33EnvCfg):
    # V35 uses the fast/clean V33 policy mix as a seed. It removes the scalar
    # secondary-axis target and instead uses a stricter BL swing cap plus a
    # mirrored mesh-axis guard, so speed cannot be recovered by curling BL inward.
    lin_vel_tracking_sigma = 0.014
    lin_vel_reward_scale = 7.35
    command_progress_reward_scale = 24.00
    command_progress_floor_reward_scale = -150.00
    min_command_progress_ratio = 0.995
    command_overspeed_reward_scale = -2.90

    yaw_rate_reward_scale = 5.25
    yaw_drift_reward_scale = -20.00
    yaw_drift_tolerance_rad = 0.024
    lateral_velocity_reward_scale = -5.45
    lateral_position_reward_scale = -7.45
    lateral_position_tolerance_m = 0.029

    rear_contact_fraction_balance_reward_scale = -0.35
    rear_touchdown_rate_balance_reward_scale = -0.25
    rear_stance_duration_balance_reward_scale = -0.14
    rear_stance_slip_balance_reward_scale = -0.35
    rear_contact_fraction_balance_tolerance = 0.13

    bl_fast_swing_rear_joint_pose_reward_scale = -0.14
    bl_fast_swing_rear_command_pose_reward_scale = -0.25
    visual_cleanup_min_progress_ratio = 0.88
    bl_fast_swing_coxa_min_rad = -0.014
    bl_fast_swing_femur_min_rad = -0.350
    bl_fast_swing_tibia_max_rad = 1.560
    bl_fast_swing_coxa_tolerance_rad = 0.052
    bl_fast_swing_femur_tolerance_rad = 0.063
    bl_fast_swing_tibia_tolerance_rad = 0.062

    rear_tibia_mesh_long_axis_mirror_reward_scale = -0.20
    bl_lifted_tibia_mesh_long_axis_mirror_reward_scale = -0.40
    rear_tibia_mesh_secondary_axis_mirror_reward_scale = -0.60
    bl_lifted_tibia_mesh_secondary_axis_mirror_reward_scale = -1.10
    rear_tibia_mesh_long_axis_mirror_tolerance = 0.095
    bl_lifted_tibia_mesh_long_axis_mirror_tolerance = 0.072
    rear_tibia_mesh_secondary_axis_mirror_tolerance = 0.070
    bl_lifted_tibia_mesh_secondary_axis_mirror_tolerance = 0.050

    rear_tibia_mesh_secondary_lateral_target_reward_scale = 0.0
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = 0.0
    rear_tibia_mesh_secondary_lateral_excess_reward_scale = -0.055
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = -0.220
    rear_tibia_mesh_secondary_lateral_cap = 0.405
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.390
    rear_tibia_mesh_secondary_lateral_tolerance = 0.040
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.027


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV35PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV35EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV36EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV35EnvCfg):
    # V36 starts from the very clean but late-stalling V35 checkpoint. Speed
    # and straight progress dominate again, while a relaxed BL cap prevents the
    # recovery pass from returning to the visible inward curl.
    lin_vel_tracking_sigma = 0.012
    lin_vel_reward_scale = 8.40
    command_progress_reward_scale = 31.00
    command_progress_floor_reward_scale = -235.00
    min_command_progress_ratio = 0.998
    command_overspeed_reward_scale = -3.30

    yaw_rate_reward_scale = 5.60
    yaw_drift_reward_scale = -22.50
    yaw_drift_tolerance_rad = 0.023
    lateral_velocity_reward_scale = -5.80
    lateral_position_reward_scale = -8.00
    lateral_position_tolerance_m = 0.027

    rear_contact_fraction_balance_reward_scale = -0.08
    rear_touchdown_rate_balance_reward_scale = -0.06
    rear_stance_duration_balance_reward_scale = -0.04
    rear_stance_slip_balance_reward_scale = -0.10
    rear_contact_fraction_balance_tolerance = 0.18

    bl_fast_swing_rear_joint_pose_reward_scale = -0.08
    bl_fast_swing_rear_command_pose_reward_scale = -0.14
    visual_cleanup_min_progress_ratio = 0.92
    bl_fast_swing_coxa_min_rad = -0.006
    bl_fast_swing_femur_min_rad = -0.342
    bl_fast_swing_tibia_max_rad = 1.565
    bl_fast_swing_coxa_tolerance_rad = 0.060
    bl_fast_swing_femur_tolerance_rad = 0.074
    bl_fast_swing_tibia_tolerance_rad = 0.072

    rear_tibia_mesh_long_axis_mirror_reward_scale = -0.12
    bl_lifted_tibia_mesh_long_axis_mirror_reward_scale = -0.24
    rear_tibia_mesh_secondary_axis_mirror_reward_scale = -0.36
    bl_lifted_tibia_mesh_secondary_axis_mirror_reward_scale = -0.70

    rear_tibia_mesh_secondary_lateral_excess_reward_scale = -0.035
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = -0.145
    rear_tibia_mesh_secondary_lateral_cap = 0.415
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.398
    rear_tibia_mesh_secondary_lateral_tolerance = 0.050
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.036


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV36PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV36EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV37EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV36EnvCfg):
    # V37 removes the one-sided BL pose nudge and instead pushes both rear tibia
    # mesh-secondary axes toward the same visual target. This should reduce the
    # BL/BR mismatch without letting PPO recover speed by keeping BR as the old
    # curled reference.
    lin_vel_tracking_sigma = 0.012
    lin_vel_reward_scale = 8.70
    command_progress_reward_scale = 33.00
    command_progress_floor_reward_scale = -255.00
    min_command_progress_ratio = 0.998
    command_overspeed_reward_scale = -3.35

    yaw_rate_reward_scale = 5.80
    yaw_drift_reward_scale = -24.00
    yaw_drift_tolerance_rad = 0.022
    lateral_velocity_reward_scale = -6.00
    lateral_position_reward_scale = -8.35
    lateral_position_tolerance_m = 0.026

    rear_contact_fraction_balance_reward_scale = -0.40
    rear_touchdown_rate_balance_reward_scale = -0.22
    rear_stance_duration_balance_reward_scale = -0.16
    rear_stance_slip_balance_reward_scale = -0.32
    rear_contact_fraction_balance_tolerance = 0.115

    bl_fast_swing_rear_joint_pose_reward_scale = 0.0
    bl_fast_swing_rear_command_pose_reward_scale = 0.0

    rear_tibia_mesh_long_axis_mirror_reward_scale = -0.08
    bl_lifted_tibia_mesh_long_axis_mirror_reward_scale = -0.12
    rear_tibia_mesh_secondary_axis_mirror_reward_scale = -0.12
    bl_lifted_tibia_mesh_secondary_axis_mirror_reward_scale = -0.20
    rear_tibia_mesh_secondary_axis_mirror_tolerance = 0.085
    bl_lifted_tibia_mesh_secondary_axis_mirror_tolerance = 0.062

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.42
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.18
    rear_tibia_mesh_secondary_lateral_target = 0.385
    bl_swing_tibia_mesh_secondary_lateral_target = 0.380
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.045
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.040

    rear_tibia_mesh_secondary_lateral_excess_reward_scale = -0.16
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = -0.10
    rear_tibia_mesh_secondary_lateral_cap = 0.430
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.410
    rear_tibia_mesh_secondary_lateral_tolerance = 0.038
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.034
    visual_cleanup_min_progress_ratio = 0.94


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV37PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV37EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV38EnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV37EnvCfg):
    # V38 keeps V37's shared visual-axis cleanup but adds an explicit BR swing
    # target. V37 proved the all-phase target alone preserves speed but leaves
    # BR high during swing, so this makes the measured asymmetric phase
    # actionable without adding a one-sided BL postprocess.
    lin_vel_tracking_sigma = 0.012
    lin_vel_reward_scale = 8.85
    command_progress_reward_scale = 34.00
    command_progress_floor_reward_scale = -265.00
    min_command_progress_ratio = 0.998

    yaw_rate_reward_scale = 5.90
    yaw_drift_reward_scale = -24.50
    yaw_drift_tolerance_rad = 0.022
    lateral_velocity_reward_scale = -6.05
    lateral_position_reward_scale = -8.50
    lateral_position_tolerance_m = 0.026

    rear_tibia_mesh_secondary_lateral_target_reward_scale = -0.24
    bl_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.10
    br_swing_tibia_mesh_secondary_lateral_target_reward_scale = -0.70
    rear_tibia_mesh_secondary_lateral_target = 0.405
    bl_swing_tibia_mesh_secondary_lateral_target = 0.395
    br_swing_tibia_mesh_secondary_lateral_target = 0.405
    rear_tibia_mesh_secondary_lateral_target_tolerance = 0.065
    bl_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.055
    br_swing_tibia_mesh_secondary_lateral_target_tolerance = 0.055

    rear_tibia_mesh_secondary_lateral_excess_reward_scale = -0.10
    bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale = -0.08
    rear_tibia_mesh_secondary_lateral_cap = 0.455
    bl_swing_tibia_mesh_secondary_lateral_cap = 0.425
    rear_tibia_mesh_secondary_lateral_tolerance = 0.050
    bl_swing_tibia_mesh_secondary_lateral_tolerance = 0.040
    visual_cleanup_min_progress_ratio = 0.93


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV38PlayEnvCfg(InsectoidMiniQuadBackwardBLTibiaLinkBalanceV38EnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadRearSymmetryEnvCfg(InsectoidMiniQuadGaitRefineEnvCfg):
    touchdown_cadence_reward_scale = -0.75
    rapid_touchdown_reward_scale = -0.85
    touchdown_stride_reward_scale = 5.0
    touchdown_short_stride_reward_scale = -3.8
    stance_anchor_slip_reward_scale = -1.05

    rear_stride_balance_reward_scale = -0.35
    rear_stance_duration_balance_reward_scale = -0.25
    rear_touchdown_rate_balance_reward_scale = -0.18
    rear_contact_fraction_balance_reward_scale = -0.08
    rear_stance_slip_balance_reward_scale = -0.55
    rear_left_stance_slip_excess_reward_scale = -0.45
    rear_lateral_mirror_reward_scale = -0.08


@configclass
class InsectoidMiniQuadRearSymmetryPlayEnvCfg(InsectoidMiniQuadRearSymmetryEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadBLTibiaPostureEnvCfg(InsectoidMiniQuadGaitRefineEnvCfg):
    touchdown_cadence_reward_scale = -0.80
    rapid_touchdown_reward_scale = -0.90
    touchdown_stride_reward_scale = 5.0
    touchdown_short_stride_reward_scale = -3.8
    stance_anchor_slip_reward_scale = -1.05

    rear_stride_balance_reward_scale = -0.18
    rear_stance_duration_balance_reward_scale = -0.12
    rear_touchdown_rate_balance_reward_scale = -0.10
    rear_contact_fraction_balance_reward_scale = -0.04
    rear_stance_slip_balance_reward_scale = -0.30
    rear_left_stance_slip_excess_reward_scale = -0.35
    rear_lateral_mirror_reward_scale = -0.04

    bl_tibia_tuck_reward_scale = -0.75
    bl_tibia_pair_offset_reward_scale = -0.65


@configclass
class InsectoidMiniQuadBLTibiaPosturePlayEnvCfg(InsectoidMiniQuadBLTibiaPostureEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


@configclass
class InsectoidMiniQuadRearTibiaSymmetryEnvCfg(InsectoidMiniQuadGaitRefineEnvCfg):
    # Keep the 1798 cadence/stride objective as the dominant behavior and add
    # only phase-tolerant rear-leg posture shaping.
    touchdown_cadence_reward_scale = -0.55
    rapid_touchdown_reward_scale = -0.65
    touchdown_stride_reward_scale = 4.5
    touchdown_short_stride_reward_scale = -3.2
    stance_anchor_slip_reward_scale = -0.95

    rear_stride_balance_reward_scale = -0.04
    rear_stance_duration_balance_reward_scale = -0.03
    rear_touchdown_rate_balance_reward_scale = -0.02
    rear_contact_fraction_balance_reward_scale = -0.01
    rear_stance_slip_balance_reward_scale = -0.04
    rear_left_stance_slip_excess_reward_scale = -0.03
    rear_lateral_mirror_reward_scale = 0.0

    # This permits both rear tibias to curl inward if that is part of the
    # stable gait. It penalizes a persistent left/right mean mismatch, guides
    # both rear tibias toward the same moderate curl, and bounds excessive tuck.
    rear_tibia_mean_symmetry_reward_scale = -3.00
    rear_tibia_mean_target_reward_scale = -2.50
    rear_tibia_mean_curl_floor_reward_scale = 0.00
    rear_tibia_command_symmetry_reward_scale = -1.50
    rear_tibia_command_target_reward_scale = -3.00
    rear_tibia_command_curl_floor_reward_scale = 0.00
    rear_tibia_tuck_envelope_reward_scale = -0.50
    rear_tibia_mean_symmetry_tolerance_rad = 0.15
    rear_tibia_mean_target_rad = 2.05
    rear_tibia_mean_target_tolerance_rad = 0.20
    rear_tibia_curl_floor_rad = 1.90
    rear_tibia_curl_floor_tolerance_rad = 0.12
    rear_tibia_command_symmetry_tolerance_rad = 0.10
    rear_tibia_command_target_tolerance_rad = 0.18
    rear_tibia_tuck_limit_rad = 2.22
    rear_tibia_tuck_tolerance_rad = 0.20


@configclass
class InsectoidMiniQuadRearTibiaSymmetryPlayEnvCfg(InsectoidMiniQuadRearTibiaSymmetryEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)


class InsectoidMiniQuadEnv(DirectRLEnv):
    cfg: InsectoidMiniQuadFlatEnvCfg

    def __init__(self, cfg: InsectoidMiniQuadFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = self._robot.data.default_joint_pos.clone()
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._initial_yaw_w = torch.zeros(self.num_envs, device=self.device)
        self._initial_root_xy_w = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_foot_pos_b = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._base_lin_vel_est_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._base_lin_vel_est_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._base_lin_vel_est_last_step = -1

        self._stance_anchor_xy = torch.zeros(self.num_envs, 4, 2, device=self.device)
        self._last_touchdown_xy = torch.zeros(self.num_envs, 4, 2, device=self.device)
        self._has_touchdown = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._latest_touchdown_stride = torch.zeros(self.num_envs, 4, device=self.device)
        self._stride_length_ema = torch.zeros(self.num_envs, 4, device=self.device)
        self._stance_time = torch.zeros(self.num_envs, 4, device=self.device)
        self._swing_time = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_foot_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._step_counts = torch.zeros(self.num_envs, 4, device=self.device)
        self._forward_distance_traveled = torch.zeros(self.num_envs, device=self.device)
        self._lateral_distance_abs = torch.zeros(self.num_envs, device=self.device)
        self._command_forward_sum = torch.zeros(self.num_envs, device=self.device)
        self._total_time = torch.zeros(self.num_envs, device=self.device)
        self._roll_rate_abs_accum = torch.zeros(self.num_envs, device=self.device)
        self._pitch_rate_abs_accum = torch.zeros(self.num_envs, device=self.device)
        self._yaw_rate_abs_accum = torch.zeros(self.num_envs, device=self.device)
        self._tilt_accum = torch.zeros(self.num_envs, device=self.device)
        self._front_up_deg_accum = torch.zeros(self.num_envs, device=self.device)
        self._projected_gravity_y_accum = torch.zeros(self.num_envs, device=self.device)
        self._base_height_accum = torch.zeros(self.num_envs, device=self.device)
        self._foot_contact_count_accum = torch.zeros(self.num_envs, device=self.device)
        self._undesired_contact_count_accum = torch.zeros(self.num_envs, device=self.device)
        self._stance_foot_speed_accum = torch.zeros(self.num_envs, device=self.device)
        self._low_swing_foot_speed_accum = torch.zeros(self.num_envs, device=self.device)
        self._per_foot_contact_time_accum = torch.zeros(self.num_envs, 4, device=self.device)
        self._per_foot_stance_slip_accum = torch.zeros(self.num_envs, 4, device=self.device)
        self._per_foot_low_swing_time_accum = torch.zeros(self.num_envs, 4, device=self.device)
        self._per_foot_low_swing_drag_accum = torch.zeros(self.num_envs, 4, device=self.device)
        self._completed_stance_time_accum = torch.zeros(self.num_envs, 4, device=self.device)
        self._completed_stance_count = torch.zeros(self.num_envs, 4, device=self.device)
        self._completed_swing_time_accum = torch.zeros(self.num_envs, 4, device=self.device)
        self._completed_swing_count = torch.zeros(self.num_envs, 4, device=self.device)
        self._bl_tibia_accum = torch.zeros(self.num_envs, device=self.device)
        self._br_tibia_accum = torch.zeros(self.num_envs, device=self.device)
        self._bl_br_tibia_offset_accum = torch.zeros(self.num_envs, device=self.device)
        self._rear_tibia_link_mirror_accum = torch.zeros(self.num_envs, device=self.device)
        self._rear_tibia_segment_mirror_accum = torch.zeros(self.num_envs, device=self.device)
        self._rear_tibia_segment_outward_accum = torch.zeros(self.num_envs, device=self.device)
        self._rear_femur_tibia_segment_mirror_accum = torch.zeros(self.num_envs, device=self.device)
        self._rear_femur_tibia_z_mirror_accum = torch.zeros(self.num_envs, device=self.device)
        self._rear_coxa_joint_abs_diff_accum = torch.zeros(self.num_envs, device=self.device)
        self._rear_femur_joint_abs_diff_accum = torch.zeros(self.num_envs, device=self.device)
        self._rear_tibia_joint_abs_diff_accum = torch.zeros(self.num_envs, device=self.device)
        self._rear_swing_outward_accum = torch.zeros(self.num_envs, 2, device=self.device)
        self._rear_swing_outward_time = torch.zeros(self.num_envs, 2, device=self.device)
        self._rear_phase_history_len = int(self.cfg.rear_phase_symmetry_delay_steps) + 1
        self._br_tibia_segment_outward_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, device=self.device
        )
        self._br_tibia_link_outward_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, device=self.device
        )
        self._br_tibia_link_pos_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, 3, device=self.device
        )
        self._br_femur_tibia_vec_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, 3, device=self.device
        )
        self._br_tibia_axis_x_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, 3, device=self.device
        )
        self._br_tibia_axis_y_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, 3, device=self.device
        )
        self._br_rear_coxa_joint_pos_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, device=self.device
        )
        self._br_rear_coxa_target_pos_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, device=self.device
        )
        self._br_rear_distal_joint_pos_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, 2, device=self.device
        )
        self._br_rear_distal_target_pos_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, 2, device=self.device
        )
        self._br_foot_height_history = torch.zeros(
            self.num_envs, self._rear_phase_history_len, device=self.device
        )
        self._br_swing_history = torch.zeros(self.num_envs, self._rear_phase_history_len, device=self.device)
        self._rear_phase_history_steps = torch.zeros(self.num_envs, device=self.device)

        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._feet_ids, feet_names = self._contact_sensor.find_bodies(FOOT_BODY_PATTERN, preserve_order=True)
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(NON_FOOT_BODY_PATTERN)
        self._feet_body_ids, _ = self._robot.find_bodies(FOOT_BODY_PATTERN, preserve_order=True)
        self._reset_joint_pos_offset = torch.zeros(
            self._robot.data.default_joint_pos.shape[1], device=self.device
        )
        for joint_name, offset in self.cfg.reset_joint_pos_offsets.items():
            joint_ids, _ = self._robot.find_joints([joint_name], preserve_order=True)
            if len(joint_ids) != 1:
                raise RuntimeError(f"Expected one reset-offset joint for {joint_name}, got {joint_ids}.")
            self._reset_joint_pos_offset[joint_ids[0]] = float(offset)
        self._action_center_offset = torch.zeros_like(self._reset_joint_pos_offset)
        for joint_name, offset in self.cfg.action_center_offsets.items():
            joint_ids, _ = self._robot.find_joints([joint_name], preserve_order=True)
            if len(joint_ids) != 1:
                raise RuntimeError(f"Expected one action-center joint for {joint_name}, got {joint_ids}.")
            self._action_center_offset[joint_ids[0]] = float(offset)
        self._action_residual_lower = torch.full_like(self._reset_joint_pos_offset, -1.0)
        self._action_residual_upper = torch.full_like(self._reset_joint_pos_offset, 1.0)
        for joint_name, limits in self.cfg.action_residual_limits.items():
            joint_ids, _ = self._robot.find_joints([joint_name], preserve_order=True)
            if len(joint_ids) != 1:
                raise RuntimeError(f"Expected one action-limit joint for {joint_name}, got {joint_ids}.")
            if len(limits) != 2:
                raise RuntimeError(f"Expected lower/upper action residual limits for {joint_name}, got {limits}.")
            self._action_residual_lower[joint_ids[0]] = float(limits[0])
            self._action_residual_upper[joint_ids[0]] = float(limits[1])
        front_up_posture_joint_ids: list[int] = []
        front_up_posture_offsets: list[float] = []
        for joint_name, offset in self.cfg.front_up_posture_joint_offsets.items():
            joint_ids, _ = self._robot.find_joints([joint_name], preserve_order=True)
            if len(joint_ids) != 1:
                raise RuntimeError(f"Expected one front-up posture joint for {joint_name}, got {joint_ids}.")
            front_up_posture_joint_ids.append(joint_ids[0])
            front_up_posture_offsets.append(float(offset))
        self._front_up_posture_joint_ids = torch.tensor(
            front_up_posture_joint_ids, dtype=torch.long, device=self.device
        )
        self._front_up_posture_offsets = torch.tensor(
            front_up_posture_offsets, dtype=torch.float32, device=self.device
        )
        rear_femur_body_ids, rear_femur_body_names = self._robot.find_bodies(
            f"{BL_FEMUR_BODY_NAME}|{BR_FEMUR_BODY_NAME}", preserve_order=True
        )
        rear_tibia_body_ids, rear_tibia_body_names = self._robot.find_bodies(
            f"{BL_TIBIA_BODY_NAME}|{BR_TIBIA_BODY_NAME}", preserve_order=True
        )
        bl_coxa_ids, _ = self._robot.find_joints([BL_COXA_JOINT_NAME], preserve_order=True)
        br_coxa_ids, _ = self._robot.find_joints([BR_COXA_JOINT_NAME], preserve_order=True)
        bl_femur_ids, _ = self._robot.find_joints([BL_FEMUR_JOINT_NAME], preserve_order=True)
        br_femur_ids, _ = self._robot.find_joints([BR_FEMUR_JOINT_NAME], preserve_order=True)
        bl_tibia_ids, _ = self._robot.find_joints([BL_TIBIA_JOINT_NAME], preserve_order=True)
        br_tibia_ids, _ = self._robot.find_joints([BR_TIBIA_JOINT_NAME], preserve_order=True)
        self._bl_coxa_joint_id = bl_coxa_ids[0]
        self._br_coxa_joint_id = br_coxa_ids[0]
        self._bl_femur_joint_id = bl_femur_ids[0]
        self._br_femur_joint_id = br_femur_ids[0]
        self._bl_tibia_joint_id = bl_tibia_ids[0]
        self._br_tibia_joint_id = br_tibia_ids[0]
        self._rear_femur_body_ids = rear_femur_body_ids
        self._rear_tibia_body_ids = rear_tibia_body_ids

        if tuple(feet_names) != FOOT_BODY_NAMES:
            raise RuntimeError(f"Unexpected quad foot order {feet_names}; expected {FOOT_BODY_NAMES}.")
        if tuple(rear_femur_body_names) != (BL_FEMUR_BODY_NAME, BR_FEMUR_BODY_NAME):
            raise RuntimeError(
                f"Unexpected rear femur body order {rear_femur_body_names}; "
                f"expected {(BL_FEMUR_BODY_NAME, BR_FEMUR_BODY_NAME)}."
            )
        if tuple(rear_tibia_body_names) != (BL_TIBIA_BODY_NAME, BR_TIBIA_BODY_NAME):
            raise RuntimeError(
                f"Unexpected rear tibia body order {rear_tibia_body_names}; "
                f"expected {(BL_TIBIA_BODY_NAME, BR_TIBIA_BODY_NAME)}."
            )

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "command_progress",
                "command_progress_floor",
                "track_ang_vel_z_exp",
                "yaw_drift_l2",
                "lateral_velocity_l2",
                "lateral_position_l2",
                "command_overspeed_l2",
                "lin_vel_z_l2",
                "base_height_floor_l2",
                "ang_vel_xy_l2",
                "ang_vel_z_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "pitch_forward_l2",
                "front_up_target_l2",
                "front_down_hinge_l2",
                "front_up_progress_bonus",
                "front_up_posture",
                "front_up_posture_command",
                "stance_foot_slip",
                "low_swing_drag",
                "bl_low_swing_drag",
                "bl_low_swing_time",
                "swing_foot_speed_l2",
                "support_distribution",
                "contact_balance",
                "diagonal_trot",
                "diagonal_pair_sync",
                "side_antiphase",
                "rear_contact_antiphase",
                "touchdown_cadence_l2",
                "touchdown_rate_balance",
                "rapid_touchdown",
                "touchdown_stride",
                "touchdown_short_stride",
                "stride_ema_shortfall",
                "stance_anchor_slip",
                "rear_stride_balance",
                "rear_stance_duration_balance",
                "rear_touchdown_rate_balance",
                "rear_contact_fraction_balance",
                "rear_contact_fraction_target",
                "bl_contact_fraction_excess",
                "rear_stance_slip_balance",
                "rear_left_stance_slip_excess",
                "rear_lateral_mirror",
                "rear_tibia_mean_symmetry",
                "rear_tibia_mean_target",
                "rear_tibia_mean_curl_floor",
                "rear_tibia_command_symmetry",
                "rear_tibia_command_target",
                "rear_tibia_command_curl_floor",
                "rear_tibia_tuck_envelope",
                "bl_tibia_tuck",
                "bl_tibia_pair_offset",
                "bl_coxa_pair_offset",
                "bl_femur_pair_offset",
                "bl_coxa_command_pair_offset",
                "bl_femur_command_pair_offset",
                "bl_tibia_command_pair_offset",
                "rear_coxa_joint_symmetry",
                "rear_femur_joint_symmetry",
                "rear_tibia_joint_symmetry",
                "rear_coxa_command_symmetry",
                "rear_femur_command_symmetry",
                "rear_tibia_command_symmetry_abs",
                "rear_femur_tibia_segment_mirror",
                "rear_femur_tibia_z_mirror",
                "rear_tibia_link_mirror",
                "rear_tibia_axis_mirror",
                "bl_lifted_tibia_axis_mirror",
                "rear_tibia_primary_axis_mirror",
                "bl_lifted_tibia_primary_axis_mirror",
                "rear_tibia_secondary_axis_mirror",
                "bl_lifted_tibia_secondary_axis_mirror",
                "rear_tibia_mesh_long_axis_mirror",
                "bl_lifted_tibia_mesh_long_axis_mirror",
                "rear_tibia_mesh_secondary_axis_mirror",
                "bl_lifted_tibia_mesh_secondary_axis_mirror",
                "rear_tibia_mesh_long_lateral_excess",
                "bl_swing_tibia_mesh_long_lateral_excess",
                "rear_tibia_mesh_secondary_lateral_excess",
                "bl_swing_tibia_mesh_secondary_lateral_excess",
                "rear_tibia_mesh_secondary_lateral_target",
                "bl_swing_tibia_mesh_secondary_lateral_target",
                "br_swing_tibia_mesh_secondary_lateral_target",
                "bl_fast_swing_rear_joint_pose",
                "bl_fast_swing_rear_command_pose",
                "bl_tibia_link_outward_vs_br",
                "bl_lifted_swing_tibia_link_outward_vs_br",
                "bl_tibia_link_outward_floor",
                "bl_lifted_swing_tibia_link_outward_floor",
                "rear_tibia_segment_mirror",
                "rear_tibia_segment_outward",
                "rear_tibia_segment_outward_target",
                "rear_swing_tibia_segment_outward",
                "bl_swing_tibia_segment_outward",
                "bl_lifted_swing_tibia_segment_outward",
                "bl_phase_delayed_tibia_outward",
                "bl_phase_delayed_tibia_link_outward",
                "bl_lifted_phase_delayed_tibia_link_outward",
                "bl_phase_delayed_tibia_link_mirror",
                "bl_lifted_phase_delayed_tibia_link_mirror",
                "bl_phase_delayed_femur_tibia_mirror",
                "bl_lifted_phase_delayed_femur_tibia_mirror",
                "bl_phase_delayed_tibia_primary_axis_mirror",
                "bl_lifted_phase_delayed_tibia_primary_axis_mirror",
                "bl_phase_delayed_tibia_secondary_axis_mirror",
                "bl_lifted_phase_delayed_tibia_secondary_axis_mirror",
                "bl_phase_delayed_rear_coxa_joint_symmetry",
                "bl_lifted_phase_delayed_rear_coxa_joint_symmetry",
                "bl_phase_delayed_rear_coxa_command_symmetry",
                "bl_lifted_phase_delayed_rear_coxa_command_symmetry",
                "bl_phase_delayed_rear_distal_joint_symmetry",
                "bl_lifted_phase_delayed_rear_distal_joint_symmetry",
                "bl_phase_delayed_rear_distal_command_symmetry",
                "bl_lifted_phase_delayed_rear_distal_command_symmetry",
                "bl_phase_delayed_foot_height",
                "rear_swing_outward_mean_balance",
                "bl_swing_outward_mean_floor",
                "bl_swing_outward_vs_br",
                "bl_lifted_swing_outward_vs_br",
                "rear_swing_femur_floor",
                "bl_swing_femur_floor",
                "rear_swing_femur_command_floor",
                "bl_swing_femur_command_floor",
                "swing_height",
                "bl_swing_height",
                "short_stance",
                "short_swing",
            ]
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clamp(
            actions,
            self._action_residual_lower.unsqueeze(0),
            self._action_residual_upper.unsqueeze(0),
        )
        target = (
            self.cfg.action_scale * self._actions
            + self._robot.data.default_joint_pos
            + self._action_center_offset.unsqueeze(0)
        )
        alpha = self.cfg.action_target_filter_alpha
        self._processed_actions = alpha * target + (1.0 - alpha) * self._processed_actions

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_foot_positions_b(self) -> torch.Tensor:
        foot_pos_w = self._robot.data.body_pos_w[:, self._feet_body_ids, :]
        root_pos_w = self._robot.data.root_pos_w.unsqueeze(1)
        foot_pos_rel_w = foot_pos_w - root_pos_w
        root_quat_w = self._robot.data.root_quat_w.unsqueeze(1).expand(-1, len(self._feet_body_ids), -1)
        return quat_apply_inverse(
            root_quat_w.reshape(-1, 4),
            foot_pos_rel_w.reshape(-1, 3),
        ).reshape(self.num_envs, len(self._feet_body_ids), 3)

    def _get_base_linear_velocity_observation(self, foot_contacts: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_privileged_base_lin_vel_obs:
            base_lin_vel_obs = self._robot.data.root_lin_vel_b.clone()
        elif self.cfg.use_foot_kinematic_base_lin_vel_obs:
            current_step = int(getattr(self, "common_step_counter", torch.max(self.episode_length_buf).item()))
            if self._base_lin_vel_est_last_step != current_step:
                foot_pos_b = self._get_foot_positions_b()
                uninitialized = ~self._base_lin_vel_est_initialized
                if torch.any(uninitialized):
                    self._prev_foot_pos_b[uninitialized] = foot_pos_b[uninitialized]
                    self._base_lin_vel_est_b[uninitialized] = 0.0
                    self._base_lin_vel_est_initialized[uninitialized] = True

                foot_pos_delta_b = foot_pos_b - self._prev_foot_pos_b
                foot_vel_b = foot_pos_delta_b / self.step_dt
                angular_foot_vel_b = torch.cross(
                    self._robot.data.root_ang_vel_b.unsqueeze(1).expand_as(foot_pos_b),
                    foot_pos_b,
                    dim=2,
                )
                per_foot_base_vel_b = -(foot_vel_b + angular_foot_vel_b)

                contact_f = foot_contacts.float()
                contact_count = torch.sum(contact_f, dim=1, keepdim=True)
                measured_base_vel_b = torch.sum(per_foot_base_vel_b * contact_f.unsqueeze(-1), dim=1) / torch.clamp(
                    contact_count, min=1.0
                )
                measured_base_vel_b[:, 2] = 0.0
                measured_base_vel_b = torch.clamp(
                    measured_base_vel_b,
                    min=-self.cfg.base_lin_vel_obs_max_abs,
                    max=self.cfg.base_lin_vel_obs_max_abs,
                )

                smoothing = self.cfg.base_lin_vel_estimator_smoothing
                updated_estimate = smoothing * self._base_lin_vel_est_b + (1.0 - smoothing) * measured_base_vel_b
                no_contact_estimate = 0.98 * self._base_lin_vel_est_b
                self._base_lin_vel_est_b = torch.where(contact_count > 0.0, updated_estimate, no_contact_estimate)
                self._prev_foot_pos_b = foot_pos_b
                self._base_lin_vel_est_last_step = current_step
            base_lin_vel_obs = self._base_lin_vel_est_b.clone()
        else:
            base_lin_vel_obs = torch.zeros(self.num_envs, 3, device=self.device)

        if self.cfg.zero_base_lin_vel_z_obs:
            base_lin_vel_obs[:, 2] = 0.0
        return base_lin_vel_obs

    def _add_observation_noise(self, value: torch.Tensor, std: float) -> torch.Tensor:
        if not self.cfg.observation_noise_enabled or std <= 0.0:
            return value
        return value + torch.randn_like(value) * std

    def _get_open_loop_gait_phase_observations(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = (self.episode_length_buf.float() * self.step_dt / self.cfg.open_loop_gait_period_s) % 1.0
        # Feet are ordered BL, BR, ML, MR. Use diagonal-pair timing: BL/MR
        # together and BR/ML half a cycle later.
        offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), device=self.device).unsqueeze(0)
        foot_phase = torch.remainder(phase.unsqueeze(1) + offsets, 1.0)
        duty = float(max(0.05, min(0.95, self.cfg.open_loop_gait_duty_factor)))
        contact = (foot_phase < duty).float()
        stance = torch.where(contact > 0.0, foot_phase / duty, torch.zeros_like(foot_phase))
        swing = torch.where(contact <= 0.0, (foot_phase - duty) / (1.0 - duty), torch.zeros_like(foot_phase))
        return contact, stance, swing

    def _get_observations(self) -> dict:
        foot_contact_f = self._get_foot_contacts().float()
        stance_time_obs = torch.clamp(self._stance_time / self.cfg.min_stance_time, 0.0, 1.0)
        swing_time_obs = torch.clamp(self._swing_time / self.cfg.min_swing_time, 0.0, 1.0)
        base_lin_vel_obs = self._add_observation_noise(
            self._get_base_linear_velocity_observation(foot_contact_f.bool()),
            self.cfg.base_lin_vel_obs_noise_std,
        )
        if self.cfg.use_rpy_attitude_obs:
            rpy_obs = _rpy_from_quat_wxyz(self._robot.data.root_quat_w)
            rpy_obs[:, 2] = _wrap_to_pi(rpy_obs[:, 2] - self._initial_yaw_w)
            angular_velocity_obs = self._add_observation_noise(rpy_obs, self.cfg.rpy_attitude_obs_noise_std)
        else:
            angular_velocity_obs = self._add_observation_noise(
                self._robot.data.root_ang_vel_b,
                self.cfg.angular_velocity_obs_noise_std,
            )
        projected_gravity_obs = self._add_observation_noise(
            self._robot.data.projected_gravity_b,
            self.cfg.projected_gravity_obs_noise_std,
        )
        joint_position_obs = self._add_observation_noise(
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self.cfg.joint_position_obs_noise_std,
        )
        joint_velocity_obs = self._add_observation_noise(
            self._robot.data.joint_vel,
            self.cfg.joint_velocity_obs_noise_std,
        )
        action_obs = self._add_observation_noise(self._actions, self.cfg.action_obs_noise_std)
        if self.cfg.use_open_loop_gait_phase_obs:
            foot_contact_f, stance_time_obs, swing_time_obs = self._get_open_loop_gait_phase_observations()
        if self.cfg.zero_contact_obs:
            foot_contact_f = torch.zeros_like(foot_contact_f)
        if self.cfg.observation_noise_enabled and self.cfg.contact_obs_dropout_prob > 0.0:
            contact_keep = torch.rand_like(foot_contact_f) > self.cfg.contact_obs_dropout_prob
            foot_contact_f = foot_contact_f * contact_keep.float()
        if self.cfg.zero_contact_timer_obs:
            stance_time_obs = torch.zeros_like(stance_time_obs)
            swing_time_obs = torch.zeros_like(swing_time_obs)
        else:
            stance_time_obs = torch.clamp(
                self._add_observation_noise(stance_time_obs, self.cfg.timer_obs_noise_std),
                0.0,
                1.0,
            )
            swing_time_obs = torch.clamp(
                self._add_observation_noise(swing_time_obs, self.cfg.timer_obs_noise_std),
                0.0,
                1.0,
            )
        obs = torch.cat(
            (
                base_lin_vel_obs,
                angular_velocity_obs,
                projected_gravity_obs,
                self._commands,
                joint_position_obs,
                joint_velocity_obs,
                action_obs,
                foot_contact_f,
                stance_time_obs,
                swing_time_obs,
            ),
            dim=-1,
        )
        self._previous_actions = self._actions.clone()
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        foot_contacts = self._get_foot_contacts()
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        foot_pos_w = self._robot.data.body_pos_w[:, self._feet_body_ids, :]
        prev_foot_contacts = self._prev_foot_contacts.clone()
        prev_stance_time = self._stance_time.clone()
        prev_swing_time = self._swing_time.clone()
        touchdown = first_contact.bool()
        liftoff = prev_foot_contacts & ~foot_contacts
        raw_touchdown_stride = torch.norm(foot_pos_w[:, :, :2] - self._last_touchdown_xy, dim=2)
        touchdown_stride = torch.where(self._has_touchdown, raw_touchdown_stride, torch.zeros_like(raw_touchdown_stride))
        stride_credit = torch.clamp(
            (touchdown_stride - self.cfg.min_touchdown_stride)
            / (self.cfg.touchdown_stride_target - self.cfg.min_touchdown_stride),
            0.0,
            1.0,
        )
        touchdown_stride_reward = torch.sum(stride_credit * touchdown.float(), dim=1)
        short_stride = torch.sum(
            torch.clamp(
                (self.cfg.touchdown_stride_target - touchdown_stride)
                / (self.cfg.touchdown_stride_target - self.cfg.min_touchdown_stride),
                0.0,
                1.0,
            )
            * touchdown.float()
            * self._has_touchdown.float(),
            dim=1,
        )
        short_stance = torch.sum(
            torch.clamp((self.cfg.min_stance_time - prev_stance_time) / self.cfg.min_stance_time, 0.0, 1.0)
            * liftoff.float(),
            dim=1,
        )
        short_swing = torch.sum(
            torch.clamp((self.cfg.min_swing_time - prev_swing_time) / self.cfg.min_swing_time, 0.0, 1.0)
            * touchdown.float()
            * self._has_touchdown.float(),
            dim=1,
        )
        self._completed_stance_time_accum += prev_stance_time * liftoff.float()
        self._completed_stance_count += liftoff.float()
        completed_swing = touchdown.float() * self._has_touchdown.float()
        self._completed_swing_time_accum += prev_swing_time * completed_swing
        self._completed_swing_count += completed_swing

        self._update_episode_progress()
        self._update_footfall_state(foot_contacts, first_contact, foot_pos_w)

        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / self.cfg.lin_vel_tracking_sigma)
        command_speed_y = torch.abs(self._commands[:, 1])
        command_active = (command_speed_y > self.cfg.moving_command_threshold).float()
        command_direction_y = torch.sign(self._commands[:, 1])
        commanded_progress_y = command_direction_y * self._robot.data.root_lin_vel_b[:, 1]
        command_progress = torch.clamp(commanded_progress_y / torch.clamp(command_speed_y, min=0.05), -1.0, 1.0)
        command_progress = command_progress * command_active
        command_progress_floor = torch.square(
            torch.clamp(
                (self.cfg.min_command_progress_ratio * command_speed_y - commanded_progress_y)
                / torch.clamp(command_speed_y, min=0.05),
                min=0.0,
            )
        )
        command_progress_floor = command_progress_floor * command_active
        command_overspeed = torch.square(
            torch.clamp(
                (commanded_progress_y - (1.0 + self.cfg.command_overspeed_tolerance) * command_speed_y)
                / torch.clamp(command_speed_y, min=0.05),
                min=0.0,
            )
        )
        command_overspeed = command_overspeed * command_active
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        yaw_w = _yaw_from_quat_wxyz(self._robot.data.root_quat_w)
        yaw_drift = torch.clamp(
            torch.square(_wrap_to_pi(yaw_w - self._initial_yaw_w) / self.cfg.yaw_drift_tolerance_rad),
            max=4.0,
        )
        lateral_velocity = torch.square(self._robot.data.root_lin_vel_b[:, 0])
        delta_xy_w = self._robot.data.root_pos_w[:, :2] - self._initial_root_xy_w
        initial_body_x_w = torch.stack((torch.cos(self._initial_yaw_w), torch.sin(self._initial_yaw_w)), dim=1)
        lateral_position = torch.clamp(
            torch.square(torch.sum(delta_xy_w * initial_body_x_w, dim=1) / self.cfg.lateral_position_tolerance_m),
            max=4.0,
        )
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        base_height_floor = torch.square(
            torch.clamp(
                (self.cfg.base_height_floor_m - base_height)
                / max(self.cfg.base_height_floor_m, 1.0e-3),
                min=0.0,
            )
        )
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        ang_vel_z_error = torch.square(self._robot.data.root_ang_vel_b[:, 2])
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        moving = torch.norm(self._commands[:, :2], dim=1) > self.cfg.moving_command_threshold
        visual_cleanup_progress = (
            torch.clamp(
                (
                    commanded_progress_y
                    - self.cfg.visual_cleanup_min_progress_ratio * command_speed_y
                )
                / torch.clamp(
                    (1.0 - self.cfg.visual_cleanup_min_progress_ratio) * command_speed_y,
                    min=0.02,
                ),
                min=0.0,
                max=1.0,
            )
            * command_active
        )
        air_time = torch.sum((last_air_time - self.cfg.feet_air_time_target) * first_contact, dim=1) * moving

        forces = self._contact_sensor.data.net_forces_w_history
        undesired_contact = torch.amax(
            torch.norm(forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1
        ) > 1.0
        undesired_contacts = torch.sum(undesired_contact.float(), dim=1)
        flat_orientation = torch.square(self._robot.data.projected_gravity_b[:, 0]) + (
            self.cfg.flat_orientation_pitch_weight * torch.square(self._robot.data.projected_gravity_b[:, 1])
        )
        pitch_forward = torch.square(self._robot.data.projected_gravity_b[:, 1])
        front_up_target = torch.square(
            self._robot.data.projected_gravity_b[:, 1] - self.cfg.front_up_target_projected_gravity_y
        )
        front_down_hinge = torch.square(
            torch.clamp(
                self._robot.data.projected_gravity_b[:, 1] - self.cfg.front_down_hinge_tolerance,
                min=0.0,
            )
        )
        front_up_progress_gate = command_active
        if self.cfg.front_up_progress_gate_ratio > 0.0:
            progress_gate = torch.clamp(
                commanded_progress_y
                / torch.clamp(self.cfg.front_up_progress_gate_ratio * command_speed_y, min=0.02),
                min=0.0,
                max=1.0,
            )
            progress_gate = torch.where(command_active > 0.5, progress_gate, torch.ones_like(progress_gate))
            front_up_progress_gate = progress_gate
            min_front_up_weight = min(max(self.cfg.front_up_progress_gate_min_weight, 0.0), 1.0)
            front_up_weight = min_front_up_weight + (1.0 - min_front_up_weight) * progress_gate
            front_up_target = front_up_target * front_up_weight
            front_down_hinge = front_down_hinge * front_up_weight
        front_up_bonus_span = max(
            self.cfg.front_up_progress_bonus_start_projected_gravity_y
            - self.cfg.front_up_target_projected_gravity_y,
            1.0e-3,
        )
        front_up_progress_bonus = (
            torch.clamp(
                (
                    self.cfg.front_up_progress_bonus_start_projected_gravity_y
                    - self._robot.data.projected_gravity_b[:, 1]
                )
                / front_up_bonus_span,
                min=0.0,
                max=1.0,
            )
            * front_up_progress_gate
            * command_active
        )
        front_up_posture = torch.zeros_like(front_up_target)
        front_up_posture_command = torch.zeros_like(front_up_target)
        if self._front_up_posture_joint_ids.numel() > 0:
            posture_joint_ids = self._front_up_posture_joint_ids
            posture_target = (
                self._robot.data.default_joint_pos[:, posture_joint_ids]
                + self._front_up_posture_offsets.unsqueeze(0)
            )
            posture_error = torch.abs(self._robot.data.joint_pos[:, posture_joint_ids] - posture_target)
            posture_command_error = torch.abs(self._processed_actions[:, posture_joint_ids] - posture_target)
            front_up_posture = (
                torch.mean(
                    torch.square(
                        torch.clamp(
                            (posture_error - self.cfg.front_up_posture_tolerance_rad)
                            / self.cfg.front_up_posture_tolerance_rad,
                            min=0.0,
                        )
                    ),
                    dim=1,
                )
                * moving.float()
            )
            front_up_posture_command = (
                torch.mean(
                    torch.square(
                        torch.clamp(
                            (posture_command_error - self.cfg.front_up_posture_tolerance_rad)
                            / self.cfg.front_up_posture_tolerance_rad,
                            min=0.0,
                        )
                    ),
                    dim=1,
                )
                * moving.float()
            )
        foot_vel_w = self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :]
        contact_f = foot_contacts.float()
        foot_xy_speed = torch.norm(foot_vel_w[:, :, :2], dim=2)
        stance_count = torch.sum(contact_f, dim=1)
        stance_foot_slip = torch.sum(foot_xy_speed * contact_f, dim=1) / (stance_count + 1.0e-6)
        foot_height = torch.clamp(foot_pos_w[:, :, 2] - self._terrain.env_origins[:, 2].unsqueeze(1), min=0.0)
        low_swing = (1.0 - contact_f) * (foot_height < self.cfg.low_swing_clearance_m).float()
        low_swing_drag = torch.sum(foot_xy_speed * low_swing, dim=1) / (torch.sum(low_swing, dim=1) + 1.0e-6)
        bl_low_swing_drag = foot_xy_speed[:, BL_FOOT_INDEX] * low_swing[:, BL_FOOT_INDEX] * moving.float()
        bl_low_swing_time = low_swing[:, BL_FOOT_INDEX] * moving.float()
        swing_mask = 1.0 - contact_f
        avg_swing_foot_speed = torch.sum(foot_xy_speed * swing_mask, dim=1) / (
            torch.sum(swing_mask, dim=1) + 1.0e-6
        )
        swing_foot_speed = torch.square(
            torch.clamp(
                (avg_swing_foot_speed - self.cfg.swing_foot_speed_target) / self.cfg.swing_foot_speed_target,
                min=0.0,
            )
        ) * moving.float()
        stride_ema_shortfall = torch.mean(
            torch.square(
                torch.clamp(
                    (self.cfg.stride_ema_floor - self._stride_length_ema) / self.cfg.stride_ema_floor,
                    min=0.0,
                    max=1.0,
                )
            ),
            dim=1,
        ) * moving.float()
        swing_height_credit = torch.clamp(
            (foot_height - self.cfg.swing_height_min) / (self.cfg.swing_height_target - self.cfg.swing_height_min),
            0.0,
            1.0,
        )
        swing_height = torch.sum(swing_height_credit * swing_mask, dim=1) / (torch.sum(swing_mask, dim=1) + 1.0e-6)
        swing_height = swing_height * moving.float()
        rear_contact_count = contact_f[:, 0] + contact_f[:, 1]
        middle_contact_count = contact_f[:, 2] + contact_f[:, 3]
        support_distribution = torch.square(torch.clamp(1.0 - rear_contact_count, min=0.0)) + torch.square(
            torch.clamp(1.0 - middle_contact_count, min=0.0)
        )
        left_contact_count = contact_f[:, 0] + contact_f[:, 2]
        right_contact_count = contact_f[:, 1] + contact_f[:, 3]
        contact_balance = 0.25 * (
            torch.square(left_contact_count - right_contact_count)
            + torch.square(rear_contact_count - middle_contact_count)
        )
        pair_a_contact = 0.5 * (contact_f[:, DIAGONAL_PAIR_A[0]] + contact_f[:, DIAGONAL_PAIR_A[1]])
        pair_b_contact = 0.5 * (contact_f[:, DIAGONAL_PAIR_B[0]] + contact_f[:, DIAGONAL_PAIR_B[1]])
        two_contact_gate = torch.exp(-torch.square(torch.sum(contact_f, dim=1) - 2.0))
        diagonal_trot = torch.abs(pair_a_contact - pair_b_contact) * two_contact_gate * moving.float()
        diagonal_pair_sync = 0.5 * (
            torch.square(contact_f[:, DIAGONAL_PAIR_A[0]] - contact_f[:, DIAGONAL_PAIR_A[1]])
            + torch.square(contact_f[:, DIAGONAL_PAIR_B[0]] - contact_f[:, DIAGONAL_PAIR_B[1]])
        ) * moving.float()
        side_antiphase = 0.5 * (
            torch.abs(contact_f[:, BL_FOOT_INDEX] - contact_f[:, 2])
            + torch.abs(contact_f[:, BR_FOOT_INDEX] - contact_f[:, 3])
        ) * two_contact_gate * moving.float()
        rear_contact_antiphase = (
            torch.abs(contact_f[:, BL_FOOT_INDEX] - contact_f[:, BR_FOOT_INDEX])
            * two_contact_gate
            * moving.float()
        )
        episode_time = torch.clamp(self._total_time, min=self.step_dt)
        touchdown_rate_by_foot = self._step_counts / episode_time.unsqueeze(1)
        touchdown_rate_mean = torch.mean(touchdown_rate_by_foot, dim=1, keepdim=True)
        touchdown_cadence = torch.square(
            torch.clamp(
                (torch.squeeze(touchdown_rate_mean, dim=1) - self.cfg.touchdown_cadence_target_hz)
                / self.cfg.touchdown_cadence_target_hz,
                min=0.0,
            )
        )
        touchdown_rate_balance = torch.mean(torch.abs(touchdown_rate_by_foot - touchdown_rate_mean), dim=1)
        rapid_touchdown = torch.square(
            torch.clamp(
                (torch.squeeze(touchdown_rate_mean, dim=1) - self.cfg.max_touchdown_rate_hz)
                / self.cfg.max_touchdown_rate_hz,
                min=0.0,
            )
        )
        anchor_slip_xy = torch.norm(foot_pos_w[:, :, :2] - self._stance_anchor_xy, dim=2)
        anchored_stance = contact_f * self._has_touchdown.float() * (self._stance_time > self.cfg.stance_anchor_grace_s).float()
        stance_anchor_slip = torch.sum(
            torch.clamp(
                (anchor_slip_xy - self.cfg.stance_anchor_slip_tolerance_m) / self.cfg.stance_anchor_slip_tolerance_m,
                min=0.0,
            )
            * anchored_stance,
            dim=1,
        ) / (torch.sum(anchored_stance, dim=1) + 1.0e-6)
        current_contact_time_accum = self._per_foot_contact_time_accum + contact_f * self.step_dt
        current_stance_slip_accum = self._per_foot_stance_slip_accum + foot_xy_speed * contact_f * self.step_dt
        per_foot_contact_fraction = current_contact_time_accum / episode_time.unsqueeze(1)
        per_foot_stance_slip = current_stance_slip_accum / torch.clamp(current_contact_time_accum, min=self.step_dt)
        per_foot_stance_duration = self._completed_stance_time_accum / torch.clamp(
            self._completed_stance_count, min=1.0
        )
        rear_symmetry_ready = (
            (self._total_time > self.cfg.rear_symmetry_ready_time_s)
            & (self._step_counts[:, BL_FOOT_INDEX] >= 2.0)
            & (self._step_counts[:, BR_FOOT_INDEX] >= 2.0)
        ).float()
        rear_stride_balance = torch.square(
            (self._stride_length_ema[:, BL_FOOT_INDEX] - self._stride_length_ema[:, BR_FOOT_INDEX])
            / self.cfg.touchdown_stride_target
        ) * rear_symmetry_ready
        rear_stance_duration_balance = torch.square(
            (per_foot_stance_duration[:, BL_FOOT_INDEX] - per_foot_stance_duration[:, BR_FOOT_INDEX])
            / self.cfg.min_stance_time
        ) * rear_symmetry_ready
        rear_touchdown_rate_balance = torch.square(
            (touchdown_rate_by_foot[:, BL_FOOT_INDEX] - touchdown_rate_by_foot[:, BR_FOOT_INDEX])
            / self.cfg.touchdown_cadence_target_hz
        ) * rear_symmetry_ready
        rear_contact_fraction_balance = torch.square(
            (per_foot_contact_fraction[:, BL_FOOT_INDEX] - per_foot_contact_fraction[:, BR_FOOT_INDEX])
            / self.cfg.rear_contact_fraction_balance_tolerance
        ) * rear_symmetry_ready
        rear_contact_fraction_target = (
            torch.square(
                (per_foot_contact_fraction[:, BL_FOOT_INDEX] - self.cfg.rear_contact_fraction_target)
                / self.cfg.rear_contact_fraction_target_tolerance
            )
            + torch.square(
                (per_foot_contact_fraction[:, BR_FOOT_INDEX] - self.cfg.rear_contact_fraction_target)
                / self.cfg.rear_contact_fraction_target_tolerance
            )
        ) * rear_symmetry_ready
        bl_contact_fraction_excess = torch.square(
            torch.clamp(
                (per_foot_contact_fraction[:, BL_FOOT_INDEX] - self.cfg.bl_contact_fraction_ceiling)
                / self.cfg.bl_contact_fraction_excess_tolerance,
                min=0.0,
            )
        ) * rear_symmetry_ready
        rear_stance_slip_balance = torch.square(
            (per_foot_stance_slip[:, BL_FOOT_INDEX] - per_foot_stance_slip[:, BR_FOOT_INDEX])
            / self.cfg.rear_stance_slip_balance_tolerance_mps
        ) * rear_symmetry_ready
        rear_left_stance_slip_excess = torch.square(
            torch.clamp(
                (per_foot_stance_slip[:, BL_FOOT_INDEX] - self.cfg.rear_left_stance_slip_tolerance_mps)
                / self.cfg.rear_stance_slip_balance_tolerance_mps,
                min=0.0,
            )
        ) * rear_symmetry_ready
        root_quat = self._robot.data.root_quat_w[:, None, :].expand(-1, foot_pos_w.shape[1], -1)
        foot_pos_b = quat_apply_inverse(
            root_quat.reshape(-1, 4),
            (foot_pos_w - self._robot.data.root_pos_w[:, None, :]).reshape(-1, 3),
        ).reshape_as(foot_pos_w)
        rear_lateral_center = foot_pos_b[:, BL_FOOT_INDEX, 0] + foot_pos_b[:, BR_FOOT_INDEX, 0]
        rear_lateral_mirror = torch.square(
            torch.clamp(
                torch.abs(rear_lateral_center - self.cfg.rear_lateral_mirror_target_m)
                - self.cfg.rear_lateral_mirror_tolerance_m,
                min=0.0,
            )
            / self.cfg.rear_lateral_mirror_tolerance_m
        ) * moving.float()
        rear_femur_body_pos_w = self._robot.data.body_pos_w[:, self._rear_femur_body_ids, :]
        rear_femur_root_quat = self._robot.data.root_quat_w[:, None, :].expand(
            -1, rear_femur_body_pos_w.shape[1], -1
        )
        rear_femur_body_pos_b = quat_apply_inverse(
            rear_femur_root_quat.reshape(-1, 4),
            (rear_femur_body_pos_w - self._robot.data.root_pos_w[:, None, :]).reshape(-1, 3),
        ).reshape_as(rear_femur_body_pos_w)
        bl_femur_body_pos_b = rear_femur_body_pos_b[:, 0]
        br_femur_body_pos_b = rear_femur_body_pos_b[:, 1]
        rear_tibia_body_pos_w = self._robot.data.body_pos_w[:, self._rear_tibia_body_ids, :]
        rear_tibia_root_quat = self._robot.data.root_quat_w[:, None, :].expand(
            -1, rear_tibia_body_pos_w.shape[1], -1
        )
        rear_tibia_body_pos_b = quat_apply_inverse(
            rear_tibia_root_quat.reshape(-1, 4),
            (rear_tibia_body_pos_w - self._robot.data.root_pos_w[:, None, :]).reshape(-1, 3),
        ).reshape_as(rear_tibia_body_pos_w)
        bl_tibia_body_pos_b = rear_tibia_body_pos_b[:, 0]
        br_tibia_body_pos_b = rear_tibia_body_pos_b[:, 1]
        bl_femur_to_tibia_b = bl_tibia_body_pos_b - bl_femur_body_pos_b
        br_femur_to_tibia_b = br_tibia_body_pos_b - br_femur_body_pos_b
        rear_femur_tibia_segment_residual = torch.stack(
            (
                bl_femur_to_tibia_b[:, 0] + br_femur_to_tibia_b[:, 0],
                bl_femur_to_tibia_b[:, 1] - br_femur_to_tibia_b[:, 1],
                bl_femur_to_tibia_b[:, 2] - br_femur_to_tibia_b[:, 2],
            ),
            dim=1,
        )
        rear_tibia_link_residual = torch.stack(
            (
                bl_tibia_body_pos_b[:, 0] + br_tibia_body_pos_b[:, 0],
                bl_tibia_body_pos_b[:, 1] - br_tibia_body_pos_b[:, 1],
                bl_tibia_body_pos_b[:, 2] - br_tibia_body_pos_b[:, 2],
            ),
            dim=1,
        )
        bl_tibia_to_foot_b = foot_pos_b[:, BL_FOOT_INDEX] - bl_tibia_body_pos_b
        br_tibia_to_foot_b = foot_pos_b[:, BR_FOOT_INDEX] - br_tibia_body_pos_b
        rear_tibia_segment_residual = torch.stack(
            (
                bl_tibia_to_foot_b[:, 0] + br_tibia_to_foot_b[:, 0],
                bl_tibia_to_foot_b[:, 1] - br_tibia_to_foot_b[:, 1],
                bl_tibia_to_foot_b[:, 2] - br_tibia_to_foot_b[:, 2],
            ),
            dim=1,
        )
        rear_femur_tibia_segment_mirror = torch.mean(
            torch.square(
                torch.clamp(
                    torch.abs(rear_femur_tibia_segment_residual)
                    - self.cfg.rear_femur_tibia_segment_mirror_tolerance_m,
                    min=0.0,
                )
                / self.cfg.rear_femur_tibia_segment_mirror_tolerance_m
            ),
            dim=1,
        ) * rear_symmetry_ready * moving.float()
        rear_femur_tibia_z_mirror = torch.square(
            torch.clamp(
                torch.abs(bl_femur_to_tibia_b[:, 2] - br_femur_to_tibia_b[:, 2])
                - self.cfg.rear_femur_tibia_z_mirror_tolerance_m,
                min=0.0,
            )
            / self.cfg.rear_femur_tibia_z_mirror_tolerance_m
        ) * rear_symmetry_ready * moving.float()
        rear_tibia_link_mirror = torch.mean(
            torch.square(
                torch.clamp(
                    torch.abs(rear_tibia_link_residual) - self.cfg.rear_tibia_link_mirror_tolerance_m,
                    min=0.0,
                )
                / self.cfg.rear_tibia_link_mirror_tolerance_m
            ),
            dim=1,
        ) * rear_symmetry_ready * moving.float()
        bl_tibia_link_outward_m = -bl_tibia_body_pos_b[:, 0]
        br_tibia_link_outward_m = br_tibia_body_pos_b[:, 0]
        bl_tibia_link_outward_vs_br = (
            torch.square(
                torch.clamp(
                    (
                        br_tibia_link_outward_m
                        - bl_tibia_link_outward_m
                        - self.cfg.bl_tibia_link_outward_vs_br_tolerance_m
                    )
                    / self.cfg.bl_tibia_link_outward_vs_br_tolerance_m,
                    min=0.0,
                )
            )
            * rear_symmetry_ready
            * moving.float()
        )
        bl_tibia_link_outward_floor = (
            torch.square(
                torch.clamp(
                    (
                        self.cfg.bl_tibia_link_outward_floor_target_m
                        - bl_tibia_link_outward_m
                    )
                    / self.cfg.bl_tibia_link_outward_floor_tolerance_m,
                    min=0.0,
                )
            )
            * rear_symmetry_ready
            * moving.float()
        )
        rear_tibia_segment_mirror = torch.mean(
            torch.square(
                torch.clamp(
                    torch.abs(rear_tibia_segment_residual) - self.cfg.rear_tibia_segment_mirror_tolerance_m,
                    min=0.0,
                )
                / self.cfg.rear_tibia_segment_mirror_tolerance_m
            ),
            dim=1,
        ) * rear_symmetry_ready * moving.float()
        bl_tibia_segment_outward_m = -bl_tibia_to_foot_b[:, 0]
        br_tibia_segment_outward_m = br_tibia_to_foot_b[:, 0]
        rear_tibia_segment_outward = (
            torch.square(
                torch.clamp(
                    (self.cfg.rear_tibia_segment_outward_min_m - bl_tibia_segment_outward_m)
                    / self.cfg.rear_tibia_segment_outward_tolerance_m,
                    min=0.0,
                )
            )
            + torch.square(
                torch.clamp(
                    (self.cfg.rear_tibia_segment_outward_min_m - br_tibia_segment_outward_m)
                    / self.cfg.rear_tibia_segment_outward_tolerance_m,
                    min=0.0,
                )
            )
        ) * rear_symmetry_ready * moving.float()
        rear_tibia_segment_outward_target = (
            torch.square(
                torch.clamp(
                    torch.abs(bl_tibia_segment_outward_m - self.cfg.rear_tibia_segment_outward_target_m)
                    - self.cfg.rear_tibia_segment_outward_target_tolerance_m,
                    min=0.0,
                )
                / self.cfg.rear_tibia_segment_outward_target_tolerance_m
            )
            + torch.square(
                torch.clamp(
                    torch.abs(br_tibia_segment_outward_m - self.cfg.rear_tibia_segment_outward_target_m)
                    - self.cfg.rear_tibia_segment_outward_target_tolerance_m,
                    min=0.0,
                )
                / self.cfg.rear_tibia_segment_outward_target_tolerance_m
            )
        ) * rear_symmetry_ready * moving.float()
        rear_swing_mask = 1.0 - contact_f[:, [BL_FOOT_INDEX, BR_FOOT_INDEX]]
        rear_swing_tibia_segment_outward_raw = torch.stack(
            (
                torch.square(
                    torch.clamp(
                        (
                            self.cfg.rear_swing_tibia_segment_outward_target_m
                            - bl_tibia_segment_outward_m
                        )
                        / self.cfg.rear_swing_tibia_segment_outward_tolerance_m,
                        min=0.0,
                    )
                ),
                torch.square(
                    torch.clamp(
                        (
                            self.cfg.rear_swing_tibia_segment_outward_target_m
                            - br_tibia_segment_outward_m
                        )
                        / self.cfg.rear_swing_tibia_segment_outward_tolerance_m,
                        min=0.0,
                    )
                ),
            ),
            dim=1,
        )
        rear_swing_tibia_segment_outward = (
            torch.sum(rear_swing_tibia_segment_outward_raw * rear_swing_mask, dim=1)
            / torch.clamp(torch.sum(rear_swing_mask, dim=1), min=1.0)
        ) * rear_symmetry_ready * moving.float()
        bl_swing_tibia_segment_outward = (
            rear_swing_tibia_segment_outward_raw[:, 0]
            * rear_swing_mask[:, 0]
            * rear_symmetry_ready
            * moving.float()
        )
        rear_swing_outward_current = torch.stack(
            (bl_tibia_segment_outward_m, br_tibia_segment_outward_m),
            dim=1,
        )
        current_rear_swing_outward_accum = (
            self._rear_swing_outward_accum + rear_swing_outward_current * rear_swing_mask * self.step_dt
        )
        current_rear_swing_outward_time = self._rear_swing_outward_time + rear_swing_mask * self.step_dt
        rear_swing_outward_mean = current_rear_swing_outward_accum / torch.clamp(
            current_rear_swing_outward_time,
            min=self.step_dt,
        )
        rear_swing_outward_mean_balance = torch.square(
            torch.clamp(
                (
                    rear_swing_outward_mean[:, 1]
                    - rear_swing_outward_mean[:, 0]
                    - self.cfg.rear_swing_outward_mean_balance_tolerance_m
                )
                / self.cfg.rear_swing_outward_mean_balance_tolerance_m,
                min=0.0,
            )
        ) * rear_symmetry_ready * moving.float()
        bl_swing_outward_mean_floor = torch.square(
            torch.clamp(
                (
                    self.cfg.bl_swing_outward_mean_floor_target_m
                    - rear_swing_outward_mean[:, 0]
                )
                / self.cfg.bl_swing_outward_mean_floor_tolerance_m,
                min=0.0,
            )
        ) * rear_symmetry_ready * moving.float()
        bl_swing_outward_vs_br = (
            torch.square(
                torch.clamp(
                    (
                        br_tibia_segment_outward_m
                        - bl_tibia_segment_outward_m
                        - self.cfg.bl_swing_outward_vs_br_tolerance_m
                    )
                    / self.cfg.bl_swing_outward_vs_br_tolerance_m,
                    min=0.0,
                )
            )
            * rear_swing_mask[:, 0]
            * rear_symmetry_ready
            * moving.float()
        )
        delayed_br_tibia_segment_outward_m = self._br_tibia_segment_outward_history[:, 0]
        delayed_br_tibia_link_outward_m = self._br_tibia_link_outward_history[:, 0]
        delayed_br_tibia_body_pos_b = self._br_tibia_link_pos_history[:, 0]
        delayed_br_femur_to_tibia_b = self._br_femur_tibia_vec_history[:, 0]
        delayed_br_tibia_axis_x_b = self._br_tibia_axis_x_history[:, 0]
        delayed_br_tibia_axis_y_b = self._br_tibia_axis_y_history[:, 0]
        delayed_br_rear_coxa_joint_pos = self._br_rear_coxa_joint_pos_history[:, 0]
        delayed_br_rear_coxa_target_pos = self._br_rear_coxa_target_pos_history[:, 0]
        delayed_br_rear_distal_joint_pos = self._br_rear_distal_joint_pos_history[:, 0]
        delayed_br_rear_distal_target_pos = self._br_rear_distal_target_pos_history[:, 0]
        delayed_br_foot_height = self._br_foot_height_history[:, 0]
        delayed_br_swing = self._br_swing_history[:, 0]
        rear_phase_delay_ready = (
            self._rear_phase_history_steps >= float(self.cfg.rear_phase_symmetry_delay_steps)
        ).float()
        delayed_br_foot_height_target = torch.maximum(
            delayed_br_foot_height,
            torch.full_like(delayed_br_foot_height, self.cfg.bl_phase_delayed_foot_height_min_m),
        )
        bl_phase_delayed_foot_height = (
            torch.square(
                torch.clamp(
                    (
                        delayed_br_foot_height_target
                        - foot_height[:, BL_FOOT_INDEX]
                        - self.cfg.bl_phase_delayed_foot_height_tolerance_m
                    )
                    / self.cfg.bl_phase_delayed_foot_height_tolerance_m,
                    min=0.0,
                )
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        phase_delayed_tibia_link_residual = torch.stack(
            (
                bl_tibia_body_pos_b[:, 0] + delayed_br_tibia_body_pos_b[:, 0],
                bl_tibia_body_pos_b[:, 1] - delayed_br_tibia_body_pos_b[:, 1],
                bl_tibia_body_pos_b[:, 2] - delayed_br_tibia_body_pos_b[:, 2],
            ),
            dim=1,
        )
        phase_delayed_femur_tibia_residual = torch.stack(
            (
                bl_femur_to_tibia_b[:, 0] + delayed_br_femur_to_tibia_b[:, 0],
                bl_femur_to_tibia_b[:, 1] - delayed_br_femur_to_tibia_b[:, 1],
                bl_femur_to_tibia_b[:, 2] - delayed_br_femur_to_tibia_b[:, 2],
            ),
            dim=1,
        )
        bl_phase_delayed_tibia_outward = (
            torch.square(
                torch.clamp(
                    (
                        torch.abs(bl_tibia_segment_outward_m - delayed_br_tibia_segment_outward_m)
                        - self.cfg.bl_phase_delayed_tibia_outward_tolerance_m
                    )
                    / self.cfg.bl_phase_delayed_tibia_outward_tolerance_m,
                    min=0.0,
                )
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_phase_delayed_tibia_link_outward = (
            torch.square(
                torch.clamp(
                    (
                        delayed_br_tibia_link_outward_m
                        - bl_tibia_link_outward_m
                        - self.cfg.bl_phase_delayed_tibia_link_outward_tolerance_m
                    )
                    / self.cfg.bl_phase_delayed_tibia_link_outward_tolerance_m,
                    min=0.0,
                )
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_phase_delayed_tibia_link_mirror = (
            torch.mean(
                torch.square(
                    torch.clamp(
                        torch.abs(phase_delayed_tibia_link_residual)
                        - self.cfg.bl_phase_delayed_tibia_link_mirror_tolerance_m,
                        min=0.0,
                    )
                    / self.cfg.bl_phase_delayed_tibia_link_mirror_tolerance_m
                ),
                dim=1,
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_phase_delayed_femur_tibia_mirror = (
            torch.mean(
                torch.square(
                    torch.clamp(
                        torch.abs(phase_delayed_femur_tibia_residual)
                        - self.cfg.bl_phase_delayed_femur_tibia_mirror_tolerance_m,
                        min=0.0,
                    )
                    / self.cfg.bl_phase_delayed_femur_tibia_mirror_tolerance_m
                ),
                dim=1,
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_swing_gate = rear_swing_mask[:, 0] * torch.clamp(
            (foot_height[:, BL_FOOT_INDEX] - self.cfg.bl_lifted_swing_min_height_m)
            / self.cfg.bl_lifted_swing_height_tolerance_m,
            0.0,
            1.0,
        )
        rear_tibia_body_quat_w = self._robot.data.body_quat_w[:, self._rear_tibia_body_ids, :]
        rear_tibia_axis_root_quat = self._robot.data.root_quat_w[:, None, :].expand(
            -1, rear_tibia_body_quat_w.shape[1], -1
        )
        tibia_axis_x_local = torch.zeros_like(rear_tibia_body_pos_b)
        tibia_axis_y_local = torch.zeros_like(rear_tibia_body_pos_b)
        tibia_axis_x_local[:, :, 0] = 1.0
        tibia_axis_y_local[:, :, 1] = 1.0
        tibia_axis_x_w = quat_apply(
            rear_tibia_body_quat_w.reshape(-1, 4),
            tibia_axis_x_local.reshape(-1, 3),
        ).reshape_as(rear_tibia_body_pos_b)
        tibia_axis_y_w = quat_apply(
            rear_tibia_body_quat_w.reshape(-1, 4),
            tibia_axis_y_local.reshape(-1, 3),
        ).reshape_as(rear_tibia_body_pos_b)
        tibia_axis_x_b = quat_apply_inverse(
            rear_tibia_axis_root_quat.reshape(-1, 4),
            tibia_axis_x_w.reshape(-1, 3),
        ).reshape_as(rear_tibia_body_pos_b)
        tibia_axis_y_b = quat_apply_inverse(
            rear_tibia_axis_root_quat.reshape(-1, 4),
            tibia_axis_y_w.reshape(-1, 3),
        ).reshape_as(rear_tibia_body_pos_b)
        tibia_mesh_long_local = torch.zeros_like(rear_tibia_body_pos_b)
        tibia_mesh_secondary_local = torch.zeros_like(rear_tibia_body_pos_b)
        tibia_mesh_long_local[:, :, 0] = 0.994444
        tibia_mesh_long_local[:, :, 2] = -0.105267
        tibia_mesh_secondary_local[:, :, 0] = 0.105267
        tibia_mesh_secondary_local[:, :, 2] = 0.994444
        tibia_mesh_long_w = quat_apply(
            rear_tibia_body_quat_w.reshape(-1, 4),
            tibia_mesh_long_local.reshape(-1, 3),
        ).reshape_as(rear_tibia_body_pos_b)
        tibia_mesh_secondary_w = quat_apply(
            rear_tibia_body_quat_w.reshape(-1, 4),
            tibia_mesh_secondary_local.reshape(-1, 3),
        ).reshape_as(rear_tibia_body_pos_b)
        tibia_mesh_long_b = quat_apply_inverse(
            rear_tibia_axis_root_quat.reshape(-1, 4),
            tibia_mesh_long_w.reshape(-1, 3),
        ).reshape_as(rear_tibia_body_pos_b)
        tibia_mesh_secondary_b = quat_apply_inverse(
            rear_tibia_axis_root_quat.reshape(-1, 4),
            tibia_mesh_secondary_w.reshape(-1, 3),
        ).reshape_as(rear_tibia_body_pos_b)
        br_tibia_axis_x_mirror_b = torch.stack(
            (
                tibia_axis_x_b[:, 1, 0],
                -tibia_axis_x_b[:, 1, 1],
                -tibia_axis_x_b[:, 1, 2],
            ),
            dim=1,
        )
        br_tibia_axis_y_mirror_b = torch.stack(
            (
                tibia_axis_y_b[:, 1, 0],
                -tibia_axis_y_b[:, 1, 1],
                -tibia_axis_y_b[:, 1, 2],
            ),
            dim=1,
        )
        br_tibia_mesh_long_mirror_b = torch.stack(
            (
                tibia_mesh_long_b[:, 1, 0],
                -tibia_mesh_long_b[:, 1, 1],
                -tibia_mesh_long_b[:, 1, 2],
            ),
            dim=1,
        )
        br_tibia_mesh_secondary_mirror_b = torch.stack(
            (
                tibia_mesh_secondary_b[:, 1, 0],
                -tibia_mesh_secondary_b[:, 1, 1],
                -tibia_mesh_secondary_b[:, 1, 2],
            ),
            dim=1,
        )
        rear_tibia_axis_mirror_error = 0.5 * (
            torch.norm(tibia_axis_x_b[:, 0] - br_tibia_axis_x_mirror_b, dim=1)
            + torch.norm(tibia_axis_y_b[:, 0] - br_tibia_axis_y_mirror_b, dim=1)
        )
        rear_tibia_axis_mirror = torch.square(
            torch.clamp(
                (rear_tibia_axis_mirror_error - self.cfg.rear_tibia_axis_mirror_tolerance)
                / self.cfg.rear_tibia_axis_mirror_tolerance,
                min=0.0,
            )
        ) * rear_symmetry_ready * moving.float()
        rear_tibia_primary_axis_mirror_error = torch.norm(
            tibia_axis_x_b[:, 0] - br_tibia_axis_x_mirror_b,
            dim=1,
        )
        rear_tibia_primary_axis_mirror = torch.square(
            torch.clamp(
                (
                    rear_tibia_primary_axis_mirror_error
                    - self.cfg.rear_tibia_primary_axis_mirror_tolerance
                )
                / self.cfg.rear_tibia_primary_axis_mirror_tolerance,
                min=0.0,
            )
        ) * rear_symmetry_ready * moving.float()
        rear_tibia_secondary_axis_mirror_error = torch.norm(
            tibia_axis_y_b[:, 0] - br_tibia_axis_y_mirror_b,
            dim=1,
        )
        rear_tibia_secondary_axis_mirror = torch.square(
            torch.clamp(
                (
                    rear_tibia_secondary_axis_mirror_error
                    - self.cfg.rear_tibia_secondary_axis_mirror_tolerance
                )
                / self.cfg.rear_tibia_secondary_axis_mirror_tolerance,
                min=0.0,
            )
        ) * rear_symmetry_ready * moving.float()
        rear_tibia_mesh_long_axis_mirror_error = torch.norm(
            tibia_mesh_long_b[:, 0] - br_tibia_mesh_long_mirror_b,
            dim=1,
        )
        rear_tibia_mesh_long_axis_mirror = torch.square(
            torch.clamp(
                (
                    rear_tibia_mesh_long_axis_mirror_error
                    - self.cfg.rear_tibia_mesh_long_axis_mirror_tolerance
                )
                / self.cfg.rear_tibia_mesh_long_axis_mirror_tolerance,
                min=0.0,
            )
        ) * rear_symmetry_ready * moving.float()
        rear_tibia_mesh_secondary_axis_mirror_error = torch.norm(
            tibia_mesh_secondary_b[:, 0] - br_tibia_mesh_secondary_mirror_b,
            dim=1,
        )
        rear_tibia_mesh_secondary_axis_mirror = torch.square(
            torch.clamp(
                (
                    rear_tibia_mesh_secondary_axis_mirror_error
                    - self.cfg.rear_tibia_mesh_secondary_axis_mirror_tolerance
                )
                / self.cfg.rear_tibia_mesh_secondary_axis_mirror_tolerance,
                min=0.0,
            )
        ) * rear_symmetry_ready * moving.float()
        bl_lifted_tibia_axis_mirror = (
            torch.square(
                torch.clamp(
                    (
                        rear_tibia_axis_mirror_error
                        - self.cfg.bl_lifted_tibia_axis_mirror_tolerance
                    )
                    / self.cfg.bl_lifted_tibia_axis_mirror_tolerance,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_tibia_primary_axis_mirror = (
            torch.square(
                torch.clamp(
                    (
                        rear_tibia_primary_axis_mirror_error
                        - self.cfg.bl_lifted_tibia_primary_axis_mirror_tolerance
                    )
                    / self.cfg.bl_lifted_tibia_primary_axis_mirror_tolerance,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_tibia_secondary_axis_mirror = (
            torch.square(
                torch.clamp(
                    (
                        rear_tibia_secondary_axis_mirror_error
                        - self.cfg.bl_lifted_tibia_secondary_axis_mirror_tolerance
                    )
                    / self.cfg.bl_lifted_tibia_secondary_axis_mirror_tolerance,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_tibia_mesh_long_axis_mirror = (
            torch.square(
                torch.clamp(
                    (
                        rear_tibia_mesh_long_axis_mirror_error
                        - self.cfg.bl_lifted_tibia_mesh_long_axis_mirror_tolerance
                    )
                    / self.cfg.bl_lifted_tibia_mesh_long_axis_mirror_tolerance,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_tibia_mesh_secondary_axis_mirror = (
            torch.square(
                torch.clamp(
                    (
                        rear_tibia_mesh_secondary_axis_mirror_error
                        - self.cfg.bl_lifted_tibia_mesh_secondary_axis_mirror_tolerance
                    )
                    / self.cfg.bl_lifted_tibia_mesh_secondary_axis_mirror_tolerance,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * rear_symmetry_ready
            * moving.float()
        )
        rear_tibia_mesh_long_lateral = torch.abs(tibia_mesh_long_b[:, :, 0])
        rear_tibia_mesh_secondary_lateral = torch.abs(tibia_mesh_secondary_b[:, :, 0])
        rear_tibia_mesh_long_lateral_excess_raw = torch.square(
            torch.clamp(
                (
                    rear_tibia_mesh_long_lateral
                    - self.cfg.rear_tibia_mesh_long_lateral_cap
                )
                / self.cfg.rear_tibia_mesh_long_lateral_tolerance,
                min=0.0,
            )
        )
        rear_tibia_mesh_secondary_lateral_excess_raw = torch.square(
            torch.clamp(
                (
                    rear_tibia_mesh_secondary_lateral
                    - self.cfg.rear_tibia_mesh_secondary_lateral_cap
                )
                / self.cfg.rear_tibia_mesh_secondary_lateral_tolerance,
                min=0.0,
            )
        )
        rear_tibia_mesh_long_lateral_excess = (
            torch.mean(rear_tibia_mesh_long_lateral_excess_raw, dim=1)
            * rear_symmetry_ready
            * moving.float()
        )
        rear_tibia_mesh_secondary_lateral_excess = (
            torch.mean(rear_tibia_mesh_secondary_lateral_excess_raw, dim=1)
            * rear_symmetry_ready
            * moving.float()
        )
        bl_swing_tibia_mesh_long_lateral_excess = (
            torch.square(
                torch.clamp(
                    (
                        rear_tibia_mesh_long_lateral[:, 0]
                        - self.cfg.bl_swing_tibia_mesh_long_lateral_cap
                    )
                    / self.cfg.bl_swing_tibia_mesh_long_lateral_tolerance,
                    min=0.0,
                )
            )
            * rear_swing_mask[:, 0]
            * rear_symmetry_ready
            * moving.float()
        )
        bl_swing_tibia_mesh_secondary_lateral_excess = (
            torch.square(
                torch.clamp(
                    (
                        rear_tibia_mesh_secondary_lateral[:, 0]
                        - self.cfg.bl_swing_tibia_mesh_secondary_lateral_cap
                    )
                    / self.cfg.bl_swing_tibia_mesh_secondary_lateral_tolerance,
                    min=0.0,
                )
            )
            * rear_swing_mask[:, 0]
            * rear_symmetry_ready
            * moving.float()
        )
        rear_tibia_mesh_secondary_lateral_target = (
            torch.mean(
                torch.square(
                    (
                        rear_tibia_mesh_secondary_lateral
                        - self.cfg.rear_tibia_mesh_secondary_lateral_target
                    )
                    / self.cfg.rear_tibia_mesh_secondary_lateral_target_tolerance
                ),
                dim=1,
            )
            * rear_symmetry_ready
            * visual_cleanup_progress
        )
        bl_swing_tibia_mesh_secondary_lateral_target = (
            torch.square(
                (
                    rear_tibia_mesh_secondary_lateral[:, 0]
                    - self.cfg.bl_swing_tibia_mesh_secondary_lateral_target
                )
                / self.cfg.bl_swing_tibia_mesh_secondary_lateral_target_tolerance
            )
            * rear_swing_mask[:, 0]
            * rear_symmetry_ready
            * visual_cleanup_progress
        )
        br_swing_tibia_mesh_secondary_lateral_target = (
            torch.square(
                (
                    rear_tibia_mesh_secondary_lateral[:, 1]
                    - self.cfg.br_swing_tibia_mesh_secondary_lateral_target
                )
                / self.cfg.br_swing_tibia_mesh_secondary_lateral_target_tolerance
            )
            * rear_swing_mask[:, 1]
            * rear_symmetry_ready
            * visual_cleanup_progress
        )
        delayed_br_tibia_axis_x_mirror_b = torch.stack(
            (
                delayed_br_tibia_axis_x_b[:, 0],
                -delayed_br_tibia_axis_x_b[:, 1],
                -delayed_br_tibia_axis_x_b[:, 2],
            ),
            dim=1,
        )
        delayed_br_tibia_axis_y_mirror_b = torch.stack(
            (
                delayed_br_tibia_axis_y_b[:, 0],
                -delayed_br_tibia_axis_y_b[:, 1],
                -delayed_br_tibia_axis_y_b[:, 2],
            ),
            dim=1,
        )
        phase_delayed_tibia_primary_axis_mirror_error = torch.norm(
            tibia_axis_x_b[:, 0] - delayed_br_tibia_axis_x_mirror_b,
            dim=1,
        )
        phase_delayed_tibia_secondary_axis_mirror_error = torch.norm(
            tibia_axis_y_b[:, 0] - delayed_br_tibia_axis_y_mirror_b,
            dim=1,
        )
        bl_phase_delayed_tibia_primary_axis_mirror = (
            torch.square(
                torch.clamp(
                    (
                        phase_delayed_tibia_primary_axis_mirror_error
                        - self.cfg.bl_phase_delayed_tibia_primary_axis_mirror_tolerance
                    )
                    / self.cfg.bl_phase_delayed_tibia_primary_axis_mirror_tolerance,
                    min=0.0,
                )
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_phase_delayed_tibia_primary_axis_mirror = (
            torch.square(
                torch.clamp(
                    (
                        phase_delayed_tibia_primary_axis_mirror_error
                        - self.cfg.bl_lifted_phase_delayed_tibia_primary_axis_mirror_tolerance
                    )
                    / self.cfg.bl_lifted_phase_delayed_tibia_primary_axis_mirror_tolerance,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_phase_delayed_tibia_secondary_axis_mirror = (
            torch.square(
                torch.clamp(
                    (
                        phase_delayed_tibia_secondary_axis_mirror_error
                        - self.cfg.bl_phase_delayed_tibia_secondary_axis_mirror_tolerance
                    )
                    / self.cfg.bl_phase_delayed_tibia_secondary_axis_mirror_tolerance,
                    min=0.0,
                )
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_phase_delayed_tibia_secondary_axis_mirror = (
            torch.square(
                torch.clamp(
                    (
                        phase_delayed_tibia_secondary_axis_mirror_error
                        - self.cfg.bl_lifted_phase_delayed_tibia_secondary_axis_mirror_tolerance
                    )
                    / self.cfg.bl_lifted_phase_delayed_tibia_secondary_axis_mirror_tolerance,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_phase_delayed_tibia_link_outward = (
            torch.square(
                torch.clamp(
                    (
                        delayed_br_tibia_link_outward_m
                        - bl_tibia_link_outward_m
                        - self.cfg.bl_lifted_phase_delayed_tibia_link_outward_tolerance_m
                    )
                    / self.cfg.bl_lifted_phase_delayed_tibia_link_outward_tolerance_m,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_phase_delayed_tibia_link_mirror = (
            torch.mean(
                torch.square(
                    torch.clamp(
                        torch.abs(phase_delayed_tibia_link_residual)
                        - self.cfg.bl_lifted_phase_delayed_tibia_link_mirror_tolerance_m,
                        min=0.0,
                    )
                    / self.cfg.bl_lifted_phase_delayed_tibia_link_mirror_tolerance_m
                ),
                dim=1,
            )
            * bl_lifted_swing_gate
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_phase_delayed_femur_tibia_mirror = (
            torch.mean(
                torch.square(
                    torch.clamp(
                        torch.abs(phase_delayed_femur_tibia_residual)
                        - self.cfg.bl_lifted_phase_delayed_femur_tibia_mirror_tolerance_m,
                        min=0.0,
                    )
                    / self.cfg.bl_lifted_phase_delayed_femur_tibia_mirror_tolerance_m
                ),
                dim=1,
            )
            * bl_lifted_swing_gate
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_swing_tibia_link_outward_vs_br = (
            torch.square(
                torch.clamp(
                    (
                        br_tibia_link_outward_m
                        - bl_tibia_link_outward_m
                        - self.cfg.bl_lifted_swing_tibia_link_outward_vs_br_tolerance_m
                    )
                    / self.cfg.bl_lifted_swing_tibia_link_outward_vs_br_tolerance_m,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_swing_tibia_link_outward_floor = (
            torch.square(
                torch.clamp(
                    (
                        self.cfg.bl_lifted_swing_tibia_link_outward_floor_target_m
                        - bl_tibia_link_outward_m
                    )
                    / self.cfg.bl_lifted_swing_tibia_link_outward_floor_tolerance_m,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_swing_tibia_segment_outward = (
            torch.square(
                torch.clamp(
                    (
                        self.cfg.bl_lifted_swing_tibia_segment_outward_target_m
                        - bl_tibia_segment_outward_m
                    )
                    / self.cfg.bl_lifted_swing_tibia_segment_outward_tolerance_m,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_swing_outward_vs_br = (
            torch.square(
                torch.clamp(
                    (
                        br_tibia_segment_outward_m
                        - bl_tibia_segment_outward_m
                        - self.cfg.bl_lifted_swing_outward_vs_br_tolerance_m
                    )
                    / self.cfg.bl_lifted_swing_outward_vs_br_tolerance_m,
                    min=0.0,
                )
            )
            * bl_lifted_swing_gate
            * rear_symmetry_ready
            * moving.float()
        )
        bl_swing_height = (
            swing_height_credit[:, BL_FOOT_INDEX]
            * rear_swing_mask[:, 0]
            * rear_symmetry_ready
            * moving.float()
        )
        self._rear_femur_tibia_segment_mirror_accum += (
            torch.norm(rear_femur_tibia_segment_residual, dim=1) * self.step_dt
        )
        self._rear_femur_tibia_z_mirror_accum += (
            torch.abs(bl_femur_to_tibia_b[:, 2] - br_femur_to_tibia_b[:, 2]) * self.step_dt
        )
        self._rear_tibia_link_mirror_accum += torch.norm(rear_tibia_link_residual, dim=1) * self.step_dt
        self._rear_tibia_segment_mirror_accum += torch.norm(rear_tibia_segment_residual, dim=1) * self.step_dt
        rear_outward_deficit = torch.clamp(
            self.cfg.rear_tibia_segment_outward_min_m + bl_tibia_to_foot_b[:, 0],
            min=0.0,
        ) + torch.clamp(
            self.cfg.rear_tibia_segment_outward_min_m - br_tibia_to_foot_b[:, 0],
            min=0.0,
        )
        self._rear_tibia_segment_outward_accum += rear_outward_deficit * self.step_dt
        bl_coxa_joint_pos = self._robot.data.joint_pos[:, self._bl_coxa_joint_id]
        br_coxa_joint_pos = self._robot.data.joint_pos[:, self._br_coxa_joint_id]
        bl_femur_joint_pos = self._robot.data.joint_pos[:, self._bl_femur_joint_id]
        br_femur_joint_pos = self._robot.data.joint_pos[:, self._br_femur_joint_id]
        bl_tibia_joint_pos = self._robot.data.joint_pos[:, self._bl_tibia_joint_id]
        br_tibia_joint_pos = self._robot.data.joint_pos[:, self._br_tibia_joint_id]
        current_bl_tibia_mean = (self._bl_tibia_accum + bl_tibia_joint_pos * self.step_dt) / torch.clamp(
            self._total_time, min=self.step_dt
        )
        current_br_tibia_mean = (self._br_tibia_accum + br_tibia_joint_pos * self.step_dt) / torch.clamp(
            self._total_time, min=self.step_dt
        )
        rear_tibia_mean_symmetry = torch.square(
            torch.clamp(
                torch.abs(current_bl_tibia_mean - current_br_tibia_mean)
                - self.cfg.rear_tibia_mean_symmetry_tolerance_rad,
                min=0.0,
            )
            / self.cfg.rear_tibia_mean_symmetry_tolerance_rad
        ) * rear_symmetry_ready * moving.float()
        rear_tibia_mean_target = (
            torch.square(
                torch.clamp(
                    torch.abs(current_bl_tibia_mean - self.cfg.rear_tibia_mean_target_rad)
                    - self.cfg.rear_tibia_mean_target_tolerance_rad,
                    min=0.0,
                )
                / self.cfg.rear_tibia_mean_target_tolerance_rad
            )
            + torch.square(
                torch.clamp(
                    torch.abs(current_br_tibia_mean - self.cfg.rear_tibia_mean_target_rad)
                    - self.cfg.rear_tibia_mean_target_tolerance_rad,
                    min=0.0,
                )
                / self.cfg.rear_tibia_mean_target_tolerance_rad
            )
        ) * rear_symmetry_ready * moving.float()
        rear_tibia_mean_curl_floor = (
            torch.square(
                torch.clamp(
                    (self.cfg.rear_tibia_curl_floor_rad - current_bl_tibia_mean)
                    / self.cfg.rear_tibia_curl_floor_tolerance_rad,
                    min=0.0,
                )
            )
            + torch.square(
                torch.clamp(
                    (self.cfg.rear_tibia_curl_floor_rad - current_br_tibia_mean)
                    / self.cfg.rear_tibia_curl_floor_tolerance_rad,
                    min=0.0,
                )
            )
        ) * rear_symmetry_ready * moving.float()
        bl_coxa_target_pos = self._processed_actions[:, self._bl_coxa_joint_id]
        br_coxa_target_pos = self._processed_actions[:, self._br_coxa_joint_id]
        bl_femur_target_pos = self._processed_actions[:, self._bl_femur_joint_id]
        br_femur_target_pos = self._processed_actions[:, self._br_femur_joint_id]
        bl_tibia_target_pos = self._processed_actions[:, self._bl_tibia_joint_id]
        br_tibia_target_pos = self._processed_actions[:, self._br_tibia_joint_id]
        rear_tibia_command_symmetry = torch.square(
            torch.clamp(
                torch.abs(bl_tibia_target_pos - br_tibia_target_pos)
                - self.cfg.rear_tibia_command_symmetry_tolerance_rad,
                min=0.0,
            )
            / self.cfg.rear_tibia_command_symmetry_tolerance_rad
        ) * moving.float()
        rear_tibia_command_target = (
            torch.square(
                torch.clamp(
                    torch.abs(bl_tibia_target_pos - self.cfg.rear_tibia_mean_target_rad)
                    - self.cfg.rear_tibia_command_target_tolerance_rad,
                    min=0.0,
                )
                / self.cfg.rear_tibia_command_target_tolerance_rad
            )
            + torch.square(
                torch.clamp(
                    torch.abs(br_tibia_target_pos - self.cfg.rear_tibia_mean_target_rad)
                    - self.cfg.rear_tibia_command_target_tolerance_rad,
                    min=0.0,
                )
                / self.cfg.rear_tibia_command_target_tolerance_rad
            )
        ) * moving.float()
        rear_tibia_command_curl_floor = (
            torch.square(
                torch.clamp(
                    (self.cfg.rear_tibia_curl_floor_rad - bl_tibia_target_pos)
                    / self.cfg.rear_tibia_curl_floor_tolerance_rad,
                    min=0.0,
                )
            )
            + torch.square(
                torch.clamp(
                    (self.cfg.rear_tibia_curl_floor_rad - br_tibia_target_pos)
                    / self.cfg.rear_tibia_curl_floor_tolerance_rad,
                    min=0.0,
                )
            )
        ) * moving.float()
        rear_swing_mask = 1.0 - contact_f[:, [BL_FOOT_INDEX, BR_FOOT_INDEX]]
        bl_fast_swing_rear_command_pose = (
            torch.square(
                torch.clamp(
                    (self.cfg.bl_fast_swing_coxa_min_rad - bl_coxa_target_pos)
                    / self.cfg.bl_fast_swing_coxa_tolerance_rad,
                    min=0.0,
                )
            )
            + torch.square(
                torch.clamp(
                    (self.cfg.bl_fast_swing_femur_min_rad - bl_femur_target_pos)
                    / self.cfg.bl_fast_swing_femur_tolerance_rad,
                    min=0.0,
                )
            )
            + torch.square(
                torch.clamp(
                    (bl_tibia_target_pos - self.cfg.bl_fast_swing_tibia_max_rad)
                    / self.cfg.bl_fast_swing_tibia_tolerance_rad,
                    min=0.0,
                )
            )
        ) * rear_swing_mask[:, 0] * rear_symmetry_ready * visual_cleanup_progress
        rear_swing_femur_floor_raw = torch.stack(
            (
                torch.square(
                    torch.clamp(
                        (bl_femur_joint_pos - self.cfg.rear_swing_femur_floor_rad)
                        / self.cfg.rear_swing_femur_floor_tolerance_rad,
                        min=0.0,
                    )
                ),
                torch.square(
                    torch.clamp(
                        (br_femur_joint_pos - self.cfg.rear_swing_femur_floor_rad)
                        / self.cfg.rear_swing_femur_floor_tolerance_rad,
                        min=0.0,
                    )
                ),
            ),
            dim=1,
        )
        rear_swing_femur_floor = (
            torch.sum(rear_swing_femur_floor_raw * rear_swing_mask, dim=1)
            / torch.clamp(torch.sum(rear_swing_mask, dim=1), min=1.0)
        ) * rear_symmetry_ready * moving.float()
        bl_swing_femur_floor = (
            rear_swing_femur_floor_raw[:, 0] * rear_swing_mask[:, 0] * rear_symmetry_ready * moving.float()
        )
        rear_swing_femur_command_floor_raw = torch.stack(
            (
                torch.square(
                    torch.clamp(
                        (bl_femur_target_pos - self.cfg.rear_swing_femur_floor_rad)
                        / self.cfg.rear_swing_femur_floor_tolerance_rad,
                        min=0.0,
                    )
                ),
                torch.square(
                    torch.clamp(
                        (br_femur_target_pos - self.cfg.rear_swing_femur_floor_rad)
                        / self.cfg.rear_swing_femur_floor_tolerance_rad,
                        min=0.0,
                    )
                ),
            ),
            dim=1,
        )
        rear_swing_femur_command_floor = (
            torch.sum(rear_swing_femur_command_floor_raw * rear_swing_mask, dim=1)
            / torch.clamp(torch.sum(rear_swing_mask, dim=1), min=1.0)
        ) * rear_symmetry_ready * moving.float()
        bl_swing_femur_command_floor = (
            rear_swing_femur_command_floor_raw[:, 0]
            * rear_swing_mask[:, 0]
            * rear_symmetry_ready
            * moving.float()
        )
        rear_tibia_tuck_envelope = (
            torch.square(
                torch.clamp(
                    (bl_tibia_joint_pos - self.cfg.rear_tibia_tuck_limit_rad)
                    / self.cfg.rear_tibia_tuck_tolerance_rad,
                    min=0.0,
                )
            )
            + torch.square(
                torch.clamp(
                    (br_tibia_joint_pos - self.cfg.rear_tibia_tuck_limit_rad)
                    / self.cfg.rear_tibia_tuck_tolerance_rad,
                    min=0.0,
                )
            )
        ) * moving.float()
        bl_tibia_tuck = torch.square(
            torch.clamp(
                (bl_tibia_joint_pos - self.cfg.bl_tibia_tuck_limit_rad) / self.cfg.bl_tibia_tuck_tolerance_rad,
                min=0.0,
            )
        ) * moving.float()
        bl_fast_swing_rear_joint_pose = (
            torch.square(
                torch.clamp(
                    (self.cfg.bl_fast_swing_coxa_min_rad - bl_coxa_joint_pos)
                    / self.cfg.bl_fast_swing_coxa_tolerance_rad,
                    min=0.0,
                )
            )
            + torch.square(
                torch.clamp(
                    (self.cfg.bl_fast_swing_femur_min_rad - bl_femur_joint_pos)
                    / self.cfg.bl_fast_swing_femur_tolerance_rad,
                    min=0.0,
                )
            )
            + torch.square(
                torch.clamp(
                    (bl_tibia_joint_pos - self.cfg.bl_fast_swing_tibia_max_rad)
                    / self.cfg.bl_fast_swing_tibia_tolerance_rad,
                    min=0.0,
                )
            )
        ) * rear_swing_mask[:, 0] * rear_symmetry_ready * visual_cleanup_progress
        bl_tibia_pair_offset = torch.square(
            torch.clamp(
                (
                    bl_tibia_joint_pos
                    - br_tibia_joint_pos
                    - self.cfg.bl_tibia_pair_offset_limit_rad
                )
                / self.cfg.bl_tibia_pair_offset_tolerance_rad,
                min=0.0,
            )
        ) * moving.float()
        bl_coxa_pair_offset = torch.square(
            torch.clamp(
                (
                    bl_coxa_joint_pos
                    - br_coxa_joint_pos
                    - self.cfg.bl_coxa_pair_offset_limit_rad
                )
                / self.cfg.bl_coxa_pair_offset_tolerance_rad,
                min=0.0,
            )
        ) * moving.float()
        bl_femur_pair_offset = torch.square(
            torch.clamp(
                (
                    bl_femur_joint_pos
                    - br_femur_joint_pos
                    - self.cfg.bl_femur_pair_offset_limit_rad
                )
                / self.cfg.bl_femur_pair_offset_tolerance_rad,
                min=0.0,
            )
        ) * moving.float()
        bl_coxa_command_pair_offset = torch.square(
            torch.clamp(
                (
                    bl_coxa_target_pos
                    - br_coxa_target_pos
                    - self.cfg.bl_coxa_pair_offset_limit_rad
                )
                / self.cfg.bl_coxa_pair_offset_tolerance_rad,
                min=0.0,
            )
        ) * moving.float()
        bl_femur_command_pair_offset = torch.square(
            torch.clamp(
                (
                    bl_femur_target_pos
                    - br_femur_target_pos
                    - self.cfg.bl_femur_pair_offset_limit_rad
                )
                / self.cfg.bl_femur_pair_offset_tolerance_rad,
                min=0.0,
            )
        ) * moving.float()
        bl_tibia_command_pair_offset = torch.square(
            torch.clamp(
                (
                    bl_tibia_target_pos
                    - br_tibia_target_pos
                    - self.cfg.bl_tibia_pair_offset_limit_rad
                )
                / self.cfg.bl_tibia_pair_offset_tolerance_rad,
                min=0.0,
            )
        ) * moving.float()
        rear_coxa_joint_symmetry = torch.square(
            torch.clamp(
                torch.abs(bl_coxa_joint_pos - br_coxa_joint_pos)
                - self.cfg.rear_coxa_joint_symmetry_tolerance_rad,
                min=0.0,
            )
            / self.cfg.rear_coxa_joint_symmetry_tolerance_rad
        ) * moving.float()
        rear_femur_joint_symmetry = torch.square(
            torch.clamp(
                torch.abs(bl_femur_joint_pos - br_femur_joint_pos)
                - self.cfg.rear_femur_joint_symmetry_tolerance_rad,
                min=0.0,
            )
            / self.cfg.rear_femur_joint_symmetry_tolerance_rad
        ) * moving.float()
        rear_tibia_joint_symmetry = torch.square(
            torch.clamp(
                torch.abs(bl_tibia_joint_pos - br_tibia_joint_pos)
                - self.cfg.rear_tibia_joint_symmetry_tolerance_rad,
                min=0.0,
            )
            / self.cfg.rear_tibia_joint_symmetry_tolerance_rad
        ) * moving.float()
        rear_coxa_command_symmetry = torch.square(
            torch.clamp(
                torch.abs(bl_coxa_target_pos - br_coxa_target_pos)
                - self.cfg.rear_coxa_command_symmetry_tolerance_rad,
                min=0.0,
            )
            / self.cfg.rear_coxa_command_symmetry_tolerance_rad
        ) * moving.float()
        rear_femur_command_symmetry = torch.square(
            torch.clamp(
                torch.abs(bl_femur_target_pos - br_femur_target_pos)
                - self.cfg.rear_femur_command_symmetry_tolerance_rad,
                min=0.0,
            )
            / self.cfg.rear_femur_command_symmetry_tolerance_rad
        ) * moving.float()
        rear_tibia_command_symmetry_abs = torch.square(
            torch.clamp(
                torch.abs(bl_tibia_target_pos - br_tibia_target_pos)
                - self.cfg.rear_tibia_command_symmetry_abs_tolerance_rad,
                min=0.0,
            )
            / self.cfg.rear_tibia_command_symmetry_abs_tolerance_rad
        ) * moving.float()
        bl_rear_distal_joint_pos = torch.stack((bl_femur_joint_pos, bl_tibia_joint_pos), dim=1)
        br_rear_distal_joint_pos = torch.stack((br_femur_joint_pos, br_tibia_joint_pos), dim=1)
        bl_rear_distal_target_pos = torch.stack((bl_femur_target_pos, bl_tibia_target_pos), dim=1)
        br_rear_distal_target_pos = torch.stack((br_femur_target_pos, br_tibia_target_pos), dim=1)
        bl_phase_delayed_rear_coxa_joint_symmetry = (
            torch.square(
                torch.clamp(
                    torch.abs(bl_coxa_joint_pos - delayed_br_rear_coxa_joint_pos)
                    - self.cfg.bl_phase_delayed_rear_coxa_joint_symmetry_tolerance_rad,
                    min=0.0,
                )
                / self.cfg.bl_phase_delayed_rear_coxa_joint_symmetry_tolerance_rad
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_phase_delayed_rear_coxa_joint_symmetry = (
            torch.square(
                torch.clamp(
                    torch.abs(bl_coxa_joint_pos - delayed_br_rear_coxa_joint_pos)
                    - self.cfg.bl_lifted_phase_delayed_rear_coxa_joint_symmetry_tolerance_rad,
                    min=0.0,
                )
                / self.cfg.bl_lifted_phase_delayed_rear_coxa_joint_symmetry_tolerance_rad
            )
            * bl_lifted_swing_gate
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_phase_delayed_rear_coxa_command_symmetry = (
            torch.square(
                torch.clamp(
                    torch.abs(bl_coxa_target_pos - delayed_br_rear_coxa_target_pos)
                    - self.cfg.bl_phase_delayed_rear_coxa_command_symmetry_tolerance_rad,
                    min=0.0,
                )
                / self.cfg.bl_phase_delayed_rear_coxa_command_symmetry_tolerance_rad
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_phase_delayed_rear_coxa_command_symmetry = (
            torch.square(
                torch.clamp(
                    torch.abs(bl_coxa_target_pos - delayed_br_rear_coxa_target_pos)
                    - self.cfg.bl_lifted_phase_delayed_rear_coxa_command_symmetry_tolerance_rad,
                    min=0.0,
                )
                / self.cfg.bl_lifted_phase_delayed_rear_coxa_command_symmetry_tolerance_rad
            )
            * bl_lifted_swing_gate
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_phase_delayed_rear_distal_joint_symmetry = (
            torch.mean(
                torch.square(
                    torch.clamp(
                        torch.abs(bl_rear_distal_joint_pos - delayed_br_rear_distal_joint_pos)
                        - self.cfg.bl_phase_delayed_rear_distal_joint_symmetry_tolerance_rad,
                        min=0.0,
                    )
                    / self.cfg.bl_phase_delayed_rear_distal_joint_symmetry_tolerance_rad
                ),
                dim=1,
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_phase_delayed_rear_distal_joint_symmetry = (
            torch.mean(
                torch.square(
                    torch.clamp(
                        torch.abs(bl_rear_distal_joint_pos - delayed_br_rear_distal_joint_pos)
                        - self.cfg.bl_lifted_phase_delayed_rear_distal_joint_symmetry_tolerance_rad,
                        min=0.0,
                    )
                    / self.cfg.bl_lifted_phase_delayed_rear_distal_joint_symmetry_tolerance_rad
                ),
                dim=1,
            )
            * bl_lifted_swing_gate
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_phase_delayed_rear_distal_command_symmetry = (
            torch.mean(
                torch.square(
                    torch.clamp(
                        torch.abs(bl_rear_distal_target_pos - delayed_br_rear_distal_target_pos)
                        - self.cfg.bl_phase_delayed_rear_distal_command_symmetry_tolerance_rad,
                        min=0.0,
                    )
                    / self.cfg.bl_phase_delayed_rear_distal_command_symmetry_tolerance_rad
                ),
                dim=1,
            )
            * rear_swing_mask[:, 0]
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        bl_lifted_phase_delayed_rear_distal_command_symmetry = (
            torch.mean(
                torch.square(
                    torch.clamp(
                        torch.abs(bl_rear_distal_target_pos - delayed_br_rear_distal_target_pos)
                        - self.cfg.bl_lifted_phase_delayed_rear_distal_command_symmetry_tolerance_rad,
                        min=0.0,
                    )
                    / self.cfg.bl_lifted_phase_delayed_rear_distal_command_symmetry_tolerance_rad
                ),
                dim=1,
            )
            * bl_lifted_swing_gate
            * delayed_br_swing
            * rear_phase_delay_ready
            * rear_symmetry_ready
            * moving.float()
        )
        self._rear_coxa_joint_abs_diff_accum += torch.abs(bl_coxa_joint_pos - br_coxa_joint_pos) * self.step_dt
        self._rear_femur_joint_abs_diff_accum += torch.abs(bl_femur_joint_pos - br_femur_joint_pos) * self.step_dt
        self._rear_tibia_joint_abs_diff_accum += torch.abs(bl_tibia_joint_pos - br_tibia_joint_pos) * self.step_dt
        self._update_episode_diagnostics(foot_contacts, undesired_contacts, foot_pos_w, foot_vel_w)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "command_progress": command_progress * self.cfg.command_progress_reward_scale * self.step_dt,
            "command_progress_floor": command_progress_floor
            * self.cfg.command_progress_floor_reward_scale
            * self.step_dt,
            "command_overspeed_l2": command_overspeed * self.cfg.command_overspeed_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "yaw_drift_l2": yaw_drift * self.cfg.yaw_drift_reward_scale * self.step_dt,
            "lateral_velocity_l2": lateral_velocity * self.cfg.lateral_velocity_reward_scale * self.step_dt,
            "lateral_position_l2": lateral_position * self.cfg.lateral_position_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "base_height_floor_l2": base_height_floor * self.cfg.base_height_floor_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "ang_vel_z_l2": ang_vel_z_error * self.cfg.ang_vel_z_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "pitch_forward_l2": pitch_forward * self.cfg.pitch_forward_reward_scale * self.step_dt,
            "front_up_target_l2": front_up_target * self.cfg.front_up_target_reward_scale * self.step_dt,
            "front_down_hinge_l2": front_down_hinge * self.cfg.front_down_hinge_reward_scale * self.step_dt,
            "front_up_progress_bonus": front_up_progress_bonus
            * self.cfg.front_up_progress_bonus_reward_scale
            * self.step_dt,
            "front_up_posture": front_up_posture
            * self.cfg.front_up_posture_reward_scale
            * self.step_dt,
            "front_up_posture_command": front_up_posture_command
            * self.cfg.front_up_posture_command_reward_scale
            * self.step_dt,
            "stance_foot_slip": stance_foot_slip * self.cfg.stance_foot_slip_reward_scale * self.step_dt,
            "low_swing_drag": low_swing_drag * self.cfg.low_swing_drag_reward_scale * self.step_dt,
            "bl_low_swing_drag": bl_low_swing_drag * self.cfg.bl_low_swing_drag_reward_scale * self.step_dt,
            "bl_low_swing_time": bl_low_swing_time * self.cfg.bl_low_swing_time_reward_scale * self.step_dt,
            "swing_foot_speed_l2": swing_foot_speed * self.cfg.swing_foot_speed_reward_scale * self.step_dt,
            "support_distribution": support_distribution * self.cfg.support_distribution_reward_scale * self.step_dt,
            "contact_balance": contact_balance * self.cfg.contact_balance_reward_scale * self.step_dt,
            "diagonal_trot": diagonal_trot * self.cfg.diagonal_trot_reward_scale * self.step_dt,
            "diagonal_pair_sync": diagonal_pair_sync * self.cfg.diagonal_pair_sync_reward_scale * self.step_dt,
            "side_antiphase": side_antiphase * self.cfg.side_antiphase_reward_scale * self.step_dt,
            "rear_contact_antiphase": rear_contact_antiphase
            * self.cfg.rear_contact_antiphase_reward_scale
            * self.step_dt,
            "touchdown_cadence_l2": touchdown_cadence * self.cfg.touchdown_cadence_reward_scale * self.step_dt,
            "touchdown_rate_balance": touchdown_rate_balance * self.cfg.touchdown_rate_balance_reward_scale * self.step_dt,
            "rapid_touchdown": rapid_touchdown * self.cfg.rapid_touchdown_reward_scale * self.step_dt,
            "touchdown_stride": touchdown_stride_reward * self.cfg.touchdown_stride_reward_scale * self.step_dt,
            "touchdown_short_stride": short_stride * self.cfg.touchdown_short_stride_reward_scale * self.step_dt,
            "stride_ema_shortfall": stride_ema_shortfall * self.cfg.stride_ema_shortfall_reward_scale * self.step_dt,
            "stance_anchor_slip": stance_anchor_slip * self.cfg.stance_anchor_slip_reward_scale * self.step_dt,
            "rear_stride_balance": rear_stride_balance * self.cfg.rear_stride_balance_reward_scale * self.step_dt,
            "rear_stance_duration_balance": rear_stance_duration_balance
            * self.cfg.rear_stance_duration_balance_reward_scale
            * self.step_dt,
            "rear_touchdown_rate_balance": rear_touchdown_rate_balance
            * self.cfg.rear_touchdown_rate_balance_reward_scale
            * self.step_dt,
            "rear_contact_fraction_balance": rear_contact_fraction_balance
            * self.cfg.rear_contact_fraction_balance_reward_scale
            * self.step_dt,
            "rear_contact_fraction_target": rear_contact_fraction_target
            * self.cfg.rear_contact_fraction_target_reward_scale
            * self.step_dt,
            "bl_contact_fraction_excess": bl_contact_fraction_excess
            * self.cfg.bl_contact_fraction_excess_reward_scale
            * self.step_dt,
            "rear_stance_slip_balance": rear_stance_slip_balance
            * self.cfg.rear_stance_slip_balance_reward_scale
            * self.step_dt,
            "rear_left_stance_slip_excess": rear_left_stance_slip_excess
            * self.cfg.rear_left_stance_slip_excess_reward_scale
            * self.step_dt,
            "rear_lateral_mirror": rear_lateral_mirror * self.cfg.rear_lateral_mirror_reward_scale * self.step_dt,
            "rear_tibia_mean_symmetry": rear_tibia_mean_symmetry
            * self.cfg.rear_tibia_mean_symmetry_reward_scale
            * self.step_dt,
            "rear_tibia_mean_target": rear_tibia_mean_target
            * self.cfg.rear_tibia_mean_target_reward_scale
            * self.step_dt,
            "rear_tibia_mean_curl_floor": rear_tibia_mean_curl_floor
            * self.cfg.rear_tibia_mean_curl_floor_reward_scale
            * self.step_dt,
            "rear_tibia_command_symmetry": rear_tibia_command_symmetry
            * self.cfg.rear_tibia_command_symmetry_reward_scale
            * self.step_dt,
            "rear_tibia_command_target": rear_tibia_command_target
            * self.cfg.rear_tibia_command_target_reward_scale
            * self.step_dt,
            "rear_tibia_command_curl_floor": rear_tibia_command_curl_floor
            * self.cfg.rear_tibia_command_curl_floor_reward_scale
            * self.step_dt,
            "rear_tibia_tuck_envelope": rear_tibia_tuck_envelope
            * self.cfg.rear_tibia_tuck_envelope_reward_scale
            * self.step_dt,
            "bl_tibia_tuck": bl_tibia_tuck * self.cfg.bl_tibia_tuck_reward_scale * self.step_dt,
            "bl_tibia_pair_offset": bl_tibia_pair_offset
            * self.cfg.bl_tibia_pair_offset_reward_scale
            * self.step_dt,
            "bl_coxa_pair_offset": bl_coxa_pair_offset
            * self.cfg.bl_coxa_pair_offset_reward_scale
            * self.step_dt,
            "bl_femur_pair_offset": bl_femur_pair_offset
            * self.cfg.bl_femur_pair_offset_reward_scale
            * self.step_dt,
            "bl_coxa_command_pair_offset": bl_coxa_command_pair_offset
            * self.cfg.bl_coxa_command_pair_offset_reward_scale
            * self.step_dt,
            "bl_femur_command_pair_offset": bl_femur_command_pair_offset
            * self.cfg.bl_femur_command_pair_offset_reward_scale
            * self.step_dt,
            "bl_tibia_command_pair_offset": bl_tibia_command_pair_offset
            * self.cfg.bl_tibia_command_pair_offset_reward_scale
            * self.step_dt,
            "rear_coxa_joint_symmetry": rear_coxa_joint_symmetry
            * self.cfg.rear_coxa_joint_symmetry_reward_scale
            * self.step_dt,
            "rear_femur_joint_symmetry": rear_femur_joint_symmetry
            * self.cfg.rear_femur_joint_symmetry_reward_scale
            * self.step_dt,
            "rear_tibia_joint_symmetry": rear_tibia_joint_symmetry
            * self.cfg.rear_tibia_joint_symmetry_reward_scale
            * self.step_dt,
            "rear_coxa_command_symmetry": rear_coxa_command_symmetry
            * self.cfg.rear_coxa_command_symmetry_reward_scale
            * self.step_dt,
            "rear_femur_command_symmetry": rear_femur_command_symmetry
            * self.cfg.rear_femur_command_symmetry_reward_scale
            * self.step_dt,
            "rear_tibia_command_symmetry_abs": rear_tibia_command_symmetry_abs
            * self.cfg.rear_tibia_command_symmetry_abs_reward_scale
            * self.step_dt,
            "rear_femur_tibia_segment_mirror": rear_femur_tibia_segment_mirror
            * self.cfg.rear_femur_tibia_segment_mirror_reward_scale
            * self.step_dt,
            "rear_femur_tibia_z_mirror": rear_femur_tibia_z_mirror
            * self.cfg.rear_femur_tibia_z_mirror_reward_scale
            * self.step_dt,
            "rear_tibia_link_mirror": rear_tibia_link_mirror
            * self.cfg.rear_tibia_link_mirror_reward_scale
            * self.step_dt,
            "rear_tibia_axis_mirror": rear_tibia_axis_mirror
            * self.cfg.rear_tibia_axis_mirror_reward_scale
            * self.step_dt,
            "bl_lifted_tibia_axis_mirror": bl_lifted_tibia_axis_mirror
            * self.cfg.bl_lifted_tibia_axis_mirror_reward_scale
            * self.step_dt,
            "rear_tibia_primary_axis_mirror": rear_tibia_primary_axis_mirror
            * self.cfg.rear_tibia_primary_axis_mirror_reward_scale
            * self.step_dt,
            "bl_lifted_tibia_primary_axis_mirror": bl_lifted_tibia_primary_axis_mirror
            * self.cfg.bl_lifted_tibia_primary_axis_mirror_reward_scale
            * self.step_dt,
            "rear_tibia_secondary_axis_mirror": rear_tibia_secondary_axis_mirror
            * self.cfg.rear_tibia_secondary_axis_mirror_reward_scale
            * self.step_dt,
            "bl_lifted_tibia_secondary_axis_mirror": bl_lifted_tibia_secondary_axis_mirror
            * self.cfg.bl_lifted_tibia_secondary_axis_mirror_reward_scale
            * self.step_dt,
            "rear_tibia_mesh_long_axis_mirror": rear_tibia_mesh_long_axis_mirror
            * self.cfg.rear_tibia_mesh_long_axis_mirror_reward_scale
            * self.step_dt,
            "bl_lifted_tibia_mesh_long_axis_mirror": bl_lifted_tibia_mesh_long_axis_mirror
            * self.cfg.bl_lifted_tibia_mesh_long_axis_mirror_reward_scale
            * self.step_dt,
            "rear_tibia_mesh_secondary_axis_mirror": rear_tibia_mesh_secondary_axis_mirror
            * self.cfg.rear_tibia_mesh_secondary_axis_mirror_reward_scale
            * self.step_dt,
            "bl_lifted_tibia_mesh_secondary_axis_mirror": bl_lifted_tibia_mesh_secondary_axis_mirror
            * self.cfg.bl_lifted_tibia_mesh_secondary_axis_mirror_reward_scale
            * self.step_dt,
            "rear_tibia_mesh_long_lateral_excess": rear_tibia_mesh_long_lateral_excess
            * self.cfg.rear_tibia_mesh_long_lateral_excess_reward_scale
            * self.step_dt,
            "bl_swing_tibia_mesh_long_lateral_excess": bl_swing_tibia_mesh_long_lateral_excess
            * self.cfg.bl_swing_tibia_mesh_long_lateral_excess_reward_scale
            * self.step_dt,
            "rear_tibia_mesh_secondary_lateral_excess": rear_tibia_mesh_secondary_lateral_excess
            * self.cfg.rear_tibia_mesh_secondary_lateral_excess_reward_scale
            * self.step_dt,
            "bl_swing_tibia_mesh_secondary_lateral_excess": bl_swing_tibia_mesh_secondary_lateral_excess
            * self.cfg.bl_swing_tibia_mesh_secondary_lateral_excess_reward_scale
            * self.step_dt,
            "rear_tibia_mesh_secondary_lateral_target": rear_tibia_mesh_secondary_lateral_target
            * self.cfg.rear_tibia_mesh_secondary_lateral_target_reward_scale
            * self.step_dt,
            "bl_swing_tibia_mesh_secondary_lateral_target": bl_swing_tibia_mesh_secondary_lateral_target
            * self.cfg.bl_swing_tibia_mesh_secondary_lateral_target_reward_scale
            * self.step_dt,
            "br_swing_tibia_mesh_secondary_lateral_target": br_swing_tibia_mesh_secondary_lateral_target
            * self.cfg.br_swing_tibia_mesh_secondary_lateral_target_reward_scale
            * self.step_dt,
            "bl_fast_swing_rear_joint_pose": bl_fast_swing_rear_joint_pose
            * self.cfg.bl_fast_swing_rear_joint_pose_reward_scale
            * self.step_dt,
            "bl_fast_swing_rear_command_pose": bl_fast_swing_rear_command_pose
            * self.cfg.bl_fast_swing_rear_command_pose_reward_scale
            * self.step_dt,
            "bl_tibia_link_outward_vs_br": bl_tibia_link_outward_vs_br
            * self.cfg.bl_tibia_link_outward_vs_br_reward_scale
            * self.step_dt,
            "bl_lifted_swing_tibia_link_outward_vs_br": bl_lifted_swing_tibia_link_outward_vs_br
            * self.cfg.bl_lifted_swing_tibia_link_outward_vs_br_reward_scale
            * self.step_dt,
            "bl_tibia_link_outward_floor": bl_tibia_link_outward_floor
            * self.cfg.bl_tibia_link_outward_floor_reward_scale
            * self.step_dt,
            "bl_lifted_swing_tibia_link_outward_floor": bl_lifted_swing_tibia_link_outward_floor
            * self.cfg.bl_lifted_swing_tibia_link_outward_floor_reward_scale
            * self.step_dt,
            "rear_tibia_segment_mirror": rear_tibia_segment_mirror
            * self.cfg.rear_tibia_segment_mirror_reward_scale
            * self.step_dt,
            "rear_tibia_segment_outward": rear_tibia_segment_outward
            * self.cfg.rear_tibia_segment_outward_reward_scale
            * self.step_dt,
            "rear_tibia_segment_outward_target": rear_tibia_segment_outward_target
            * self.cfg.rear_tibia_segment_outward_target_reward_scale
            * self.step_dt,
            "rear_swing_tibia_segment_outward": rear_swing_tibia_segment_outward
            * self.cfg.rear_swing_tibia_segment_outward_reward_scale
            * self.step_dt,
            "bl_swing_tibia_segment_outward": bl_swing_tibia_segment_outward
            * self.cfg.bl_swing_tibia_segment_outward_reward_scale
            * self.step_dt,
            "bl_lifted_swing_tibia_segment_outward": bl_lifted_swing_tibia_segment_outward
            * self.cfg.bl_lifted_swing_tibia_segment_outward_reward_scale
            * self.step_dt,
            "bl_phase_delayed_tibia_outward": bl_phase_delayed_tibia_outward
            * self.cfg.bl_phase_delayed_tibia_outward_reward_scale
            * self.step_dt,
            "bl_phase_delayed_tibia_link_outward": bl_phase_delayed_tibia_link_outward
            * self.cfg.bl_phase_delayed_tibia_link_outward_reward_scale
            * self.step_dt,
            "bl_lifted_phase_delayed_tibia_link_outward": bl_lifted_phase_delayed_tibia_link_outward
            * self.cfg.bl_lifted_phase_delayed_tibia_link_outward_reward_scale
            * self.step_dt,
            "bl_phase_delayed_tibia_link_mirror": bl_phase_delayed_tibia_link_mirror
            * self.cfg.bl_phase_delayed_tibia_link_mirror_reward_scale
            * self.step_dt,
            "bl_lifted_phase_delayed_tibia_link_mirror": bl_lifted_phase_delayed_tibia_link_mirror
            * self.cfg.bl_lifted_phase_delayed_tibia_link_mirror_reward_scale
            * self.step_dt,
            "bl_phase_delayed_femur_tibia_mirror": bl_phase_delayed_femur_tibia_mirror
            * self.cfg.bl_phase_delayed_femur_tibia_mirror_reward_scale
            * self.step_dt,
            "bl_lifted_phase_delayed_femur_tibia_mirror": bl_lifted_phase_delayed_femur_tibia_mirror
            * self.cfg.bl_lifted_phase_delayed_femur_tibia_mirror_reward_scale
            * self.step_dt,
            "bl_phase_delayed_tibia_primary_axis_mirror": bl_phase_delayed_tibia_primary_axis_mirror
            * self.cfg.bl_phase_delayed_tibia_primary_axis_mirror_reward_scale
            * self.step_dt,
            "bl_lifted_phase_delayed_tibia_primary_axis_mirror": (
                bl_lifted_phase_delayed_tibia_primary_axis_mirror
                * self.cfg.bl_lifted_phase_delayed_tibia_primary_axis_mirror_reward_scale
                * self.step_dt
            ),
            "bl_phase_delayed_tibia_secondary_axis_mirror": bl_phase_delayed_tibia_secondary_axis_mirror
            * self.cfg.bl_phase_delayed_tibia_secondary_axis_mirror_reward_scale
            * self.step_dt,
            "bl_lifted_phase_delayed_tibia_secondary_axis_mirror": (
                bl_lifted_phase_delayed_tibia_secondary_axis_mirror
                * self.cfg.bl_lifted_phase_delayed_tibia_secondary_axis_mirror_reward_scale
                * self.step_dt
            ),
            "bl_phase_delayed_rear_coxa_joint_symmetry": bl_phase_delayed_rear_coxa_joint_symmetry
            * self.cfg.bl_phase_delayed_rear_coxa_joint_symmetry_reward_scale
            * self.step_dt,
            "bl_lifted_phase_delayed_rear_coxa_joint_symmetry": (
                bl_lifted_phase_delayed_rear_coxa_joint_symmetry
                * self.cfg.bl_lifted_phase_delayed_rear_coxa_joint_symmetry_reward_scale
                * self.step_dt
            ),
            "bl_phase_delayed_rear_coxa_command_symmetry": bl_phase_delayed_rear_coxa_command_symmetry
            * self.cfg.bl_phase_delayed_rear_coxa_command_symmetry_reward_scale
            * self.step_dt,
            "bl_lifted_phase_delayed_rear_coxa_command_symmetry": (
                bl_lifted_phase_delayed_rear_coxa_command_symmetry
                * self.cfg.bl_lifted_phase_delayed_rear_coxa_command_symmetry_reward_scale
                * self.step_dt
            ),
            "bl_phase_delayed_rear_distal_joint_symmetry": bl_phase_delayed_rear_distal_joint_symmetry
            * self.cfg.bl_phase_delayed_rear_distal_joint_symmetry_reward_scale
            * self.step_dt,
            "bl_lifted_phase_delayed_rear_distal_joint_symmetry": bl_lifted_phase_delayed_rear_distal_joint_symmetry
            * self.cfg.bl_lifted_phase_delayed_rear_distal_joint_symmetry_reward_scale
            * self.step_dt,
            "bl_phase_delayed_rear_distal_command_symmetry": bl_phase_delayed_rear_distal_command_symmetry
            * self.cfg.bl_phase_delayed_rear_distal_command_symmetry_reward_scale
            * self.step_dt,
            "bl_lifted_phase_delayed_rear_distal_command_symmetry": (
                bl_lifted_phase_delayed_rear_distal_command_symmetry
                * self.cfg.bl_lifted_phase_delayed_rear_distal_command_symmetry_reward_scale
                * self.step_dt
            ),
            "bl_phase_delayed_foot_height": bl_phase_delayed_foot_height
            * self.cfg.bl_phase_delayed_foot_height_reward_scale
            * self.step_dt,
            "rear_swing_outward_mean_balance": rear_swing_outward_mean_balance
            * self.cfg.rear_swing_outward_mean_balance_reward_scale
            * self.step_dt,
            "bl_swing_outward_mean_floor": bl_swing_outward_mean_floor
            * self.cfg.bl_swing_outward_mean_floor_reward_scale
            * self.step_dt,
            "bl_swing_outward_vs_br": bl_swing_outward_vs_br
            * self.cfg.bl_swing_outward_vs_br_reward_scale
            * self.step_dt,
            "bl_lifted_swing_outward_vs_br": bl_lifted_swing_outward_vs_br
            * self.cfg.bl_lifted_swing_outward_vs_br_reward_scale
            * self.step_dt,
            "rear_swing_femur_floor": rear_swing_femur_floor
            * self.cfg.rear_swing_femur_floor_reward_scale
            * self.step_dt,
            "bl_swing_femur_floor": bl_swing_femur_floor
            * self.cfg.bl_swing_femur_floor_reward_scale
            * self.step_dt,
            "rear_swing_femur_command_floor": rear_swing_femur_command_floor
            * self.cfg.rear_swing_femur_command_floor_reward_scale
            * self.step_dt,
            "bl_swing_femur_command_floor": bl_swing_femur_command_floor
            * self.cfg.bl_swing_femur_command_floor_reward_scale
            * self.step_dt,
            "swing_height": swing_height * self.cfg.swing_height_reward_scale * self.step_dt,
            "bl_swing_height": bl_swing_height * self.cfg.bl_swing_height_reward_scale * self.step_dt,
            "short_stance": short_stance * self.cfg.short_stance_reward_scale * self.step_dt,
            "short_swing": short_swing * self.cfg.short_swing_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._br_tibia_segment_outward_history[:, :-1] = self._br_tibia_segment_outward_history[:, 1:].clone()
        self._br_tibia_segment_outward_history[:, -1] = br_tibia_segment_outward_m.detach()
        self._br_tibia_link_outward_history[:, :-1] = self._br_tibia_link_outward_history[:, 1:].clone()
        self._br_tibia_link_outward_history[:, -1] = br_tibia_link_outward_m.detach()
        self._br_tibia_link_pos_history[:, :-1] = self._br_tibia_link_pos_history[:, 1:].clone()
        self._br_tibia_link_pos_history[:, -1] = br_tibia_body_pos_b.detach()
        self._br_femur_tibia_vec_history[:, :-1] = self._br_femur_tibia_vec_history[:, 1:].clone()
        self._br_femur_tibia_vec_history[:, -1] = br_femur_to_tibia_b.detach()
        self._br_tibia_axis_x_history[:, :-1] = self._br_tibia_axis_x_history[:, 1:].clone()
        self._br_tibia_axis_x_history[:, -1] = tibia_axis_x_b[:, 1].detach()
        self._br_tibia_axis_y_history[:, :-1] = self._br_tibia_axis_y_history[:, 1:].clone()
        self._br_tibia_axis_y_history[:, -1] = tibia_axis_y_b[:, 1].detach()
        self._br_rear_coxa_joint_pos_history[:, :-1] = self._br_rear_coxa_joint_pos_history[:, 1:].clone()
        self._br_rear_coxa_joint_pos_history[:, -1] = br_coxa_joint_pos.detach()
        self._br_rear_coxa_target_pos_history[:, :-1] = self._br_rear_coxa_target_pos_history[:, 1:].clone()
        self._br_rear_coxa_target_pos_history[:, -1] = br_coxa_target_pos.detach()
        self._br_rear_distal_joint_pos_history[:, :-1] = self._br_rear_distal_joint_pos_history[:, 1:].clone()
        self._br_rear_distal_joint_pos_history[:, -1] = br_rear_distal_joint_pos.detach()
        self._br_rear_distal_target_pos_history[:, :-1] = self._br_rear_distal_target_pos_history[:, 1:].clone()
        self._br_rear_distal_target_pos_history[:, -1] = br_rear_distal_target_pos.detach()
        self._br_foot_height_history[:, :-1] = self._br_foot_height_history[:, 1:].clone()
        self._br_foot_height_history[:, -1] = foot_height[:, BR_FOOT_INDEX].detach()
        self._br_swing_history[:, :-1] = self._br_swing_history[:, 1:].clone()
        self._br_swing_history[:, -1] = rear_swing_mask[:, 1].detach()
        self._rear_swing_outward_accum = current_rear_swing_outward_accum.detach()
        self._rear_swing_outward_time = current_rear_swing_outward_time.detach()
        self._rear_phase_history_steps = torch.clamp(
            self._rear_phase_history_steps + 1.0,
            max=float(self._rear_phase_history_len),
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        forces = self._contact_sensor.data.net_forces_w_history
        base_contact = torch.amax(torch.norm(forces[:, :, self._base_id], dim=-1), dim=(1, 2)) > 1.0
        base_too_low = self._robot.data.root_pos_w[:, 2] < 0.08
        base_tilted = torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) > 0.9
        invalid_state = (
            ~torch.isfinite(self._robot.data.root_pos_w).all(dim=1)
            | ~torch.isfinite(self._robot.data.joint_pos).all(dim=1)
            | ~torch.isfinite(self._robot.data.joint_vel).all(dim=1)
        )
        died = base_contact | base_too_low | base_tilted | invalid_state
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = (
            self._robot.data.default_joint_pos[env_ids] + self._action_center_offset.unsqueeze(0)
        )
        self._sample_commands(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        joint_pos += self._reset_joint_pos_offset.unsqueeze(0)
        joint_pos += torch.empty_like(joint_pos).uniform_(-0.08, 0.08)
        joint_limits = self._robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, joint_limits[:, :, 0], joint_limits[:, :, 1])
        joint_vel += torch.empty_like(joint_vel).uniform_(-0.05, 0.05)
        if torch.any(self._reset_joint_pos_offset != 0.0):
            self._processed_actions[env_ids] = joint_pos

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        root_state[:, 2] += self.cfg.front_up_reset_root_z_offset
        root_state[:, 0:2] += torch.empty(len(env_ids), 2, device=self.device).uniform_(-0.1, 0.1)
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        front_up = torch.full_like(yaw, math.radians(self.cfg.front_up_reset_deg))
        yaw_half = 0.5 * yaw
        front_up_half = 0.5 * front_up
        yaw_w = torch.cos(yaw_half)
        yaw_z = torch.sin(yaw_half)
        front_up_w = torch.cos(front_up_half)
        front_up_x = torch.sin(front_up_half)
        root_state[:, 3] = yaw_w * front_up_w
        root_state[:, 4] = yaw_w * front_up_x
        root_state[:, 5] = yaw_z * front_up_x
        root_state[:, 6] = yaw_z * front_up_w
        root_state[:, 7:] += torch.empty_like(root_state[:, 7:]).uniform_(-0.05, 0.05)
        self._initial_yaw_w[env_ids] = yaw
        self._initial_root_xy_w[env_ids] = root_state[:, :2]

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._base_lin_vel_est_b[env_ids] = 0.0
        self._base_lin_vel_est_initialized[env_ids] = False
        self._base_lin_vel_est_last_step = -1

        extras = {}
        episode_time = torch.clamp(self._total_time[env_ids], min=self.step_dt)
        extras["Episode_Metric/avg_forward_vel_y"] = torch.mean(
            self._forward_distance_traveled[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_abs_lateral_vel_x"] = torch.mean(
            self._lateral_distance_abs[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_abs_roll_rate"] = torch.mean(
            self._roll_rate_abs_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_abs_pitch_rate"] = torch.mean(
            self._pitch_rate_abs_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_abs_yaw_rate"] = torch.mean(self._yaw_rate_abs_accum[env_ids] / episode_time).item()
        extras["Episode_Metric/avg_tilt_xy"] = torch.mean(self._tilt_accum[env_ids] / episode_time).item()
        extras["Episode_Metric/avg_front_up_deg"] = torch.mean(
            self._front_up_deg_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_projected_gravity_y"] = torch.mean(
            self._projected_gravity_y_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_base_height"] = torch.mean(self._base_height_accum[env_ids] / episode_time).item()
        extras["Episode_Metric/avg_foot_contact_count"] = torch.mean(
            self._foot_contact_count_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/foot_touchdown_rate_hz"] = torch.mean(
            torch.sum(self._step_counts[env_ids], dim=1) / episode_time
        ).item()
        extras["Episode_Metric/avg_undesired_contact_count"] = torch.mean(
            self._undesired_contact_count_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_stance_foot_xy_speed"] = torch.mean(
            self._stance_foot_speed_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_low_swing_foot_xy_speed"] = torch.mean(
            self._low_swing_foot_speed_accum[env_ids] / episode_time
        ).item()
        per_foot_contact_fraction = self._per_foot_contact_time_accum[env_ids] / episode_time.unsqueeze(1)
        per_foot_stance_slip = self._per_foot_stance_slip_accum[env_ids] / torch.clamp(
            self._per_foot_contact_time_accum[env_ids], min=self.step_dt
        )
        per_foot_low_swing_drag = self._per_foot_low_swing_drag_accum[env_ids] / torch.clamp(
            self._per_foot_low_swing_time_accum[env_ids], min=self.step_dt
        )
        per_foot_touchdown_rate = self._step_counts[env_ids] / episode_time.unsqueeze(1)
        per_foot_stance_duration = self._completed_stance_time_accum[env_ids] / torch.clamp(
            self._completed_stance_count[env_ids], min=1.0
        )
        per_foot_swing_duration = self._completed_swing_time_accum[env_ids] / torch.clamp(
            self._completed_swing_count[env_ids], min=1.0
        )
        extras["Episode_Metric/avg_touchdown_stride_ema"] = torch.mean(self._stride_length_ema[env_ids]).item()
        extras["Episode_Metric/avg_latest_touchdown_stride"] = torch.mean(
            self._latest_touchdown_stride[env_ids]
        ).item()
        extras["Episode_Metric/avg_completed_stance_duration"] = torch.mean(per_foot_stance_duration).item()
        extras["Episode_Metric/avg_completed_swing_duration"] = torch.mean(per_foot_swing_duration).item()
        extras["Episode_Metric/contact_fraction_imbalance"] = torch.mean(
            torch.abs(per_foot_contact_fraction - torch.mean(per_foot_contact_fraction, dim=1, keepdim=True))
        ).item()
        extras["Episode_Metric/touchdown_rate_imbalance"] = torch.mean(
            torch.abs(per_foot_touchdown_rate - torch.mean(per_foot_touchdown_rate, dim=1, keepdim=True))
        ).item()
        extras["Episode_Metric/rear_stride_abs_diff"] = torch.mean(
            torch.abs(self._stride_length_ema[env_ids, BL_FOOT_INDEX] - self._stride_length_ema[env_ids, BR_FOOT_INDEX])
        ).item()
        extras["Episode_Metric/rear_contact_fraction_abs_diff"] = torch.mean(
            torch.abs(
                per_foot_contact_fraction[:, BL_FOOT_INDEX] - per_foot_contact_fraction[:, BR_FOOT_INDEX]
            )
        ).item()
        extras["Episode_Metric/rear_touchdown_rate_abs_diff"] = torch.mean(
            torch.abs(per_foot_touchdown_rate[:, BL_FOOT_INDEX] - per_foot_touchdown_rate[:, BR_FOOT_INDEX])
        ).item()
        extras["Episode_Metric/rear_stance_duration_abs_diff"] = torch.mean(
            torch.abs(per_foot_stance_duration[:, BL_FOOT_INDEX] - per_foot_stance_duration[:, BR_FOOT_INDEX])
        ).item()
        extras["Episode_Metric/rear_stance_slip_abs_diff"] = torch.mean(
            torch.abs(per_foot_stance_slip[:, BL_FOOT_INDEX] - per_foot_stance_slip[:, BR_FOOT_INDEX])
        ).item()
        extras["Episode_Metric/avg_bl_tibia_rad"] = torch.mean(
            self._bl_tibia_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_br_tibia_rad"] = torch.mean(
            self._br_tibia_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_bl_br_tibia_offset_rad"] = torch.mean(
            self._bl_br_tibia_offset_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/rear_tibia_mean_abs_diff_rad"] = torch.mean(
            torch.abs((self._bl_tibia_accum[env_ids] - self._br_tibia_accum[env_ids]) / episode_time)
        ).item()
        extras["Episode_Metric/rear_tibia_link_mirror_error_m"] = torch.mean(
            self._rear_tibia_link_mirror_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/rear_tibia_segment_mirror_error_m"] = torch.mean(
            self._rear_tibia_segment_mirror_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/rear_tibia_segment_outward_deficit_m"] = torch.mean(
            self._rear_tibia_segment_outward_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/rear_femur_tibia_segment_mirror_error_m"] = torch.mean(
            self._rear_femur_tibia_segment_mirror_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/rear_femur_tibia_z_mirror_error_m"] = torch.mean(
            self._rear_femur_tibia_z_mirror_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/rear_coxa_joint_abs_diff_rad"] = torch.mean(
            self._rear_coxa_joint_abs_diff_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/rear_femur_joint_abs_diff_rad"] = torch.mean(
            self._rear_femur_joint_abs_diff_accum[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/rear_tibia_joint_abs_diff_rad"] = torch.mean(
            self._rear_tibia_joint_abs_diff_accum[env_ids] / episode_time
        ).item()
        for foot_index, foot_name in enumerate(FOOT_BODY_NAMES):
            extras[f"Episode_Metric/{foot_name}_contact_fraction"] = torch.mean(
                per_foot_contact_fraction[:, foot_index]
            ).item()
            extras[f"Episode_Metric/{foot_name}_touchdown_rate_hz"] = torch.mean(
                per_foot_touchdown_rate[:, foot_index]
            ).item()
            extras[f"Episode_Metric/{foot_name}_stride_ema"] = torch.mean(
                self._stride_length_ema[env_ids, foot_index]
            ).item()
            extras[f"Episode_Metric/{foot_name}_stance_slip_mps"] = torch.mean(
                per_foot_stance_slip[:, foot_index]
            ).item()
            extras[f"Episode_Metric/{foot_name}_low_swing_drag_mps"] = torch.mean(
                per_foot_low_swing_drag[:, foot_index]
            ).item()
            extras[f"Episode_Metric/{foot_name}_low_swing_fraction"] = torch.mean(
                self._per_foot_low_swing_time_accum[env_ids, foot_index] / episode_time
            ).item()
            extras[f"Episode_Metric/{foot_name}_completed_stance_duration"] = torch.mean(
                per_foot_stance_duration[:, foot_index]
            ).item()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/base_contact_or_fall"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"] = extras

        self._stance_anchor_xy[env_ids] = 0.0
        self._last_touchdown_xy[env_ids] = 0.0
        self._has_touchdown[env_ids] = False
        self._latest_touchdown_stride[env_ids] = 0.0
        self._stride_length_ema[env_ids] = 0.0
        self._stance_time[env_ids] = 0.0
        self._swing_time[env_ids] = 0.0
        self._prev_foot_contacts[env_ids] = False
        self._step_counts[env_ids] = 0.0
        self._forward_distance_traveled[env_ids] = 0.0
        self._lateral_distance_abs[env_ids] = 0.0
        self._command_forward_sum[env_ids] = 0.0
        self._total_time[env_ids] = 0.0
        self._roll_rate_abs_accum[env_ids] = 0.0
        self._pitch_rate_abs_accum[env_ids] = 0.0
        self._yaw_rate_abs_accum[env_ids] = 0.0
        self._tilt_accum[env_ids] = 0.0
        self._front_up_deg_accum[env_ids] = 0.0
        self._projected_gravity_y_accum[env_ids] = 0.0
        self._base_height_accum[env_ids] = 0.0
        self._foot_contact_count_accum[env_ids] = 0.0
        self._undesired_contact_count_accum[env_ids] = 0.0
        self._stance_foot_speed_accum[env_ids] = 0.0
        self._low_swing_foot_speed_accum[env_ids] = 0.0
        self._per_foot_contact_time_accum[env_ids] = 0.0
        self._per_foot_stance_slip_accum[env_ids] = 0.0
        self._per_foot_low_swing_time_accum[env_ids] = 0.0
        self._per_foot_low_swing_drag_accum[env_ids] = 0.0
        self._completed_stance_time_accum[env_ids] = 0.0
        self._completed_stance_count[env_ids] = 0.0
        self._completed_swing_time_accum[env_ids] = 0.0
        self._completed_swing_count[env_ids] = 0.0
        self._bl_tibia_accum[env_ids] = 0.0
        self._br_tibia_accum[env_ids] = 0.0
        self._bl_br_tibia_offset_accum[env_ids] = 0.0
        self._rear_tibia_link_mirror_accum[env_ids] = 0.0
        self._rear_tibia_segment_mirror_accum[env_ids] = 0.0
        self._rear_tibia_segment_outward_accum[env_ids] = 0.0
        self._rear_femur_tibia_segment_mirror_accum[env_ids] = 0.0
        self._rear_femur_tibia_z_mirror_accum[env_ids] = 0.0
        self._rear_coxa_joint_abs_diff_accum[env_ids] = 0.0
        self._rear_femur_joint_abs_diff_accum[env_ids] = 0.0
        self._rear_tibia_joint_abs_diff_accum[env_ids] = 0.0
        self._rear_swing_outward_accum[env_ids] = 0.0
        self._rear_swing_outward_time[env_ids] = 0.0
        self._br_tibia_segment_outward_history[env_ids] = 0.0
        self._br_tibia_link_outward_history[env_ids] = 0.0
        self._br_tibia_link_pos_history[env_ids] = 0.0
        self._br_femur_tibia_vec_history[env_ids] = 0.0
        self._br_tibia_axis_x_history[env_ids] = 0.0
        self._br_tibia_axis_y_history[env_ids] = 0.0
        self._br_rear_coxa_joint_pos_history[env_ids] = 0.0
        self._br_rear_coxa_target_pos_history[env_ids] = 0.0
        self._br_rear_distal_joint_pos_history[env_ids] = 0.0
        self._br_rear_distal_target_pos_history[env_ids] = 0.0
        self._br_foot_height_history[env_ids] = 0.0
        self._br_swing_history[env_ids] = 0.0
        self._rear_phase_history_steps[env_ids] = 0.0

    def _sample_commands(self, env_ids: torch.Tensor):
        commands = torch.zeros(len(env_ids), 3, device=self.device)
        standing = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_standing_envs
        moving = ~standing
        count = int(torch.count_nonzero(moving).item())
        if count:
            commands[moving, 0] = torch.empty(count, device=self.device).uniform_(*self.cfg.lin_vel_x_range)
            commands[moving, 1] = torch.empty(count, device=self.device).uniform_(*self.cfg.lin_vel_y_range)
            commands[moving, 2] = torch.empty(count, device=self.device).uniform_(*self.cfg.ang_vel_z_range)
        self._commands[env_ids] = commands

    def _get_foot_contacts(self) -> torch.Tensor:
        forces = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids]
        return torch.amax(torch.norm(forces, dim=-1), dim=1) > 1.0

    def _update_episode_progress(self):
        self._forward_distance_traveled += self._robot.data.root_lin_vel_b[:, 1] * self.step_dt
        self._lateral_distance_abs += torch.abs(self._robot.data.root_lin_vel_b[:, 0]) * self.step_dt
        self._command_forward_sum += self._commands[:, 1] * self.step_dt
        self._total_time += self.step_dt

    def _update_episode_diagnostics(
        self,
        foot_contacts: torch.Tensor,
        undesired_contacts: torch.Tensor,
        foot_pos_w: torch.Tensor,
        foot_vel_w: torch.Tensor,
    ):
        self._roll_rate_abs_accum += torch.abs(self._robot.data.root_ang_vel_b[:, 0]) * self.step_dt
        self._pitch_rate_abs_accum += torch.abs(self._robot.data.root_ang_vel_b[:, 1]) * self.step_dt
        self._yaw_rate_abs_accum += torch.abs(self._robot.data.root_ang_vel_b[:, 2]) * self.step_dt
        self._tilt_accum += torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) * self.step_dt
        self._front_up_deg_accum += (
            torch.rad2deg(torch.asin(torch.clamp(-self._robot.data.projected_gravity_b[:, 1], -1.0, 1.0)))
            * self.step_dt
        )
        self._projected_gravity_y_accum += self._robot.data.projected_gravity_b[:, 1] * self.step_dt
        self._base_height_accum += self._robot.data.root_pos_w[:, 2] * self.step_dt
        contact_f = foot_contacts.float()
        self._foot_contact_count_accum += torch.sum(contact_f, dim=1) * self.step_dt
        self._undesired_contact_count_accum += undesired_contacts * self.step_dt
        foot_xy_speed = torch.norm(foot_vel_w[:, :, :2], dim=2)
        stance_count = torch.sum(contact_f, dim=1)
        avg_stance_speed = torch.sum(foot_xy_speed * contact_f, dim=1) / (stance_count + 1.0e-6)
        foot_height = torch.clamp(foot_pos_w[:, :, 2] - self._terrain.env_origins[:, 2].unsqueeze(1), min=0.0)
        low_swing = (1.0 - contact_f) * (foot_height < 0.025).float()
        avg_low_swing_speed = torch.sum(foot_xy_speed * low_swing, dim=1) / (torch.sum(low_swing, dim=1) + 1.0e-6)
        self._stance_foot_speed_accum += avg_stance_speed * self.step_dt
        self._low_swing_foot_speed_accum += avg_low_swing_speed * self.step_dt
        self._per_foot_contact_time_accum += contact_f * self.step_dt
        self._per_foot_stance_slip_accum += foot_xy_speed * contact_f * self.step_dt
        self._per_foot_low_swing_time_accum += low_swing * self.step_dt
        self._per_foot_low_swing_drag_accum += foot_xy_speed * low_swing * self.step_dt
        bl_tibia_joint_pos = self._robot.data.joint_pos[:, self._bl_tibia_joint_id]
        br_tibia_joint_pos = self._robot.data.joint_pos[:, self._br_tibia_joint_id]
        self._bl_tibia_accum += bl_tibia_joint_pos * self.step_dt
        self._br_tibia_accum += br_tibia_joint_pos * self.step_dt
        self._bl_br_tibia_offset_accum += (bl_tibia_joint_pos - br_tibia_joint_pos) * self.step_dt

    def _update_footfall_state(self, foot_contacts: torch.Tensor, first_contact: torch.Tensor, foot_pos_w: torch.Tensor):
        contact_f = foot_contacts.float()
        self._stance_time += contact_f * self.step_dt
        self._swing_time += (1.0 - contact_f) * self.step_dt

        touchdown = first_contact.bool()
        if torch.any(touchdown):
            new_xy = foot_pos_w[:, :, :2]
            raw_stride = torch.norm(new_xy - self._last_touchdown_xy, dim=2)
            stride = torch.where(self._has_touchdown, raw_stride, torch.zeros_like(raw_stride))
            alpha = self.cfg.stride_length_ema_alpha
            self._stride_length_ema = torch.where(touchdown, (1.0 - alpha) * self._stride_length_ema + alpha * stride, self._stride_length_ema)
            self._latest_touchdown_stride = torch.where(touchdown, stride, self._latest_touchdown_stride)
            self._last_touchdown_xy = torch.where(touchdown.unsqueeze(-1), new_xy, self._last_touchdown_xy)
            self._has_touchdown = torch.where(touchdown, torch.ones_like(self._has_touchdown), self._has_touchdown)
            self._stance_anchor_xy = torch.where(touchdown.unsqueeze(-1), new_xy, self._stance_anchor_xy)
            self._step_counts += touchdown.float()
            self._stance_time = torch.where(touchdown, torch.zeros_like(self._stance_time), self._stance_time)
            self._swing_time = torch.where(touchdown, torch.zeros_like(self._swing_time), self._swing_time)
        self._prev_foot_contacts = foot_contacts.clone()
