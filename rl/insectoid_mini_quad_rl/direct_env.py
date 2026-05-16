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

from .articulation import INSECTOID_MINI_QUAD_CFG


FOOT_BODY_NAMES = ("BL_FOOT", "BR_FOOT", "ML_FOOT", "MR_FOOT")
FOOT_BODY_PATTERN = "BL_FOOT|BR_FOOT|ML_FOOT|MR_FOOT"
NON_FOOT_BODY_PATTERN = "base_link|.*_coxa_1|.*_femur_1|.*_tibia_1"


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
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 1.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -8.0
    pitch_forward_reward_scale = -8.0
    stance_foot_slip_reward_scale = -0.45
    low_swing_drag_reward_scale = -0.25
    support_distribution_reward_scale = -0.30
    touchdown_stride_reward_scale = 2.2
    # STEP HEIGHT TEST: kept reward for clearer swing lift.
    swing_height_reward_scale = 0.70
    short_stance_reward_scale = -0.35
    short_swing_reward_scale = -0.35
    min_touchdown_stride = 0.05
    touchdown_stride_target = 0.14
    swing_height_min = 0.030
    swing_height_target = 0.075
    min_stance_time = 0.12
    min_swing_time = 0.10

    # Kept only for diagnostics and success metrics, not as direct reward terms.
    stride_length_ema_alpha = 0.2


@configclass
class InsectoidMiniQuadFlatPlayEnvCfg(InsectoidMiniQuadFlatEnvCfg):
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
        self._yaw_rate_abs_accum = torch.zeros(self.num_envs, device=self.device)
        self._tilt_accum = torch.zeros(self.num_envs, device=self.device)
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

        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._feet_ids, feet_names = self._contact_sensor.find_bodies(FOOT_BODY_PATTERN, preserve_order=True)
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(NON_FOOT_BODY_PATTERN)
        self._feet_body_ids, _ = self._robot.find_bodies(FOOT_BODY_PATTERN, preserve_order=True)

        if tuple(feet_names) != FOOT_BODY_NAMES:
            raise RuntimeError(f"Unexpected quad foot order {feet_names}; expected {FOOT_BODY_NAMES}.")

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "command_progress",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "pitch_forward_l2",
                "stance_foot_slip",
                "low_swing_drag",
                "support_distribution",
                "touchdown_stride",
                "swing_height",
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
        self._actions = actions.clone()
        target = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        alpha = self.cfg.action_target_filter_alpha
        self._processed_actions = alpha * target + (1.0 - alpha) * self._processed_actions

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        foot_contact_f = self._get_foot_contacts().float()
        stance_time_obs = torch.clamp(self._stance_time / self.cfg.min_stance_time, 0.0, 1.0)
        swing_time_obs = torch.clamp(self._swing_time / self.cfg.min_swing_time, 0.0, 1.0)
        obs = torch.cat(
            (
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._actions,
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
        command_direction_y = torch.sign(self._commands[:, 1])
        commanded_progress_y = command_direction_y * self._robot.data.root_lin_vel_b[:, 1]
        command_progress = torch.clamp(commanded_progress_y / torch.clamp(command_speed_y, min=0.05), -1.0, 1.0)
        command_progress = command_progress * (command_speed_y > 0.1).float()
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        moving = torch.norm(self._commands[:, :2], dim=1) > 0.1
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * moving

        forces = self._contact_sensor.data.net_forces_w_history
        undesired_contact = torch.amax(
            torch.norm(forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1
        ) > 1.0
        undesired_contacts = torch.sum(undesired_contact.float(), dim=1)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        pitch_forward = torch.square(self._robot.data.projected_gravity_b[:, 1])
        foot_vel_w = self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :]
        contact_f = foot_contacts.float()
        foot_xy_speed = torch.norm(foot_vel_w[:, :, :2], dim=2)
        stance_count = torch.sum(contact_f, dim=1)
        stance_foot_slip = torch.sum(foot_xy_speed * contact_f, dim=1) / (stance_count + 1.0e-6)
        foot_height = torch.clamp(foot_pos_w[:, :, 2] - self._terrain.env_origins[:, 2].unsqueeze(1), min=0.0)
        low_swing = (1.0 - contact_f) * (foot_height < 0.025).float()
        low_swing_drag = torch.sum(foot_xy_speed * low_swing, dim=1) / (torch.sum(low_swing, dim=1) + 1.0e-6)
        swing_mask = 1.0 - contact_f
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
        self._update_episode_diagnostics(foot_contacts, undesired_contacts, foot_pos_w, foot_vel_w)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "command_progress": command_progress * self.cfg.command_progress_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "pitch_forward_l2": pitch_forward * self.cfg.pitch_forward_reward_scale * self.step_dt,
            "stance_foot_slip": stance_foot_slip * self.cfg.stance_foot_slip_reward_scale * self.step_dt,
            "low_swing_drag": low_swing_drag * self.cfg.low_swing_drag_reward_scale * self.step_dt,
            "support_distribution": support_distribution * self.cfg.support_distribution_reward_scale * self.step_dt,
            "touchdown_stride": touchdown_stride_reward * self.cfg.touchdown_stride_reward_scale * self.step_dt,
            "swing_height": swing_height * self.cfg.swing_height_reward_scale * self.step_dt,
            "short_stance": short_stance * self.cfg.short_stance_reward_scale * self.step_dt,
            "short_swing": short_swing * self.cfg.short_swing_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
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
        self._processed_actions[env_ids] = self._robot.data.default_joint_pos[env_ids]
        self._sample_commands(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        joint_pos += torch.empty_like(joint_pos).uniform_(-0.08, 0.08)
        joint_limits = self._robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, joint_limits[:, :, 0], joint_limits[:, :, 1])
        joint_vel += torch.empty_like(joint_vel).uniform_(-0.05, 0.05)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        root_state[:, 0:2] += torch.empty(len(env_ids), 2, device=self.device).uniform_(-0.1, 0.1)
        yaw = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        root_state[:, 3] = torch.cos(0.5 * yaw)
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = torch.sin(0.5 * yaw)
        root_state[:, 7:] += torch.empty_like(root_state[:, 7:]).uniform_(-0.05, 0.05)

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        extras = {}
        episode_time = torch.clamp(self._total_time[env_ids], min=self.step_dt)
        extras["Episode_Metric/avg_forward_vel_y"] = torch.mean(
            self._forward_distance_traveled[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_abs_lateral_vel_x"] = torch.mean(
            self._lateral_distance_abs[env_ids] / episode_time
        ).item()
        extras["Episode_Metric/avg_abs_yaw_rate"] = torch.mean(self._yaw_rate_abs_accum[env_ids] / episode_time).item()
        extras["Episode_Metric/avg_tilt_xy"] = torch.mean(self._tilt_accum[env_ids] / episode_time).item()
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
        self._yaw_rate_abs_accum[env_ids] = 0.0
        self._tilt_accum[env_ids] = 0.0
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
        self._yaw_rate_abs_accum += torch.abs(self._robot.data.root_ang_vel_b[:, 2]) * self.step_dt
        self._tilt_accum += torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) * self.step_dt
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
