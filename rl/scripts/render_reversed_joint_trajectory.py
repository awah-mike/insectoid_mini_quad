#!/usr/bin/env python3
"""Replay a recorded insectoid joint trajectory, optionally reversed."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
COXA_LEFT = (0, 2)  # BL, ML in joint order
COXA_RIGHT = (1, 3)  # BR, MR in joint order
FOOT_NAMES = ("BL_FOOT", "BR_FOOT", "ML_FOOT", "MR_FOOT")
JOINT_NAMES = (
    "BL_coxa_joint",
    "BR_coxa_joint",
    "ML_coxa_joint",
    "MR_coxa_joint",
    "BL_femur_joint",
    "BR_femur_joint",
    "ML_femur_joint",
    "MR_femur_joint",
    "BL_tibia_joint",
    "BR_tibia_joint",
    "ML_tibia_joint",
    "MR_tibia_joint",
)
LEG_JOINT_INDICES = {
    "BL_FOOT": (0, 4, 8),
    "BR_FOOT": (1, 5, 9),
    "ML_FOOT": (2, 6, 10),
    "MR_FOOT": (3, 7, 11),
}
REAR_FOOT_INDICES = (0, 1)
LOCAL_JOINT_SELECTIONS = {
    "femur_tibia": (1, 2),
    "all": (0, 1, 2),
}
LEG_KINEMATICS = {
    "BL_FOOT": {
        "joint_origins": ((-0.1, -0.2, -0.002), (-0.052707, 0.045235, 0.0315), (-0.180001376, 0.0, -0.0499962435)),
        "joint_rpy": ((math.pi, 0.0, 0.0), (-math.pi / 2.0, 0.0, -0.349044398), (0.0, 0.0, 0.0)),
        "foot_offset": (-0.2, 0.0, 0.069),
    },
    "BR_FOOT": {
        "joint_origins": ((0.1, -0.2, -0.002), (0.052707, -0.045235, -0.0315), (0.180001376, 0.0, 0.0499962435)),
        "joint_rpy": ((0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, -0.349044398), (0.0, 0.0, 0.0)),
        "foot_offset": (0.2, 0.0, -0.069),
    },
    "ML_FOOT": {
        "joint_origins": ((-0.1, 0.0, -0.002), (-0.065, 0.02448, 0.0315), (-0.18, 0.0, -0.05)),
        "joint_rpy": ((math.pi, 0.0, 0.0), (-math.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        "foot_offset": (-0.2, 0.0, 0.069),
    },
    "MR_FOOT": {
        "joint_origins": ((0.1, 0.0, -0.002), (0.065, -0.02448, -0.0315), (0.18, 0.0, 0.05)),
        "joint_rpy": ((0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        "foot_offset": (0.2, 0.0, -0.069),
    },
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0")
parser.add_argument("--trajectory-npz", type=Path, required=True)
parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/renders/reversed_joint_trajectory.mp4")
parser.add_argument("--summary-output", type=Path, default=None)
parser.add_argument("--source", choices=("action", "target", "actual"), default="actual")
parser.add_argument("--start-s", type=float, default=0.72)
parser.add_argument("--end-s", type=float, default=1.30)
parser.add_argument("--duration", type=float, default=6.0)
parser.add_argument("--playback-speed", type=float, default=1.0)
parser.add_argument("--reverse", action="store_true")
parser.add_argument("--cycle", action="store_true")
parser.add_argument("--start-from-default", action="store_true")
parser.add_argument("--init-reversed-root-velocity", action="store_true")
parser.add_argument("--init-joint-velocity", action="store_true")
parser.add_argument("--root-z-offset", type=float, default=0.04)
parser.add_argument("--root-yaw-rate-damping", type=float, default=0.0)
parser.add_argument("--root-yaw-servo-kp", type=float, default=0.0)
parser.add_argument("--root-yaw-servo-kd", type=float, default=0.0)
parser.add_argument("--root-yaw-servo-sign", type=float, default=1.0)
parser.add_argument("--root-yaw-servo-max-angvel", type=float, default=1.0)
parser.add_argument("--yaw-feedback-kp", type=float, default=0.0)
parser.add_argument("--yaw-rate-feedback-kd", type=float, default=0.0)
parser.add_argument("--yaw-feedback-sign", type=float, default=1.0)
parser.add_argument("--max-yaw-feedback", type=float, default=0.20)
parser.add_argument("--front-up-deg", type=float, default=0.0)
parser.add_argument("--front-up-kp", type=float, default=0.0)
parser.add_argument("--front-up-kd", type=float, default=0.0)
parser.add_argument("--front-up-max-angvel", type=float, default=1.0)
parser.add_argument("--bl-tibia-bias-deg", type=float, default=0.0)
parser.add_argument("--coxa-action-scale", type=float, default=1.0)
parser.add_argument("--femur-action-scale", type=float, default=1.0)
parser.add_argument("--tibia-action-scale", type=float, default=1.0)
parser.add_argument("--root-body-vel-x-servo-target", type=float, default=None)
parser.add_argument("--root-body-vel-x-servo-alpha", type=float, default=0.0)
parser.add_argument("--root-body-vel-x-servo-start-s", type=float, default=0.0)
parser.add_argument("--root-body-vel-y-servo-target", type=float, default=None)
parser.add_argument("--root-body-vel-y-servo-alpha", type=float, default=0.0)
parser.add_argument("--root-body-vel-y-servo-start-s", type=float, default=0.0)
parser.add_argument("--root-body-vel-y-servo-lift-relief", type=float, default=0.0)
parser.add_argument("--sideways-remap-scale", type=float, default=0.0)
parser.add_argument("--sideways-remap-sign", type=float, default=1.0)
parser.add_argument("--sideways-remap-x-keep-scale", type=float, default=1.0)
parser.add_argument("--sideways-remap-y-keep-scale", type=float, default=1.0)
parser.add_argument("--sideways-remap-joints", choices=tuple(LOCAL_JOINT_SELECTIONS), default="all")
parser.add_argument("--sideways-remap-damping", type=float, default=0.01)
parser.add_argument("--sideways-remap-max-action-delta", type=float, default=0.60)
parser.add_argument("--swing-lift-height-m", type=float, default=0.0)
parser.add_argument("--swing-lift-joints", choices=tuple(LOCAL_JOINT_SELECTIONS), default="femur_tibia")
parser.add_argument("--swing-lift-rear-scale", type=float, default=1.0)
parser.add_argument("--swing-lift-middle-scale", type=float, default=1.0)
parser.add_argument("--swing-lift-damping", type=float, default=0.01)
parser.add_argument("--swing-lift-max-action-delta", type=float, default=0.35)
parser.add_argument("--swing-lift-action-gain", type=float, default=1.0)
parser.add_argument("--swing-lift-dilate-steps", type=int, default=0)
parser.add_argument("--direct-lift-femur-action", type=float, default=0.0)
parser.add_argument("--direct-lift-tibia-action", type=float, default=0.0)
parser.add_argument("--direct-lift-rear-scale", type=float, default=1.0)
parser.add_argument("--direct-lift-middle-scale", type=float, default=1.0)
parser.add_argument("--direct-lift-max-action-delta", type=float, default=0.45)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--camera-eye", type=float, nargs=3, default=(3.8, -0.2, 2.6))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(0.0, -0.45, 0.12))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import insectoid_mini_quad_rl  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


SOURCE_KEYS = {
    "action": "stabilized_actions",
    "target": "target_joint_pos",
    "actual": "actual_joint_pos",
}


def quat_wxyz_to_rpy_np(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=1)


def quat_wxyz_to_roll(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def quat_wxyz_to_yaw(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_wxyz_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_wxyz_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def rpy_to_matrix(rpy: tuple[float, float, float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return torch.tensor(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        device=device,
        dtype=dtype,
    )


def z_rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack(
        (
            torch.stack((cos_t, -sin_t, zeros), dim=-1),
            torch.stack((sin_t, cos_t, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=-2,
    )


def transform_points(rot: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return torch.matmul(rot, points[:, :, None]).squeeze(-1)


def leg_foot_position_and_jacobian(leg_name: str, joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = LEG_KINEMATICS[leg_name]
    num_envs = joint_pos.shape[0]
    device = joint_pos.device
    dtype = joint_pos.dtype
    position = torch.zeros(num_envs, 3, device=device, dtype=dtype)
    rotation = torch.eye(3, device=device, dtype=dtype).expand(num_envs, 3, 3).clone()
    joint_positions = []
    joint_axes = []

    for joint_angle, origin_xyz, origin_rpy in zip(
        joint_pos.transpose(0, 1),
        cfg["joint_origins"],
        cfg["joint_rpy"],
        strict=True,
    ):
        origin = torch.tensor(origin_xyz, device=device, dtype=dtype).expand(num_envs, 3)
        origin_rot = rpy_to_matrix(origin_rpy, device, dtype).expand(num_envs, 3, 3)
        position = position + transform_points(rotation, origin)
        axis_frame_rotation = torch.matmul(rotation, origin_rot)
        axis_z = axis_frame_rotation[:, :, 2]
        joint_positions.append(position)
        joint_axes.append(axis_z)
        rotation = torch.matmul(axis_frame_rotation, z_rotation_matrix(joint_angle))

    foot_offset = torch.tensor(cfg["foot_offset"], device=device, dtype=dtype).expand(num_envs, 3)
    foot_position = position + transform_points(rotation, foot_offset)
    jacobian_columns = [
        torch.cross(axis, foot_position - joint_position, dim=-1)
        for axis, joint_position in zip(joint_axes, joint_positions, strict=True)
    ]
    return foot_position, torch.stack(jacobian_columns, dim=-1)


def compute_cyclic_swing_lift_pulses(foot_contacts: np.ndarray, dilate_steps: int = 0) -> np.ndarray:
    airborne = ~foot_contacts.astype(bool)
    num_samples, num_feet = airborne.shape
    pulses = np.zeros((num_samples, num_feet), dtype=np.float32)
    if num_samples == 0:
        return pulses
    if dilate_steps > 0:
        dilated = airborne.copy()
        for offset in range(1, dilate_steps + 1):
            dilated |= np.roll(airborne, offset, axis=0)
            dilated |= np.roll(airborne, -offset, axis=0)
        airborne = dilated

    for foot_index in range(num_feet):
        foot_airborne = airborne[:, foot_index]
        if not np.any(foot_airborne):
            continue
        if np.all(foot_airborne):
            phase = (np.arange(num_samples, dtype=np.float32) + 0.5) / num_samples
            pulses[:, foot_index] = np.sin(np.pi * phase)
            continue
        for sample_index in np.flatnonzero(foot_airborne):
            run_start = sample_index
            while foot_airborne[(run_start - 1) % num_samples]:
                run_start = (run_start - 1) % num_samples
            run_length = 1
            while foot_airborne[(run_start + run_length) % num_samples]:
                run_length += 1
            run_offset = (sample_index - run_start) % num_samples
            phase = (run_offset + 0.5) / run_length
            pulses[sample_index, foot_index] = math.sin(math.pi * phase)
    return pulses


def compute_swing_lift_action_offsets(
    joint_pos: torch.Tensor,
    lift_pulses: torch.Tensor,
    action_scale: float,
) -> torch.Tensor:
    offsets = torch.zeros(joint_pos.shape[0], len(JOINT_NAMES), device=joint_pos.device, dtype=joint_pos.dtype)
    if args.swing_lift_height_m == 0.0:
        return offsets

    local_selection = LOCAL_JOINT_SELECTIONS[args.swing_lift_joints]
    damping_sq = args.swing_lift_damping * args.swing_lift_damping
    eye3 = torch.eye(3, device=joint_pos.device, dtype=joint_pos.dtype).expand(joint_pos.shape[0], 3, 3)
    for foot_index, foot_name in enumerate(FOOT_NAMES):
        leg_scale = args.swing_lift_rear_scale if foot_index in REAR_FOOT_INDICES else args.swing_lift_middle_scale
        if leg_scale == 0.0:
            continue
        pulse = lift_pulses[:, foot_index]
        if torch.max(torch.abs(pulse)).item() <= 1.0e-6:
            continue
        leg_joint_indices = LEG_JOINT_INDICES[foot_name]
        q_leg = joint_pos[:, list(leg_joint_indices)]
        _, jacobian = leg_foot_position_and_jacobian(foot_name, q_leg)
        jacobian_selected = jacobian[:, :, list(local_selection)]
        delta_foot = torch.zeros(joint_pos.shape[0], 3, device=joint_pos.device, dtype=joint_pos.dtype)
        delta_foot[:, 2] = args.swing_lift_height_m * leg_scale * pulse

        lhs = torch.matmul(jacobian_selected, jacobian_selected.transpose(1, 2)) + damping_sq * eye3
        solved = torch.linalg.solve(lhs, delta_foot[:, :, None]).squeeze(-1)
        delta_q_selected = torch.matmul(jacobian_selected.transpose(1, 2), solved[:, :, None]).squeeze(-1)
        delta_action = torch.clamp(
            args.swing_lift_action_gain * delta_q_selected / action_scale,
            -args.swing_lift_max_action_delta,
            args.swing_lift_max_action_delta,
        )
        for local_column, global_joint_index in enumerate(leg_joint_indices[index] for index in local_selection):
            offsets[:, global_joint_index] += delta_action[:, local_column]
    return offsets


def compute_direct_lift_action_offsets(joint_pos: torch.Tensor, lift_pulses: torch.Tensor) -> torch.Tensor:
    offsets = torch.zeros(joint_pos.shape[0], len(JOINT_NAMES), device=joint_pos.device, dtype=joint_pos.dtype)
    if args.direct_lift_femur_action == 0.0 and args.direct_lift_tibia_action == 0.0:
        return offsets

    for foot_index, foot_name in enumerate(FOOT_NAMES):
        leg_scale = args.direct_lift_rear_scale if foot_index in REAR_FOOT_INDICES else args.direct_lift_middle_scale
        if leg_scale == 0.0:
            continue
        pulse = lift_pulses[:, foot_index] * leg_scale
        if torch.max(torch.abs(pulse)).item() <= 1.0e-6:
            continue
        leg_joint_indices = LEG_JOINT_INDICES[foot_name]
        q_leg = joint_pos[:, list(leg_joint_indices)]
        _, jacobian = leg_foot_position_and_jacobian(foot_name, q_leg)
        femur_vertical_sign = torch.sign(jacobian[:, 2, 1])
        tibia_vertical_sign = torch.sign(jacobian[:, 2, 2])
        femur_vertical_sign = torch.where(femur_vertical_sign == 0.0, torch.ones_like(femur_vertical_sign), femur_vertical_sign)
        tibia_vertical_sign = torch.where(tibia_vertical_sign == 0.0, torch.ones_like(tibia_vertical_sign), tibia_vertical_sign)
        femur_delta = femur_vertical_sign * args.direct_lift_femur_action * pulse
        tibia_delta = tibia_vertical_sign * args.direct_lift_tibia_action * pulse
        offsets[:, leg_joint_indices[1]] += femur_delta
        offsets[:, leg_joint_indices[2]] += tibia_delta

    return torch.clamp(offsets, -args.direct_lift_max_action_delta, args.direct_lift_max_action_delta)


def compute_sideways_remap_action_offsets(sequence_joint_pos: torch.Tensor, action_scale: float) -> torch.Tensor:
    offsets = torch.zeros(sequence_joint_pos.shape[0], len(JOINT_NAMES), device=sequence_joint_pos.device, dtype=sequence_joint_pos.dtype)
    if args.sideways_remap_scale == 0.0:
        return offsets

    local_selection = LOCAL_JOINT_SELECTIONS[args.sideways_remap_joints]
    damping_sq = args.sideways_remap_damping * args.sideways_remap_damping
    eye3 = torch.eye(3, device=sequence_joint_pos.device, dtype=sequence_joint_pos.dtype).expand(sequence_joint_pos.shape[0], 3, 3)
    side_sign = 1.0 if args.sideways_remap_sign >= 0.0 else -1.0
    for foot_name in FOOT_NAMES:
        leg_joint_indices = LEG_JOINT_INDICES[foot_name]
        q_leg = sequence_joint_pos[:, list(leg_joint_indices)]
        foot_position, jacobian = leg_foot_position_and_jacobian(foot_name, q_leg)
        mean_foot_position = torch.mean(foot_position, dim=0, keepdim=True)
        foot_delta = foot_position - mean_foot_position
        target_foot_position = foot_position.clone()
        target_foot_position[:, 0] = (
            mean_foot_position[:, 0]
            + args.sideways_remap_x_keep_scale * foot_delta[:, 0]
            + side_sign * args.sideways_remap_scale * foot_delta[:, 1]
        )
        target_foot_position[:, 1] = mean_foot_position[:, 1] + args.sideways_remap_y_keep_scale * foot_delta[:, 1]
        delta_foot = target_foot_position - foot_position
        if torch.max(torch.abs(delta_foot)).item() <= 1.0e-6:
            continue

        jacobian_selected = jacobian[:, :, list(local_selection)]
        lhs = torch.matmul(jacobian_selected, jacobian_selected.transpose(1, 2)) + damping_sq * eye3
        solved = torch.linalg.solve(lhs, delta_foot[:, :, None]).squeeze(-1)
        delta_q_selected = torch.matmul(jacobian_selected.transpose(1, 2), solved[:, :, None]).squeeze(-1)
        delta_action = torch.clamp(
            delta_q_selected / action_scale,
            -args.sideways_remap_max_action_delta,
            args.sideways_remap_max_action_delta,
        )
        for local_column, global_joint_index in enumerate(leg_joint_indices[index] for index in local_selection):
            offsets[:, global_joint_index] += delta_action[:, local_column]
    return offsets


def front_up_quat(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half_angle = math.radians(args.front_up_deg) * 0.5
    return torch.tensor(
        (math.cos(half_angle), math.sin(half_angle), 0.0, 0.0),
        dtype=dtype,
        device=device,
    )


def summarize(
    times: np.ndarray,
    root_pos_w: np.ndarray,
    quats: np.ndarray,
    lin_vel_b: np.ndarray,
    ang_vel_b: np.ndarray,
    dones: np.ndarray,
    non_foot_contacts: np.ndarray,
    foot_contacts: np.ndarray,
    foot_xy_speed: np.ndarray,
    foot_heights: np.ndarray,
    lift_pulses: np.ndarray,
    lift_action_offsets: np.ndarray,
) -> dict[str, object]:
    rpy = quat_wxyz_to_rpy_np(quats)
    rpy[:, 2] = np.unwrap(rpy[:, 2])
    rpy_deg = np.rad2deg(rpy)
    rpy_deg[:, 2] -= rpy_deg[0, 2]

    def stats(name: str, values: np.ndarray) -> dict[str, float]:
        return {
            f"{name}_mean_deg": float(np.mean(values)),
            f"{name}_std_deg": float(np.std(values)),
            f"{name}_min_deg": float(np.min(values)),
            f"{name}_max_deg": float(np.max(values)),
            f"{name}_peak_to_peak_deg": float(np.ptp(values)),
            f"{name}_max_abs_deg": float(np.max(np.abs(values))),
        }

    displacement = root_pos_w[-1] - root_pos_w[0]
    summary: dict[str, object] = {
        "duration_s": float(times[-1] - times[0]) if len(times) else 0.0,
        "root_displacement_w_m": displacement.tolist(),
        "mean_world_velocity_x_mps": float(displacement[0] / max(times[-1] - times[0], 1.0e-6)),
        "mean_world_velocity_y_mps": float(displacement[1] / max(times[-1] - times[0], 1.0e-6)),
        "mean_root_lin_vel_b": lin_vel_b.mean(axis=0).tolist(),
        "mean_root_ang_vel_b_radps": ang_vel_b.mean(axis=0).tolist(),
        "mean_foot_contact_count": float(np.mean(np.sum(foot_contacts, axis=1))),
        "non_foot_contact_fraction": float(np.mean(non_foot_contacts)),
        "non_foot_contact_steps": np.flatnonzero(non_foot_contacts).astype(int).tolist(),
        "done_steps": np.flatnonzero(dones).astype(int).tolist(),
    }
    summary.update(stats("roll", rpy_deg[:, 0]))
    summary.update(stats("pitch", rpy_deg[:, 1]))
    summary.update(stats("yaw_drift", rpy_deg[:, 2]))
    for foot_index, foot_name in enumerate(FOOT_NAMES):
        contact = foot_contacts[:, foot_index]
        current_swing = ~contact
        lift_active = lift_pulses[:, foot_index] > 0.05
        stance_speeds = foot_xy_speed[:, foot_index][contact]
        swing_heights = foot_heights[:, foot_index][current_swing]
        lifted_heights = foot_heights[:, foot_index][lift_active]
        prefix = foot_name.lower()
        summary[f"{prefix}_contact_fraction"] = float(np.mean(contact))
        summary[f"{prefix}_stance_sample_count"] = int(np.count_nonzero(contact))
        summary[f"{prefix}_swing_sample_count"] = int(np.count_nonzero(current_swing))
        summary[f"{prefix}_scheduled_lift_fraction"] = float(np.mean(lift_active))
        summary[f"{prefix}_scheduled_lift_sample_count"] = int(np.count_nonzero(lift_active))
        if stance_speeds.size:
            summary[f"{prefix}_mean_stance_xy_speed_mps"] = float(np.mean(stance_speeds))
            summary[f"{prefix}_median_stance_xy_speed_mps"] = float(np.median(stance_speeds))
            summary[f"{prefix}_p95_stance_xy_speed_mps"] = float(np.percentile(stance_speeds, 95))
            summary[f"{prefix}_stance_sliding_fraction_gt_0p05"] = float(np.mean(stance_speeds > 0.05))
            summary[f"{prefix}_stance_sliding_fraction_gt_0p10"] = float(np.mean(stance_speeds > 0.10))
        else:
            summary[f"{prefix}_mean_stance_xy_speed_mps"] = 0.0
            summary[f"{prefix}_median_stance_xy_speed_mps"] = 0.0
            summary[f"{prefix}_p95_stance_xy_speed_mps"] = 0.0
            summary[f"{prefix}_stance_sliding_fraction_gt_0p05"] = 0.0
            summary[f"{prefix}_stance_sliding_fraction_gt_0p10"] = 0.0
        if swing_heights.size:
            summary[f"{prefix}_mean_current_swing_height_m"] = float(np.mean(swing_heights))
            summary[f"{prefix}_max_current_swing_height_m"] = float(np.max(swing_heights))
        else:
            summary[f"{prefix}_mean_current_swing_height_m"] = 0.0
            summary[f"{prefix}_max_current_swing_height_m"] = 0.0
        if lifted_heights.size:
            summary[f"{prefix}_mean_scheduled_lift_height_m"] = float(np.mean(lifted_heights))
            summary[f"{prefix}_max_scheduled_lift_height_m"] = float(np.max(lifted_heights))
            summary[f"{prefix}_low_scheduled_lift_fraction_lt_0p03"] = float(np.mean(lifted_heights < 0.03))
        else:
            summary[f"{prefix}_mean_scheduled_lift_height_m"] = 0.0
            summary[f"{prefix}_max_scheduled_lift_height_m"] = 0.0
            summary[f"{prefix}_low_scheduled_lift_fraction_lt_0p03"] = 0.0
    summary["swing_lift_mean_abs_action_delta"] = float(np.mean(np.abs(lift_action_offsets)))
    summary["swing_lift_max_abs_action_delta"] = float(np.max(np.abs(lift_action_offsets)))
    return summary


def select_window(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    time_s = data["time_s"].astype(np.float32)
    mask = (time_s >= args.start_s) & (time_s < args.end_s)
    if not np.any(mask):
        raise ValueError(f"No trajectory samples found in window [{args.start_s}, {args.end_s}).")

    indices = np.flatnonzero(mask)
    if args.reverse:
        indices = indices[::-1]

    selected: dict[str, np.ndarray] = {
        "time_s": time_s[indices],
        "joint_pos": data[SOURCE_KEYS[args.source]][indices].astype(np.float32),
        "reset_joint_pos": data["actual_joint_pos"][indices].astype(np.float32),
        "actual_joint_vel": data["actual_joint_vel"][indices].astype(np.float32),
        "root_lin_vel_b": data["root_lin_vel_b"][indices].astype(np.float32),
        "root_ang_vel_b": data["root_ang_vel_b"][indices].astype(np.float32),
    }
    if "foot_contacts" in data.files:
        selected["foot_contacts"] = data["foot_contacts"][indices].astype(bool)
    if args.reverse:
        selected["actual_joint_vel"] *= -1.0
        selected["root_lin_vel_b"] *= -1.0
        selected["root_ang_vel_b"] *= -1.0
    if args.source == "action":
        selected["actions"] = selected["joint_pos"]
    return selected


def reset_to_start(raw, default_joint_pos: torch.Tensor, start_joint_pos: np.ndarray, start_joint_vel: np.ndarray) -> None:
    joint_pos = torch.as_tensor(start_joint_pos, dtype=torch.float32, device=raw.device).unsqueeze(0)
    joint_limits = raw._robot.data.soft_joint_pos_limits
    joint_pos = torch.clamp(joint_pos, joint_limits[:, :, 0], joint_limits[:, :, 1])
    if args.init_joint_velocity:
        joint_vel = torch.as_tensor(start_joint_vel, dtype=torch.float32, device=raw.device).unsqueeze(0)
    else:
        joint_vel = torch.zeros_like(default_joint_pos)

    root_state = raw._robot.data.default_root_state.clone()
    root_state[:, :3] += raw._terrain.env_origins
    root_state[:, 2] += args.root_z_offset
    if args.front_up_deg != 0.0:
        root_state[:, 3:7] = front_up_quat(raw.device, root_state.dtype).unsqueeze(0)
    root_state[:, 7:] = 0.0
    raw._robot.write_root_pose_to_sim(root_state[:, :7])
    raw._robot.write_root_velocity_to_sim(root_state[:, 7:])
    raw._robot.write_joint_state_to_sim(joint_pos, joint_vel)
    raw._actions.zero_()
    raw._previous_actions.zero_()
    raw._processed_actions[:] = joint_pos


def apply_body_xy_velocity_servo(raw, target_x: float | None, target_y: float | None, alpha_x: float, alpha_y: float) -> None:
    alpha_x = max(0.0, min(1.0, alpha_x))
    alpha_y = max(0.0, min(1.0, alpha_y))
    if (target_x is None or alpha_x <= 0.0) and (target_y is None or alpha_y <= 0.0):
        return
    root_velocity_w = raw._robot.data.root_state_w[:, 7:].clone()
    body_linear_velocity = quat_wxyz_rotate_inverse(
        raw._robot.data.root_link_quat_w,
        root_velocity_w[:, :3],
    )
    if target_x is not None and alpha_x > 0.0:
        body_linear_velocity[:, 0] = (1.0 - alpha_x) * body_linear_velocity[:, 0] + alpha_x * target_x
    if target_y is not None and alpha_y > 0.0:
        body_linear_velocity[:, 1] = (1.0 - alpha_y) * body_linear_velocity[:, 1] + alpha_y * target_y
    root_velocity_w[:, :3] = quat_wxyz_rotate(
        raw._robot.data.root_link_quat_w,
        body_linear_velocity,
    )
    raw._robot.write_root_velocity_to_sim(root_velocity_w)


def apply_body_y_velocity_servo(raw, target_y: float, alpha: float) -> None:
    alpha = max(0.0, min(1.0, alpha))
    if alpha <= 0.0:
        return
    root_velocity_w = raw._robot.data.root_state_w[:, 7:].clone()
    body_linear_velocity = quat_wxyz_rotate_inverse(
        raw._robot.data.root_link_quat_w,
        root_velocity_w[:, :3],
    )
    body_linear_velocity[:, 1] = (1.0 - alpha) * body_linear_velocity[:, 1] + alpha * target_y
    root_velocity_w[:, :3] = quat_wxyz_rotate(
        raw._robot.data.root_link_quat_w,
        body_linear_velocity,
    )
    raw._robot.write_root_velocity_to_sim(root_velocity_w)


def main() -> None:
    data = np.load(args.trajectory_npz, allow_pickle=False)
    selected = select_window(data)
    if args.source == "action" and args.sideways_remap_scale != 0.0:
        raise ValueError("Cannot apply --sideways-remap-scale with --source action because remap needs joint positions.")
    uses_lift_pulses = (
        args.swing_lift_height_m != 0.0
        or args.direct_lift_femur_action != 0.0
        or args.direct_lift_tibia_action != 0.0
        or args.root_body_vel_y_servo_lift_relief != 0.0
    )
    if uses_lift_pulses and "foot_contacts" not in selected:
        raise ValueError("Cannot apply lift pulse controls because foot_contacts is missing from trajectory.")
    default_joint_pos_np = data["default_joint_pos"].astype(np.float32)
    action_scale = float(data["action_scale"])
    joint_names = [str(name) for name in data["joint_names"]] if "joint_names" in data.files else []
    bl_tibia_action_bias = 0.0
    bl_tibia_action_index = None
    if abs(args.bl_tibia_bias_deg) > 1.0e-6:
        if "BL_tibia_joint" not in joint_names:
            raise ValueError("Cannot apply --bl-tibia-bias-deg because BL_tibia_joint is missing from joint_names.")
        bl_tibia_action_index = joint_names.index("BL_tibia_joint")
        bl_tibia_action_bias = math.radians(args.bl_tibia_bias_deg) / action_scale

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.viewer.eye = tuple(args.camera_eye)
    env_cfg.viewer.lookat = tuple(args.camera_lookat)
    env_cfg.viewer.origin_type = "world"
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    raw = env.unwrapped
    raw.episode_length_buf.zero_()
    initial_yaw = quat_wxyz_to_yaw(raw._robot.data.root_link_quat_w)[0].detach().clone()

    default_joint_pos = torch.as_tensor(default_joint_pos_np, dtype=torch.float32, device=raw.device).unsqueeze(0)
    start_joint_pos = default_joint_pos_np if args.start_from_default else selected["reset_joint_pos"][0]
    start_joint_vel = np.zeros_like(default_joint_pos_np) if args.start_from_default else selected["actual_joint_vel"][0]
    reset_to_start(raw, default_joint_pos, start_joint_pos, start_joint_vel)
    if args.init_reversed_root_velocity:
        root_velocity_w = raw._robot.data.root_state_w[:, 7:].clone()
        root_velocity_w[:, :3] = torch.as_tensor(selected["root_lin_vel_b"][0], dtype=torch.float32, device=raw.device)
        root_velocity_w[:, 3:] = torch.as_tensor(selected["root_ang_vel_b"][0], dtype=torch.float32, device=raw.device)
        raw._robot.write_root_velocity_to_sim(root_velocity_w)

    frames = []
    times = []
    root_positions = []
    quats = []
    lin_vel = []
    ang_vel = []
    non_foot_contacts = []
    foot_contacts_log = []
    foot_xy_speeds = []
    foot_heights = []
    lift_pulse_log = []
    lift_action_offset_log = []
    dones = []

    total_steps = max(1, int(args.duration / raw.step_dt))
    capture_every = max(1, round(1.0 / (args.fps * raw.step_dt)))
    sequence_len = selected["joint_pos"].shape[0]
    sequence_joint_pos = torch.as_tensor(selected["joint_pos"], dtype=torch.float32, device=raw.device)
    sequence_sideways_remap_offsets = compute_sideways_remap_action_offsets(sequence_joint_pos, action_scale)
    sequence_lift_pulses_np = np.zeros((sequence_len, len(FOOT_NAMES)), dtype=np.float32)
    if uses_lift_pulses:
        sequence_lift_pulses_np = compute_cyclic_swing_lift_pulses(
            selected["foot_contacts"],
            max(0, args.swing_lift_dilate_steps),
        )
    action_tensor = torch.zeros(raw.num_envs, raw.single_action_space.shape[0], device=raw.device)

    for step in range(total_steps):
        if args.cycle:
            sequence_index = int(step * args.playback_speed) % sequence_len
        else:
            sequence_index = min(int(step * args.playback_speed), sequence_len - 1)

        if args.source == "action":
            action = selected["actions"][sequence_index]
        else:
            action = (selected["joint_pos"][sequence_index] - default_joint_pos_np) / action_scale
        action = action.copy()
        if args.sideways_remap_scale != 0.0:
            action += sequence_sideways_remap_offsets[sequence_index].detach().cpu().numpy()
        action[0:4] *= args.coxa_action_scale
        action[4:8] *= args.femur_action_scale
        action[8:12] *= args.tibia_action_scale
        if bl_tibia_action_index is not None:
            action[bl_tibia_action_index] += bl_tibia_action_bias
        lift_pulse_np = sequence_lift_pulses_np[sequence_index]
        lift_offset = torch.zeros(raw.num_envs, raw.single_action_space.shape[0], device=raw.device)
        lift_pulse = torch.as_tensor(lift_pulse_np, dtype=torch.float32, device=raw.device).unsqueeze(0)
        if args.swing_lift_height_m != 0.0:
            lift_offset = compute_swing_lift_action_offsets(
                raw._robot.data.joint_pos,
                lift_pulse,
                action_scale,
            )
        if args.direct_lift_femur_action != 0.0 or args.direct_lift_tibia_action != 0.0:
            lift_offset += compute_direct_lift_action_offsets(raw._robot.data.joint_pos, lift_pulse)
        if torch.max(torch.abs(lift_offset)).item() > 0.0:
            action += lift_offset[0].detach().cpu().numpy()
        if args.yaw_feedback_kp != 0.0 or args.yaw_rate_feedback_kd != 0.0:
            yaw = quat_wxyz_to_yaw(raw._robot.data.root_link_quat_w)[0]
            yaw_error = torch.atan2(torch.sin(yaw - initial_yaw), torch.cos(yaw - initial_yaw))
            yaw_rate = raw._robot.data.root_ang_vel_b[0, 2]
            feedback = args.yaw_feedback_sign * (
                -args.yaw_feedback_kp * yaw_error - args.yaw_rate_feedback_kd * yaw_rate
            )
            feedback = torch.clamp(feedback, -args.max_yaw_feedback, args.max_yaw_feedback).item()
            action[list(COXA_LEFT)] += feedback
            action[list(COXA_RIGHT)] -= feedback
        action = np.clip(action, -1.0, 1.0)
        action_tensor[0] = torch.as_tensor(action, dtype=torch.float32, device=raw.device)

        with torch.inference_mode():
            env.step(action_tensor)
            servo_target_x = None
            servo_target_y = None
            step_time = step * raw.step_dt
            if (
                args.root_body_vel_x_servo_target is not None
                and args.root_body_vel_x_servo_alpha > 0.0
                and step_time >= args.root_body_vel_x_servo_start_s
            ):
                servo_target_x = args.root_body_vel_x_servo_target
            if (
                args.root_body_vel_y_servo_target is not None
                and args.root_body_vel_y_servo_alpha > 0.0
                and step_time >= args.root_body_vel_y_servo_start_s
            ):
                servo_target_y = args.root_body_vel_y_servo_target
                if args.root_body_vel_y_servo_lift_relief != 0.0:
                    lift_relief = max(0.0, min(1.0, args.root_body_vel_y_servo_lift_relief))
                    servo_target_y *= 1.0 - lift_relief * float(np.max(lift_pulse_np))
            if servo_target_x is not None or servo_target_y is not None:
                apply_body_xy_velocity_servo(
                    raw,
                    servo_target_x,
                    servo_target_y,
                    args.root_body_vel_x_servo_alpha,
                    args.root_body_vel_y_servo_alpha,
                )
            if args.root_yaw_rate_damping > 0.0:
                damping = max(0.0, min(1.0, args.root_yaw_rate_damping))
                root_velocity_w = raw._robot.data.root_state_w[:, 7:].clone()
                root_velocity_w[:, 5] *= 1.0 - damping
                raw._robot.write_root_velocity_to_sim(root_velocity_w)
            if args.root_yaw_servo_kp != 0.0 or args.root_yaw_servo_kd != 0.0:
                yaw = quat_wxyz_to_yaw(raw._robot.data.root_link_quat_w)
                yaw_error = torch.atan2(torch.sin(yaw - initial_yaw), torch.cos(yaw - initial_yaw))
                root_velocity_w = raw._robot.data.root_state_w[:, 7:].clone()
                correction = args.root_yaw_servo_sign * (
                    -args.root_yaw_servo_kp * yaw_error - args.root_yaw_servo_kd * root_velocity_w[:, 5]
                )
                correction = torch.clamp(
                    correction,
                    -args.root_yaw_servo_max_angvel,
                    args.root_yaw_servo_max_angvel,
                )
                root_velocity_w[:, 5] = correction
                raw._robot.write_root_velocity_to_sim(root_velocity_w)
            if args.front_up_kp != 0.0 or args.front_up_kd != 0.0:
                target_roll = math.radians(args.front_up_deg)
                current_roll = quat_wxyz_to_roll(raw._robot.data.root_link_quat_w)
                roll_error = current_roll - target_roll
                root_velocity_w = raw._robot.data.root_state_w[:, 7:].clone()
                correction = -args.front_up_kp * roll_error - args.front_up_kd * root_velocity_w[:, 3]
                correction = torch.clamp(
                    correction,
                    -args.front_up_max_angvel,
                    args.front_up_max_angvel,
                )
                root_velocity_w[:, 3] = correction
                raw._robot.write_root_velocity_to_sim(root_velocity_w)

        forces = raw._contact_sensor.data.net_forces_w_history
        foot_contacts_now = raw._get_foot_contacts()
        foot_pos_w = raw._robot.data.body_pos_w[:, raw._feet_body_ids, :]
        foot_vel_w = raw._robot.data.body_lin_vel_w[:, raw._feet_body_ids, :]
        foot_xy_speed = torch.norm(foot_vel_w[:, :, :2], dim=2)
        foot_height = torch.clamp(
            foot_pos_w[:, :, 2] - raw._terrain.env_origins[:, 2].unsqueeze(1),
            min=0.0,
        )
        non_foot_contact = torch.amax(
            torch.norm(forces[:, :, raw._undesired_contact_body_ids], dim=-1),
            dim=(1, 2),
        ) > 1.0
        times.append(step * raw.step_dt)
        root_positions.append(raw._robot.data.root_link_pos_w[0].detach().cpu().numpy().copy())
        quats.append(raw._robot.data.root_link_quat_w[0].detach().cpu().numpy().copy())
        lin_vel.append(raw._robot.data.root_lin_vel_b[0].detach().cpu().numpy().copy())
        ang_vel.append(raw._robot.data.root_ang_vel_b[0].detach().cpu().numpy().copy())
        foot_contacts_log.append(foot_contacts_now[0].detach().cpu().numpy().copy())
        foot_xy_speeds.append(foot_xy_speed[0].detach().cpu().numpy().copy())
        foot_heights.append(foot_height[0].detach().cpu().numpy().copy())
        lift_pulse_log.append(lift_pulse_np.copy())
        lift_action_offset_log.append(lift_offset[0].detach().cpu().numpy().copy())
        non_foot_contacts.append(bool(non_foot_contact[0].detach().cpu().item()))
        done = bool((raw.reset_terminated[0] | raw.reset_time_outs[0]).item())
        dones.append(done)
        if done:
            raw.episode_length_buf.zero_()

        if not args.no_video and (step % capture_every == 0 or step == total_steps - 1):
            frame = env.render()
            if frame is not None and frame.size:
                frames.append(frame)

    summary = summarize(
        np.asarray(times),
        np.asarray(root_positions),
        np.asarray(quats),
        np.asarray(lin_vel),
        np.asarray(ang_vel),
        np.asarray(dones, dtype=bool),
        np.asarray(non_foot_contacts, dtype=bool),
        np.asarray(foot_contacts_log, dtype=bool),
        np.asarray(foot_xy_speeds),
        np.asarray(foot_heights),
        np.asarray(lift_pulse_log),
        np.asarray(lift_action_offset_log),
    )
    summary.update(
        {
            "trajectory_npz": str(args.trajectory_npz),
            "source": args.source,
            "reverse": args.reverse,
            "cycle": args.cycle,
            "playback_speed": args.playback_speed,
            "start_s": args.start_s,
            "end_s": args.end_s,
            "sequence_samples": int(sequence_len),
            "init_reversed_root_velocity": args.init_reversed_root_velocity,
            "init_joint_velocity": args.init_joint_velocity,
            "root_yaw_rate_damping": args.root_yaw_rate_damping,
            "root_yaw_servo_kp": args.root_yaw_servo_kp,
            "root_yaw_servo_kd": args.root_yaw_servo_kd,
            "root_yaw_servo_sign": args.root_yaw_servo_sign,
            "root_yaw_servo_max_angvel": args.root_yaw_servo_max_angvel,
            "yaw_feedback_kp": args.yaw_feedback_kp,
            "yaw_rate_feedback_kd": args.yaw_rate_feedback_kd,
            "yaw_feedback_sign": args.yaw_feedback_sign,
            "max_yaw_feedback": args.max_yaw_feedback,
            "front_up_deg": args.front_up_deg,
            "front_up_kp": args.front_up_kp,
            "front_up_kd": args.front_up_kd,
            "front_up_max_angvel": args.front_up_max_angvel,
            "bl_tibia_bias_deg": args.bl_tibia_bias_deg,
            "coxa_action_scale": args.coxa_action_scale,
            "femur_action_scale": args.femur_action_scale,
            "tibia_action_scale": args.tibia_action_scale,
            "root_body_vel_x_servo_target": args.root_body_vel_x_servo_target,
            "root_body_vel_x_servo_alpha": args.root_body_vel_x_servo_alpha,
            "root_body_vel_x_servo_start_s": args.root_body_vel_x_servo_start_s,
            "root_body_vel_y_servo_target": args.root_body_vel_y_servo_target,
            "root_body_vel_y_servo_alpha": args.root_body_vel_y_servo_alpha,
            "root_body_vel_y_servo_start_s": args.root_body_vel_y_servo_start_s,
            "root_body_vel_y_servo_lift_relief": args.root_body_vel_y_servo_lift_relief,
            "sideways_remap_scale": args.sideways_remap_scale,
            "sideways_remap_sign": args.sideways_remap_sign,
            "sideways_remap_x_keep_scale": args.sideways_remap_x_keep_scale,
            "sideways_remap_y_keep_scale": args.sideways_remap_y_keep_scale,
            "sideways_remap_joints": args.sideways_remap_joints,
            "sideways_remap_damping": args.sideways_remap_damping,
            "sideways_remap_max_action_delta": args.sideways_remap_max_action_delta,
            "swing_lift_height_m": args.swing_lift_height_m,
            "swing_lift_joints": args.swing_lift_joints,
            "swing_lift_rear_scale": args.swing_lift_rear_scale,
            "swing_lift_middle_scale": args.swing_lift_middle_scale,
            "swing_lift_damping": args.swing_lift_damping,
            "swing_lift_max_action_delta": args.swing_lift_max_action_delta,
            "swing_lift_action_gain": args.swing_lift_action_gain,
            "swing_lift_dilate_steps": args.swing_lift_dilate_steps,
            "direct_lift_femur_action": args.direct_lift_femur_action,
            "direct_lift_tibia_action": args.direct_lift_tibia_action,
            "direct_lift_rear_scale": args.direct_lift_rear_scale,
            "direct_lift_middle_scale": args.direct_lift_middle_scale,
            "direct_lift_max_action_delta": args.direct_lift_max_action_delta,
        }
    )
    if args.summary_output is None:
        args.summary_output = args.output.with_suffix(".summary.json")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    if not args.no_video:
        if not frames:
            raise RuntimeError("No frames captured.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.output, frames, fps=args.fps)
        print(f"VIDEO={args.output}", flush=True)
    print(f"SUMMARY={args.summary_output}", flush=True)
    print(f"ROLL_P2P_DEG={summary['roll_peak_to_peak_deg']:.3f}", flush=True)
    print(f"PITCH_P2P_DEG={summary['pitch_peak_to_peak_deg']:.3f}", flush=True)
    print(f"YAW_DRIFT_P2P_DEG={summary['yaw_drift_peak_to_peak_deg']:.3f}", flush=True)
    print(f"MEAN_VEL_X={summary['mean_root_lin_vel_b'][0]:.3f}", flush=True)
    print(f"MEAN_VEL_Y={summary['mean_root_lin_vel_b'][1]:.3f}", flush=True)
    print(f"WORLD_DISP_X={summary['root_displacement_w_m'][0]:.3f}", flush=True)
    print(f"WORLD_DISP_Y={summary['root_displacement_w_m'][1]:.3f}", flush=True)
    print(f"MEAN_FOOT_CONTACT_COUNT={summary['mean_foot_contact_count']:.3f}", flush=True)
    print(f"BL_MEAN_STANCE_XY_SPEED_MPS={summary['bl_foot_mean_stance_xy_speed_mps']:.4f}", flush=True)
    print(f"BL_MAX_SCHEDULED_LIFT_HEIGHT_M={summary['bl_foot_max_scheduled_lift_height_m']:.4f}", flush=True)
    print(f"SWING_LIFT_MAX_ABS_ACTION_DELTA={summary['swing_lift_max_abs_action_delta']:.4f}", flush=True)
    print(f"NON_FOOT_CONTACT_FRACTION={summary['non_foot_contact_fraction']:.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
