#!/usr/bin/env python3
"""Render a policy rollout with light action smoothing and yaw stabilization."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
COXA_LEFT = (0, 2)  # BL, ML in joint order
COXA_RIGHT = (1, 3)  # BR, MR in joint order
BL_FOOT_INDEX = 0
BL_FEMUR_INDEX = 4
BL_TIBIA_INDEX = 8
BL_REAR_LEG_INDICES = (0, 4, 8)
BR_REAR_LEG_INDICES = (1, 5, 9)
BL_REAR_DISTAL_INDICES = (4, 8)
BR_REAR_DISTAL_INDICES = (5, 9)
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
MIDDLE_FOOT_INDICES = (2, 3)
LOCAL_JOINT_SELECTIONS = {
    "coxa": (0,),
    "coxa_tibia": (0, 2),
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
parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "trained_models/forward_strict_best/model_1399.pt")
parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/renders/model_1399_stabilized_forward.mp4")
parser.add_argument("--attitude-output", type=Path, default=None)
parser.add_argument("--telemetry-output", type=Path, default=None)
parser.add_argument("--trajectory-output", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--forward-vel", type=float, default=0.30)
parser.add_argument("--lateral-vel", type=float, default=0.0)
parser.add_argument("--yaw-rate", type=float, default=0.0)
parser.add_argument("--video-length", type=int, default=300)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--action-filter-alpha", type=float, default=0.65)
parser.add_argument("--yaw-feedback-kp", type=float, default=0.22)
parser.add_argument("--yaw-rate-feedback-kd", type=float, default=0.40)
parser.add_argument("--yaw-feedback-sign", type=float, default=1.0)
parser.add_argument("--max-yaw-feedback", type=float, default=0.18)
parser.add_argument("--root-yaw-rate-damping", type=float, default=0.0)
parser.add_argument("--bl-lift-boost", type=float, default=0.0)
parser.add_argument("--bl-tibia-boost", type=float, default=0.0)
parser.add_argument("--bl-lift-duration", type=float, default=0.14)
parser.add_argument("--mirror-bl-from-br", action="store_true")
parser.add_argument("--mirror-bl-joints", choices=("all", "distal"), default="all")
parser.add_argument("--mirror-bl-blend", type=float, default=1.0)
parser.add_argument("--mirror-bl-delay-phase", type=float, default=0.34227248404234994)
parser.add_argument("--mirror-cycle-steps", type=float, default=17.77777777777778)
parser.add_argument("--stride-ik-rear-forward-m", type=float, default=0.0)
parser.add_argument("--stride-ik-middle-forward-m", type=float, default=0.0)
parser.add_argument("--stride-ik-rear-joints", choices=tuple(LOCAL_JOINT_SELECTIONS), default="coxa")
parser.add_argument("--stride-ik-middle-joints", choices=tuple(LOCAL_JOINT_SELECTIONS), default="coxa_tibia")
parser.add_argument("--stride-ik-profile", choices=("swing_reach", "stroke"), default="swing_reach")
parser.add_argument("--stride-ik-duration", type=float, default=0.16)
parser.add_argument("--stride-ik-stance-duration", type=float, default=0.18)
parser.add_argument("--stride-ik-damping", type=float, default=0.01)
parser.add_argument("--stride-ik-max-action-delta", type=float, default=0.20)
parser.add_argument("--recycle-actions", action="store_true")
parser.add_argument("--recycle-reverse", action="store_true")
parser.add_argument("--recycle-coxa-scale", type=float, default=1.0)
parser.add_argument("--recycle-femur-scale", type=float, default=1.0)
parser.add_argument("--recycle-tibia-scale", type=float, default=1.0)
parser.add_argument("--recycle-reverse-init-root-vel-y", type=float, default=None)
parser.add_argument("--recycle-reverse-joint-vel-scale", type=float, default=0.0)
parser.add_argument("--recycle-reverse-zero-angular-velocity", action="store_true")
parser.add_argument("--recycle-start-s", type=float, default=0.54)
parser.add_argument("--recycle-end-s", type=float, default=1.12)
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--camera-eye", type=float, nargs=3, default=(3.6, -3.6, 3.9))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(0.0, 0.0, 0.12))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_video

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import insectoid_mini_quad_rl  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def quat_wxyz_to_yaw(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_wxyz_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def quat_wxyz_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_wxyz_to_rpy_np(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=1)


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

    for joint_index, (origin_xyz, origin_rpy) in enumerate(zip(cfg["joint_origins"], cfg["joint_rpy"], strict=True)):
        origin = torch.tensor(origin_xyz, device=device, dtype=dtype).expand(num_envs, 3)
        origin_rot = rpy_to_matrix(origin_rpy, device, dtype).expand(num_envs, 3, 3)
        position = position + transform_points(rotation, origin)
        axis_frame_rotation = torch.matmul(rotation, origin_rot)
        axis_z = axis_frame_rotation[:, :, 2]
        joint_positions.append(position)
        joint_axes.append(axis_z)
        rotation = torch.matmul(axis_frame_rotation, z_rotation_matrix(joint_pos[:, joint_index]))

    foot_offset = torch.tensor(cfg["foot_offset"], device=device, dtype=dtype).expand(num_envs, 3)
    foot_position = position + transform_points(rotation, foot_offset)
    jacobian_columns = [
        torch.cross(axis, foot_position - joint_position, dim=-1)
        for axis, joint_position in zip(joint_axes, joint_positions, strict=True)
    ]
    return foot_position, torch.stack(jacobian_columns, dim=-1)


def compute_stride_ik_action_offsets(
    joint_pos: torch.Tensor,
    foot_airborne: torch.Tensor,
    foot_swing_time: torch.Tensor,
    foot_stance_time: torch.Tensor,
    action_scale: float,
) -> torch.Tensor:
    offsets = torch.zeros(joint_pos.shape[0], len(JOINT_NAMES), device=joint_pos.device, dtype=joint_pos.dtype)
    if args.stride_ik_rear_forward_m == 0.0 and args.stride_ik_middle_forward_m == 0.0:
        return offsets

    swing_duration = max(args.stride_ik_duration, 1.0e-6)
    stance_duration = max(args.stride_ik_stance_duration, 1.0e-6)
    damping_sq = args.stride_ik_damping * args.stride_ik_damping
    eye3 = torch.eye(3, device=joint_pos.device, dtype=joint_pos.dtype).expand(joint_pos.shape[0], 3, 3)
    forward_sign = 1.0 if args.forward_vel >= 0.0 else -1.0

    for foot_index, foot_name in enumerate(FOOT_NAMES):
        if foot_index in REAR_FOOT_INDICES:
            forward_amplitude = args.stride_ik_rear_forward_m
            selection_name = args.stride_ik_rear_joints
        else:
            forward_amplitude = args.stride_ik_middle_forward_m
            selection_name = args.stride_ik_middle_joints
        if forward_amplitude == 0.0:
            continue

        leg_joint_indices = LEG_JOINT_INDICES[foot_name]
        local_selection = LOCAL_JOINT_SELECTIONS[selection_name]
        q_leg = joint_pos[:, list(leg_joint_indices)]
        _, jacobian = leg_foot_position_and_jacobian(foot_name, q_leg)
        jacobian_selected = jacobian[:, :, list(local_selection)]
        contact = ~foot_airborne[:, foot_index]
        swing_phase = torch.clamp(foot_swing_time[:, foot_index] / swing_duration, 0.0, 1.0)
        swing_ramp = swing_phase * swing_phase * (3.0 - 2.0 * swing_phase)
        airborne_f = foot_airborne[:, foot_index].to(joint_pos.dtype)
        contact_f = contact.to(joint_pos.dtype)
        if args.stride_ik_profile == "stroke":
            stance_phase = torch.clamp(foot_stance_time[:, foot_index] / stance_duration, 0.0, 1.0)
            stance_ramp = stance_phase * stance_phase * (3.0 - 2.0 * stance_phase)
            stroke_shape = (2.0 * swing_ramp - 1.0) * airborne_f + (1.0 - 2.0 * stance_ramp) * contact_f
            pulse = stroke_shape
        else:
            pulse = swing_ramp * airborne_f
        delta_foot = torch.zeros(joint_pos.shape[0], 3, device=joint_pos.device, dtype=joint_pos.dtype)
        delta_foot[:, 1] = forward_sign * forward_amplitude * pulse

        lhs = torch.matmul(jacobian_selected, jacobian_selected.transpose(1, 2)) + damping_sq * eye3
        solved = torch.linalg.solve(lhs, delta_foot[:, :, None]).squeeze(-1)
        delta_q_selected = torch.matmul(jacobian_selected.transpose(1, 2), solved[:, :, None]).squeeze(-1)
        delta_action = torch.clamp(
            delta_q_selected / action_scale,
            -args.stride_ik_max_action_delta,
            args.stride_ik_max_action_delta,
        )
        for local_column, global_joint_index in enumerate(leg_joint_indices[index] for index in local_selection):
            offsets[:, global_joint_index] += delta_action[:, local_column]
    return offsets


def summarize_attitude(
    times: np.ndarray,
    quats: np.ndarray,
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    dones: np.ndarray,
    foot_heights: np.ndarray,
    foot_body_pos: np.ndarray,
    foot_xy_speed: np.ndarray,
    foot_contacts: np.ndarray,
    non_foot_contacts: np.ndarray,
    bl_lift_offsets: np.ndarray,
    bl_tibia_offsets: np.ndarray,
    stride_action_offsets: np.ndarray,
):
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

    summary = {
        "duration_s": float(times[-1] - times[0]) if len(times) else 0.0,
        "done_steps": np.flatnonzero(dones).astype(int).tolist(),
        "mean_root_lin_vel_b": lin_vel.mean(axis=0).tolist(),
        "mean_root_ang_vel_b_radps": ang_vel.mean(axis=0).tolist(),
        "rms_root_ang_vel_b_radps": np.sqrt(np.mean(np.square(ang_vel), axis=0)).tolist(),
        "mean_foot_contact_count": float(np.mean(np.sum(foot_contacts, axis=1))),
        "non_foot_contact_fraction": float(np.mean(non_foot_contacts)),
        "non_foot_contact_steps": np.flatnonzero(non_foot_contacts).astype(int).tolist(),
        "action_filter_alpha": args.action_filter_alpha,
        "yaw_feedback_kp": args.yaw_feedback_kp,
        "yaw_rate_feedback_kd": args.yaw_rate_feedback_kd,
        "yaw_feedback_sign": args.yaw_feedback_sign,
        "max_yaw_feedback": args.max_yaw_feedback,
        "root_yaw_rate_damping": args.root_yaw_rate_damping,
        "bl_lift_boost": args.bl_lift_boost,
        "bl_tibia_boost": args.bl_tibia_boost,
        "bl_lift_duration": args.bl_lift_duration,
        "mirror_bl_from_br": args.mirror_bl_from_br,
        "mirror_bl_joints": args.mirror_bl_joints,
        "mirror_bl_blend": args.mirror_bl_blend,
        "mirror_bl_delay_phase": args.mirror_bl_delay_phase,
        "mirror_cycle_steps": args.mirror_cycle_steps,
        "stride_ik_rear_forward_m": args.stride_ik_rear_forward_m,
        "stride_ik_middle_forward_m": args.stride_ik_middle_forward_m,
        "stride_ik_rear_joints": args.stride_ik_rear_joints,
        "stride_ik_middle_joints": args.stride_ik_middle_joints,
        "stride_ik_profile": args.stride_ik_profile,
        "stride_ik_duration": args.stride_ik_duration,
        "stride_ik_stance_duration": args.stride_ik_stance_duration,
        "stride_ik_damping": args.stride_ik_damping,
        "stride_ik_max_action_delta": args.stride_ik_max_action_delta,
        "recycle_actions": args.recycle_actions,
        "recycle_reverse": args.recycle_reverse,
        "recycle_coxa_scale": args.recycle_coxa_scale,
        "recycle_femur_scale": args.recycle_femur_scale,
        "recycle_tibia_scale": args.recycle_tibia_scale,
        "recycle_reverse_init_root_vel_y": args.recycle_reverse_init_root_vel_y,
        "recycle_reverse_joint_vel_scale": args.recycle_reverse_joint_vel_scale,
        "recycle_reverse_zero_angular_velocity": args.recycle_reverse_zero_angular_velocity,
        "recycle_start_s": args.recycle_start_s,
        "recycle_end_s": args.recycle_end_s,
    }
    summary.update(stats("roll", rpy_deg[:, 0]))
    summary.update(stats("pitch", rpy_deg[:, 1]))
    summary.update(stats("yaw_drift", rpy_deg[:, 2]))

    for foot_index, foot_name in enumerate(FOOT_NAMES):
        contact = foot_contacts[:, foot_index]
        swing = ~contact
        stance_speeds = foot_xy_speed[:, foot_index][contact]
        swing_heights = foot_heights[:, foot_index][swing]
        swing_body_pos = foot_body_pos[:, foot_index][swing]
        prefix = foot_name.lower()
        summary[f"{prefix}_contact_fraction"] = float(np.mean(contact))
        summary[f"{prefix}_stance_sample_count"] = int(np.count_nonzero(contact))
        summary[f"{prefix}_swing_sample_count"] = int(np.count_nonzero(swing))
        if stance_speeds.size:
            summary[f"{prefix}_mean_stance_xy_speed_mps"] = float(np.mean(stance_speeds))
            summary[f"{prefix}_median_stance_xy_speed_mps"] = float(np.median(stance_speeds))
            summary[f"{prefix}_p95_stance_xy_speed_mps"] = float(np.percentile(stance_speeds, 95))
            summary[f"{prefix}_max_stance_xy_speed_mps"] = float(np.max(stance_speeds))
            summary[f"{prefix}_stance_sliding_fraction_gt_0p05"] = float(np.mean(stance_speeds > 0.05))
            summary[f"{prefix}_stance_sliding_fraction_gt_0p10"] = float(np.mean(stance_speeds > 0.10))
        else:
            summary[f"{prefix}_mean_stance_xy_speed_mps"] = 0.0
            summary[f"{prefix}_median_stance_xy_speed_mps"] = 0.0
            summary[f"{prefix}_p95_stance_xy_speed_mps"] = 0.0
            summary[f"{prefix}_max_stance_xy_speed_mps"] = 0.0
            summary[f"{prefix}_stance_sliding_fraction_gt_0p05"] = 0.0
            summary[f"{prefix}_stance_sliding_fraction_gt_0p10"] = 0.0
        if swing_heights.size:
            summary[f"{prefix}_mean_swing_height_m"] = float(np.mean(swing_heights))
            summary[f"{prefix}_max_swing_height_m"] = float(np.max(swing_heights))
            summary[f"{prefix}_low_swing_fraction"] = float(np.mean(swing_heights < 0.025))
            swing_body_xy_radius = np.linalg.norm(swing_body_pos[:, :2], axis=1)
            summary[f"{prefix}_mean_swing_body_x_m"] = float(np.mean(swing_body_pos[:, 0]))
            summary[f"{prefix}_mean_swing_body_y_m"] = float(np.mean(swing_body_pos[:, 1]))
            summary[f"{prefix}_mean_swing_body_xy_radius_m"] = float(np.mean(swing_body_xy_radius))
            summary[f"{prefix}_min_swing_body_xy_radius_m"] = float(np.min(swing_body_xy_radius))
            summary[f"{prefix}_max_swing_body_xy_radius_m"] = float(np.max(swing_body_xy_radius))
        else:
            summary[f"{prefix}_mean_swing_height_m"] = 0.0
            summary[f"{prefix}_max_swing_height_m"] = 0.0
            summary[f"{prefix}_low_swing_fraction"] = 0.0
            summary[f"{prefix}_mean_swing_body_x_m"] = 0.0
            summary[f"{prefix}_mean_swing_body_y_m"] = 0.0
            summary[f"{prefix}_mean_swing_body_xy_radius_m"] = 0.0
            summary[f"{prefix}_min_swing_body_xy_radius_m"] = 0.0
            summary[f"{prefix}_max_swing_body_xy_radius_m"] = 0.0

    summary["bl_lift_offset_action_mean"] = float(np.mean(bl_lift_offsets))
    summary["bl_lift_offset_action_min"] = float(np.min(bl_lift_offsets))
    summary["bl_lift_offset_action_max"] = float(np.max(bl_lift_offsets))
    summary["bl_tibia_offset_action_mean"] = float(np.mean(bl_tibia_offsets))
    summary["bl_tibia_offset_action_min"] = float(np.min(bl_tibia_offsets))
    summary["bl_tibia_offset_action_max"] = float(np.max(bl_tibia_offsets))
    if stride_action_offsets.size:
        abs_stride_offsets = np.abs(stride_action_offsets)
        summary["stride_ik_mean_abs_action_delta"] = float(np.mean(abs_stride_offsets))
        summary["stride_ik_max_abs_action_delta"] = float(np.max(abs_stride_offsets))
        for joint_index, joint_name in enumerate(JOINT_NAMES):
            prefix = f"stride_ik_{joint_name}"
            summary[f"{prefix}_mean_abs_action_delta"] = float(np.mean(abs_stride_offsets[:, joint_index]))
            summary[f"{prefix}_max_abs_action_delta"] = float(np.max(abs_stride_offsets[:, joint_index]))
    return summary


def main() -> None:
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.lin_vel_x_range = (args.lateral_vel, args.lateral_vel)
    env_cfg.lin_vel_y_range = (args.forward_vel, args.forward_vel)
    env_cfg.ang_vel_z_range = (args.yaw_rate, args.yaw_rate)
    env_cfg.rel_standing_envs = 0.0
    if hasattr(env_cfg, "events"):
        env_cfg.events.physics_material = None
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = tuple(args.camera_eye)
        env_cfg.viewer.lookat = tuple(args.camera_lookat)
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.resolution = (args.width, args.height)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device

    raw_env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if not args.no_video else None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = raw_env.unwrapped
    raw.episode_length_buf.zero_()
    obs = env.get_observations()
    initial_yaw = quat_wxyz_to_yaw(raw._robot.data.root_link_quat_w)[0].detach().clone()
    filtered_actions = torch.zeros(raw.num_envs, raw.single_action_space.shape[0], device=raw.device)
    bl_swing_time = torch.zeros(raw.num_envs, device=raw.device)
    foot_swing_time = torch.zeros(raw.num_envs, len(FOOT_NAMES), device=raw.device)
    foot_stance_time = torch.zeros(raw.num_envs, len(FOOT_NAMES), device=raw.device)
    br_rear_action_history: list[torch.Tensor] = []
    recycled_base_action_history: list[torch.Tensor] = []
    reverse_recycle_initialized = False
    mirror_delay_steps = max(0.0, args.mirror_bl_delay_phase * args.mirror_cycle_steps)

    def delayed_br_rear_action(source: torch.Tensor) -> torch.Tensor:
        br_rear_action_history.append(source.detach().clone())
        max_history = int(math.ceil(mirror_delay_steps)) + 3
        if len(br_rear_action_history) > max_history:
            del br_rear_action_history[:-max_history]
        if mirror_delay_steps <= 0.0 or len(br_rear_action_history) == 1:
            return source

        lower = int(math.floor(mirror_delay_steps))
        upper = int(math.ceil(mirror_delay_steps))
        frac = mirror_delay_steps - lower

        def history_at_lag(lag: int) -> torch.Tensor:
            index = max(0, len(br_rear_action_history) - 1 - lag)
            return br_rear_action_history[index]

        newer = history_at_lag(lower)
        older = history_at_lag(upper)
        return (1.0 - frac) * newer + frac * older

    frames = []
    times = []
    quats = []
    root_positions = []
    policy_action_log = []
    stabilized_action_log = []
    processed_target_log = []
    joint_pos_log = []
    joint_vel_log = []
    lin_vel = []
    ang_vel = []
    foot_heights = []
    foot_body_positions = []
    foot_xy_speeds = []
    foot_contacts_log = []
    non_foot_contacts_log = []
    bl_lift_offsets = []
    bl_tibia_offsets = []
    stride_action_offsets = []
    yaw_feedback_offsets = []
    dones = []

    for step in range(args.video_length):
        with torch.inference_mode():
            time_s = step * raw.step_dt
            raw._commands[:, 0] = args.lateral_vel
            raw._commands[:, 1] = args.forward_vel
            raw._commands[:, 2] = args.yaw_rate
            policy_actions = policy(obs)
            filtered_actions = args.action_filter_alpha * policy_actions + (1.0 - args.action_filter_alpha) * filtered_actions
            stabilized_actions = filtered_actions.clone()

            if args.mirror_bl_from_br:
                if args.mirror_bl_joints == "distal":
                    br_indices = BR_REAR_DISTAL_INDICES
                    bl_indices = BL_REAR_DISTAL_INDICES
                else:
                    br_indices = BR_REAR_LEG_INDICES
                    bl_indices = BL_REAR_LEG_INDICES
                br_rear_action = filtered_actions[:, list(br_indices)]
                mirrored_bl_action = delayed_br_rear_action(br_rear_action)
                blend = max(0.0, min(1.0, args.mirror_bl_blend))
                stabilized_actions[:, list(bl_indices)] = (
                    (1.0 - blend) * stabilized_actions[:, list(bl_indices)]
                    + blend * mirrored_bl_action
                )

            foot_contacts_before = raw._get_foot_contacts()
            foot_airborne = ~foot_contacts_before
            foot_swing_time = torch.where(
                foot_airborne,
                foot_swing_time + raw.step_dt,
                torch.zeros_like(foot_swing_time),
            )
            foot_stance_time = torch.where(
                foot_contacts_before,
                foot_stance_time + raw.step_dt,
                torch.zeros_like(foot_stance_time),
            )
            stride_ik_offset = compute_stride_ik_action_offsets(
                raw._robot.data.joint_pos,
                foot_airborne,
                foot_swing_time,
                foot_stance_time,
                float(raw.cfg.action_scale),
            )
            stabilized_actions += stride_ik_offset

            if args.recycle_actions:
                if args.recycle_start_s <= time_s < args.recycle_end_s:
                    recycled_base_action_history.append(stabilized_actions.detach().clone())
                elif time_s >= args.recycle_end_s and recycled_base_action_history:
                    if (
                        args.recycle_reverse
                        and not reverse_recycle_initialized
                        and args.recycle_reverse_init_root_vel_y is not None
                    ):
                        root_velocity_w = raw._robot.data.root_state_w[:, 7:].clone()
                        desired_body_velocity = torch.zeros(raw.num_envs, 3, device=raw.device)
                        desired_body_velocity[:, 1] = args.recycle_reverse_init_root_vel_y
                        root_velocity_w[:, :3] = quat_wxyz_rotate(
                            raw._robot.data.root_link_quat_w,
                            desired_body_velocity,
                        )
                        if args.recycle_reverse_zero_angular_velocity:
                            root_velocity_w[:, 3:] = 0.0
                        raw._robot.write_root_velocity_to_sim(root_velocity_w)
                        if args.recycle_reverse_joint_vel_scale != 0.0:
                            raw._robot.write_joint_state_to_sim(
                                raw._robot.data.joint_pos,
                                -args.recycle_reverse_joint_vel_scale * raw._robot.data.joint_vel,
                            )
                        reverse_recycle_initialized = True
                    cycle_index = int((time_s - args.recycle_end_s) / raw.step_dt) % len(recycled_base_action_history)
                    if args.recycle_reverse:
                        cycle_index = len(recycled_base_action_history) - 1 - cycle_index
                    stabilized_actions = recycled_base_action_history[cycle_index].clone()
                    stabilized_actions[:, 0:4] *= args.recycle_coxa_scale
                    stabilized_actions[:, 4:8] *= args.recycle_femur_scale
                    stabilized_actions[:, 8:12] *= args.recycle_tibia_scale

            yaw = quat_wxyz_to_yaw(raw._robot.data.root_link_quat_w)[0]
            yaw_error = torch.atan2(torch.sin(yaw - initial_yaw), torch.cos(yaw - initial_yaw))
            yaw_rate = raw._robot.data.root_ang_vel_b[0, 2]
            feedback = args.yaw_feedback_sign * (-args.yaw_feedback_kp * yaw_error - args.yaw_rate_feedback_kd * yaw_rate)
            feedback = torch.clamp(feedback, -args.max_yaw_feedback, args.max_yaw_feedback)
            stabilized_actions[:, list(COXA_LEFT)] += feedback
            stabilized_actions[:, list(COXA_RIGHT)] -= feedback

            bl_airborne = ~foot_contacts_before[:, BL_FOOT_INDEX]
            bl_swing_time = torch.where(
                bl_airborne,
                bl_swing_time + raw.step_dt,
                torch.zeros_like(bl_swing_time),
            )
            if args.bl_lift_duration <= 0.0:
                lift_phase = torch.ones_like(bl_swing_time)
            else:
                lift_phase = torch.clamp(bl_swing_time / args.bl_lift_duration, 0.0, 1.0)
            bl_lift_pulse = torch.sin(math.pi * lift_phase) * bl_airborne.float()
            bl_lift_offset = args.bl_lift_boost * bl_lift_pulse
            bl_tibia_offset = args.bl_tibia_boost * bl_lift_pulse
            stabilized_actions[:, BL_FEMUR_INDEX] += bl_lift_offset
            stabilized_actions[:, BL_TIBIA_INDEX] += bl_tibia_offset
            stabilized_actions = torch.clamp(stabilized_actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)

            obs, _, done_tensor, _ = env.step(stabilized_actions)
            if args.root_yaw_rate_damping > 0.0:
                damping = max(0.0, min(1.0, args.root_yaw_rate_damping))
                root_velocity_w = raw._robot.data.root_state_w[:, 7:].clone()
                root_velocity_w[:, 5] *= 1.0 - damping
                raw._robot.write_root_velocity_to_sim(root_velocity_w)

        foot_contacts_now = raw._get_foot_contacts()
        foot_pos_w = raw._robot.data.body_pos_w[:, raw._feet_body_ids, :]
        foot_vel_w = raw._robot.data.body_lin_vel_w[:, raw._feet_body_ids, :]
        foot_xy_speed = torch.norm(foot_vel_w[:, :, :2], dim=2)
        forces = raw._contact_sensor.data.net_forces_w_history
        non_foot_contact = torch.amax(
            torch.norm(forces[:, :, raw._undesired_contact_body_ids], dim=-1),
            dim=(1, 2),
        ) > 1.0
        foot_height = torch.clamp(
            foot_pos_w[:, :, 2] - raw._terrain.env_origins[:, 2].unsqueeze(1),
            min=0.0,
        )
        root_pos_w = raw._robot.data.root_link_pos_w[:, None, :]
        root_quat_w = raw._robot.data.root_link_quat_w[:, None, :]
        foot_body_pos = quat_wxyz_rotate_inverse(root_quat_w, foot_pos_w - root_pos_w)
        times.append(step * raw.step_dt)
        quats.append(raw._robot.data.root_link_quat_w[0].detach().cpu().numpy().copy())
        root_positions.append(raw._robot.data.root_link_pos_w[0].detach().cpu().numpy().copy())
        policy_action_log.append(policy_actions[0].detach().cpu().numpy().copy())
        stabilized_action_log.append(stabilized_actions[0].detach().cpu().numpy().copy())
        processed_target_log.append(raw._processed_actions[0].detach().cpu().numpy().copy())
        joint_pos_log.append(raw._robot.data.joint_pos[0].detach().cpu().numpy().copy())
        joint_vel_log.append(raw._robot.data.joint_vel[0].detach().cpu().numpy().copy())
        lin_vel.append(raw._robot.data.root_lin_vel_b[0].detach().cpu().numpy().copy())
        ang_vel.append(raw._robot.data.root_ang_vel_b[0].detach().cpu().numpy().copy())
        foot_heights.append(foot_height[0].detach().cpu().numpy().copy())
        foot_body_positions.append(foot_body_pos[0].detach().cpu().numpy().copy())
        foot_xy_speeds.append(foot_xy_speed[0].detach().cpu().numpy().copy())
        foot_contacts_log.append(foot_contacts_now[0].detach().cpu().numpy().copy())
        non_foot_contacts_log.append(bool(non_foot_contact[0].detach().cpu().item()))
        bl_lift_offsets.append(float(bl_lift_offset[0].detach().cpu().item()))
        bl_tibia_offsets.append(float(bl_tibia_offset[0].detach().cpu().item()))
        stride_action_offsets.append(stride_ik_offset[0].detach().cpu().numpy().copy())
        yaw_feedback_offsets.append(float(feedback.detach().cpu().item()))
        done = bool(done_tensor[0].item())
        dones.append(done)
        if done:
            raw.episode_length_buf.zero_()
        if not args.no_video:
            frame = raw_env.render()
            if frame is not None and frame.size:
                frames.append(frame)

    attitude = summarize_attitude(
        np.asarray(times),
        np.asarray(quats),
        np.asarray(lin_vel),
        np.asarray(ang_vel),
        np.asarray(dones, dtype=bool),
        np.asarray(foot_heights),
        np.asarray(foot_body_positions),
        np.asarray(foot_xy_speeds),
        np.asarray(foot_contacts_log, dtype=bool),
        np.asarray(non_foot_contacts_log, dtype=bool),
        np.asarray(bl_lift_offsets),
        np.asarray(bl_tibia_offsets),
        np.asarray(stride_action_offsets),
    )

    if args.attitude_output is None:
        args.attitude_output = args.output.with_suffix(".attitude.json")
    args.attitude_output.parent.mkdir(parents=True, exist_ok=True)
    args.attitude_output.write_text(json.dumps(attitude, indent=2) + "\n")

    if args.telemetry_output is not None:
        telemetry_path = args.telemetry_output
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        times_array = np.asarray(times)
        root_positions_array = np.asarray(root_positions)
        quats_array = np.asarray(quats)
        rpy_deg = np.rad2deg(quat_wxyz_to_rpy_np(quats_array))
        rpy_deg[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(rpy_deg[:, 2])))
        rpy_deg[:, 2] -= rpy_deg[0, 2]
        lin_vel_array = np.asarray(lin_vel)
        ang_vel_array = np.asarray(ang_vel)
        foot_contacts_array = np.asarray(foot_contacts_log, dtype=bool)
        foot_xy_speeds_array = np.asarray(foot_xy_speeds)
        stride_offsets_array = np.asarray(stride_action_offsets)
        header = [
            "step",
            "time_s",
            "root_x_w",
            "root_y_w",
            "root_z_w",
            "root_vx_b",
            "root_vy_b",
            "root_vz_b",
            "root_wx_b",
            "root_wy_b",
            "root_wz_b",
            "roll_deg",
            "pitch_deg",
            "yaw_drift_deg",
            "yaw_feedback_action",
            "foot_contact_count",
            "stride_ik_max_abs_action_delta",
        ]
        header += [f"contact_{name}" for name in FOOT_NAMES]
        header += [f"foot_xy_speed_{name}" for name in FOOT_NAMES]
        with telemetry_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for step, time_s in enumerate(times_array):
                row = [
                    step,
                    time_s,
                    *root_positions_array[step].tolist(),
                    *lin_vel_array[step].tolist(),
                    *ang_vel_array[step].tolist(),
                    *rpy_deg[step].tolist(),
                    yaw_feedback_offsets[step],
                    int(np.sum(foot_contacts_array[step])),
                    float(np.max(np.abs(stride_offsets_array[step]))),
                ]
                row += foot_contacts_array[step].astype(int).tolist()
                row += foot_xy_speeds_array[step].tolist()
                writer.writerow(row)

    if args.trajectory_output is not None:
        trajectory_path = args.trajectory_output
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory_path,
            time_s=np.asarray(times, dtype=np.float32),
            joint_names=np.asarray(JOINT_NAMES),
            foot_names=np.asarray(FOOT_NAMES),
            default_joint_pos=raw._robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32),
            action_scale=np.asarray(float(raw.cfg.action_scale), dtype=np.float32),
            step_dt=np.asarray(float(raw.step_dt), dtype=np.float32),
            policy_actions=np.asarray(policy_action_log, dtype=np.float32),
            stabilized_actions=np.asarray(stabilized_action_log, dtype=np.float32),
            target_joint_pos=np.asarray(processed_target_log, dtype=np.float32),
            actual_joint_pos=np.asarray(joint_pos_log, dtype=np.float32),
            actual_joint_vel=np.asarray(joint_vel_log, dtype=np.float32),
            root_pos_w=np.asarray(root_positions, dtype=np.float32),
            root_quat_w=np.asarray(quats, dtype=np.float32),
            root_lin_vel_b=np.asarray(lin_vel, dtype=np.float32),
            root_ang_vel_b=np.asarray(ang_vel, dtype=np.float32),
            foot_contacts=np.asarray(foot_contacts_log, dtype=bool),
            config_json=np.asarray(json.dumps(attitude, sort_keys=True)),
        )
        print(f"TRAJECTORY={trajectory_path}", flush=True)

    if not args.no_video:
        if not frames:
            raise RuntimeError("No frames captured.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.output, frames, fps=args.fps)
        print(f"VIDEO={args.output}", flush=True)
    print(f"ATTITUDE={args.attitude_output}", flush=True)
    print(f"ROLL_P2P_DEG={attitude['roll_peak_to_peak_deg']:.3f}", flush=True)
    print(f"PITCH_P2P_DEG={attitude['pitch_peak_to_peak_deg']:.3f}", flush=True)
    print(f"YAW_DRIFT_P2P_DEG={attitude['yaw_drift_peak_to_peak_deg']:.3f}", flush=True)
    print(f"MEAN_VEL_Y={attitude['mean_root_lin_vel_b'][1]:.3f}", flush=True)
    print(f"BL_MEAN_SWING_HEIGHT_M={attitude['bl_foot_mean_swing_height_m']:.4f}", flush=True)
    print(f"BL_MAX_SWING_HEIGHT_M={attitude['bl_foot_max_swing_height_m']:.4f}", flush=True)
    print(f"BL_MEAN_SWING_BODY_XY_RADIUS_M={attitude['bl_foot_mean_swing_body_xy_radius_m']:.4f}", flush=True)
    print(f"BL_MEAN_STANCE_XY_SPEED_MPS={attitude['bl_foot_mean_stance_xy_speed_mps']:.4f}", flush=True)
    print(f"STRIDE_IK_MAX_ABS_ACTION_DELTA={attitude.get('stride_ik_max_abs_action_delta', 0.0):.4f}", flush=True)
    print(f"NON_FOOT_CONTACT_FRACTION={attitude['non_foot_contact_fraction']:.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
