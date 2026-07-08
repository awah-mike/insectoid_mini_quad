#!/usr/bin/env python3
"""Bake an additional back-left swing clearance pulse into a cyclic gait NPZ."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


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
FOOT_NAMES = ("BL_FOOT", "BR_FOOT", "ML_FOOT", "MR_FOOT")
LEG_JOINT_INDICES = {
    "BL_FOOT": (0, 4, 8),
    "BR_FOOT": (1, 5, 9),
    "ML_FOOT": (2, 6, 10),
    "MR_FOOT": (3, 7, 11),
}
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


def rotation_matrix_from_rpy(rpy: tuple[float, float, float], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        device=device,
        dtype=dtype,
    )


def z_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((cos_angle, -sin_angle, zeros), dim=-1),
            torch.stack((sin_angle, cos_angle, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=-2,
    )


def transform_points(rotation: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return torch.matmul(rotation, points.unsqueeze(-1)).squeeze(-1)


def leg_foot_position_and_jacobian(foot_name: str, q_leg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = LEG_KINEMATICS[foot_name]
    num_samples = q_leg.shape[0]
    device = q_leg.device
    dtype = q_leg.dtype
    position = torch.zeros(num_samples, 3, device=device, dtype=dtype)
    rotation = torch.eye(3, device=device, dtype=dtype).expand(num_samples, 3, 3).clone()
    joint_positions = []
    joint_axes = []

    for joint_index, (origin, origin_rpy) in enumerate(zip(cfg["joint_origins"], cfg["joint_rpy"], strict=True)):
        origin_tensor = torch.tensor(origin, device=device, dtype=dtype).expand(num_samples, 3)
        position = position + transform_points(rotation, origin_tensor)
        origin_rot = rotation_matrix_from_rpy(origin_rpy, device=device, dtype=dtype)
        axis_frame_rotation = torch.matmul(rotation, origin_rot)
        axis_z = axis_frame_rotation[:, :, 2]
        joint_positions.append(position)
        joint_axes.append(axis_z)
        rotation = torch.matmul(axis_frame_rotation, z_rotation_matrix(q_leg[:, joint_index]))

    foot_offset = torch.tensor(cfg["foot_offset"], device=device, dtype=dtype).expand(num_samples, 3)
    foot_position = position + transform_points(rotation, foot_offset)
    jacobian_columns = [
        torch.cross(axis, foot_position - joint_position, dim=-1)
        for axis, joint_position in zip(joint_axes, joint_positions, strict=True)
    ]
    return foot_position, torch.stack(jacobian_columns, dim=-1)


def compute_cyclic_swing_lift_pulses(foot_contacts: np.ndarray, foot_index: int, dilate_steps: int) -> np.ndarray:
    airborne = ~foot_contacts.astype(bool)
    foot_airborne = airborne[:, foot_index].copy()
    if dilate_steps > 0:
        dilated = foot_airborne.copy()
        for offset in range(1, dilate_steps + 1):
            dilated |= np.roll(foot_airborne, offset)
            dilated |= np.roll(foot_airborne, -offset)
        foot_airborne = dilated

    pulses = np.zeros(foot_airborne.shape[0], dtype=np.float32)
    if not np.any(foot_airborne):
        return pulses

    for sample_index in np.flatnonzero(foot_airborne):
        run_start = sample_index
        while foot_airborne[(run_start - 1) % foot_airborne.shape[0]]:
            run_start = (run_start - 1) % foot_airborne.shape[0]
        run_length = 1
        while foot_airborne[(run_start + run_length) % foot_airborne.shape[0]]:
            run_length += 1
        run_offset = (sample_index - run_start) % foot_airborne.shape[0]
        phase = (run_offset + 0.5) / run_length
        pulses[sample_index] = math.sin(math.pi * phase)
    return pulses


def compute_cyclic_swing_phases(foot_contacts: np.ndarray, foot_index: int, dilate_steps: int = 0) -> np.ndarray:
    foot_airborne = ~foot_contacts.astype(bool)[:, foot_index]
    if dilate_steps > 0:
        dilated = foot_airborne.copy()
        for offset in range(1, dilate_steps + 1):
            dilated |= np.roll(foot_airborne, offset)
            dilated |= np.roll(foot_airborne, -offset)
        foot_airborne = dilated

    phases = np.full(foot_airborne.shape[0], -1.0, dtype=np.float32)
    if not np.any(foot_airborne):
        return phases

    for sample_index in np.flatnonzero(foot_airborne):
        run_start = sample_index
        while foot_airborne[(run_start - 1) % foot_airborne.shape[0]]:
            run_start = (run_start - 1) % foot_airborne.shape[0]
        run_length = 1
        while foot_airborne[(run_start + run_length) % foot_airborne.shape[0]]:
            run_length += 1
        run_offset = (sample_index - run_start) % foot_airborne.shape[0]
        phases[sample_index] = (run_offset + 0.5) / run_length
    return phases


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (values >= edge1).astype(np.float32)
    t = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def apply_bl_swing_pose_blend(
    actions: np.ndarray,
    foot_contacts: np.ndarray,
    *,
    blend: float,
    phase_start: float,
    phase_full: float,
    phase_fall_start: float,
    fall_amount: float,
    target_coxa_action: float | None,
    target_femur_action: float,
    target_tibia_action: float,
    dilate_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros(actions.shape[0], dtype=np.float32)
    if blend <= 0.0:
        return actions, weights

    swing_phases = compute_cyclic_swing_phases(foot_contacts, foot_index=0, dilate_steps=dilate_steps)
    swing_mask = swing_phases >= 0.0
    if not np.any(swing_mask):
        return actions, weights

    rise = smoothstep(phase_start, phase_full, swing_phases[swing_mask])
    fall = 1.0 - np.clip(fall_amount, 0.0, 1.0) * smoothstep(phase_fall_start, 1.0, swing_phases[swing_mask])
    weights[swing_mask] = np.clip(blend * rise * fall, 0.0, 1.0)

    blended = actions.copy()
    if target_coxa_action is None:
        target = np.asarray([target_femur_action, target_tibia_action], dtype=np.float32)
        joint_indices = [LEG_JOINT_INDICES["BL_FOOT"][1], LEG_JOINT_INDICES["BL_FOOT"][2]]
    else:
        target = np.asarray([target_coxa_action, target_femur_action, target_tibia_action], dtype=np.float32)
        joint_indices = list(LEG_JOINT_INDICES["BL_FOOT"])
    blended[:, joint_indices] = (
        (1.0 - weights[:, None]) * blended[:, joint_indices]
        + weights[:, None] * target[None, :]
    )
    return np.clip(blended, -1.0, 1.0).astype(np.float32), weights


def compute_bl_lift_offsets(
    joint_pos: np.ndarray,
    lift_pulses: np.ndarray,
    action_scale: float,
    *,
    lift_height_m: float,
    joints: str,
    damping: float,
    action_gain: float,
    max_action_delta: float,
) -> np.ndarray:
    q = torch.as_tensor(joint_pos[:, list(LEG_JOINT_INDICES["BL_FOOT"])], dtype=torch.float32)
    pulse = torch.as_tensor(lift_pulses, dtype=torch.float32)
    _, jacobian = leg_foot_position_and_jacobian("BL_FOOT", q)
    selection = LOCAL_JOINT_SELECTIONS[joints]
    jacobian_selected = jacobian[:, :, list(selection)]
    delta_foot = torch.zeros(joint_pos.shape[0], 3, dtype=torch.float32)
    delta_foot[:, 2] = lift_height_m * pulse
    eye3 = torch.eye(3, dtype=torch.float32).expand(joint_pos.shape[0], 3, 3)
    lhs = torch.matmul(jacobian_selected, jacobian_selected.transpose(1, 2)) + (damping * damping) * eye3
    solved = torch.linalg.solve(lhs, delta_foot[:, :, None]).squeeze(-1)
    delta_q_selected = torch.matmul(jacobian_selected.transpose(1, 2), solved[:, :, None]).squeeze(-1)
    delta_action = torch.clamp(action_gain * delta_q_selected / action_scale, -max_action_delta, max_action_delta)

    offsets = torch.zeros(joint_pos.shape[0], len(JOINT_NAMES), dtype=torch.float32)
    bl_indices = LEG_JOINT_INDICES["BL_FOOT"]
    for local_column, local_joint_index in enumerate(selection):
        offsets[:, bl_indices[local_joint_index]] = delta_action[:, local_column]
    return offsets.numpy()


def parse_sample_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def cyclic_contact_blocks(contact_mask: np.ndarray) -> list[list[int]]:
    sample_count = int(contact_mask.shape[0])
    if sample_count == 0 or not np.any(contact_mask):
        return []
    if np.all(contact_mask):
        return [list(range(sample_count))]

    starts = [
        index
        for index in range(sample_count)
        if contact_mask[index] and not contact_mask[(index - 1) % sample_count]
    ]
    blocks = []
    for start in starts:
        block = [start]
        cursor = (start + 1) % sample_count
        while cursor != start and contact_mask[cursor]:
            block.append(cursor)
            cursor = (cursor + 1) % sample_count
        blocks.append(block)
    return blocks


def solve_bl_foot_path(
    base_target_joint_pos: np.ndarray,
    default_joint_pos: np.ndarray,
    action_scale: float,
    target_foot_pos: np.ndarray,
    active_mask: np.ndarray,
    *,
    damping: float,
    iterations: int,
    max_step_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_full = torch.as_tensor(base_target_joint_pos.copy(), dtype=torch.float32)
    target = torch.as_tensor(target_foot_pos, dtype=torch.float32)
    active = torch.as_tensor(active_mask.astype(bool))
    bl_indices = list(LEG_JOINT_INDICES["BL_FOOT"])
    default = torch.as_tensor(default_joint_pos, dtype=torch.float32)
    damping_sq = damping * damping
    eye3 = torch.eye(3, dtype=torch.float32).expand(q_full.shape[0], 3, 3)
    max_step = abs(max_step_rad)

    if not torch.any(active):
        actions = (q_full.numpy() - default_joint_pos[None, :]) / action_scale
        return np.clip(actions, -1.0, 1.0).astype(np.float32), q_full.numpy().astype(np.float32), np.zeros_like(
            base_target_joint_pos, dtype=np.float32
        )

    for _ in range(max(1, iterations)):
        q_leg = q_full[:, bl_indices]
        foot_pos, jacobian = leg_foot_position_and_jacobian("BL_FOOT", q_leg)
        delta_foot = target - foot_pos
        delta_foot[~active] = 0.0
        lhs = torch.matmul(jacobian, jacobian.transpose(1, 2)) + damping_sq * eye3
        solved = torch.linalg.solve(lhs, delta_foot[:, :, None]).squeeze(-1)
        delta_q = torch.matmul(jacobian.transpose(1, 2), solved[:, :, None]).squeeze(-1)
        delta_q = torch.clamp(delta_q, -max_step, max_step)
        delta_q[~active] = 0.0
        q_full[:, bl_indices] += delta_q
        action_full = torch.clamp((q_full - default[None, :]) / action_scale, -1.0, 1.0)
        q_full = default[None, :] + action_scale * action_full

    actions = torch.clamp((q_full - default[None, :]) / action_scale, -1.0, 1.0).numpy()
    target_joint_pos = (default_joint_pos[None, :] + action_scale * actions).astype(np.float32)
    offsets = ((target_joint_pos - base_target_joint_pos) / action_scale).astype(np.float32)
    return actions.astype(np.float32), target_joint_pos, offsets


def apply_bl_stance_path_cleanup(
    actions: np.ndarray,
    foot_contacts: np.ndarray,
    default_joint_pos: np.ndarray,
    action_scale: float,
    *,
    push_distance_m: float,
    stance_samples: list[int],
    swing_samples: list[int],
    swing_extra_height_m: float,
    damping: float,
    iterations: int,
    max_step_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    base_target_joint_pos = default_joint_pos[None, :] + action_scale * actions
    sample_count = actions.shape[0]
    bl_indices = list(LEG_JOINT_INDICES["BL_FOOT"])
    foot_pos, _ = leg_foot_position_and_jacobian(
        "BL_FOOT",
        torch.as_tensor(base_target_joint_pos[:, bl_indices], dtype=torch.float32),
    )
    foot_pos_np = foot_pos.numpy().astype(np.float32)
    target_foot_pos = foot_pos_np.copy()
    active_mask = np.zeros(sample_count, dtype=bool)

    if not stance_samples:
        blocks = cyclic_contact_blocks(foot_contacts[:, 0].astype(bool))
        stance_samples = max(blocks, key=len) if blocks else []
    if not swing_samples:
        stance_set = set(stance_samples)
        swing_samples = [index for index in range(sample_count) if index not in stance_set]

    if stance_samples:
        stance_y0 = float(foot_pos_np[stance_samples[0], 1])
        stance_y1 = stance_y0 - abs(push_distance_m)
        stance_x0 = float(foot_pos_np[stance_samples[0], 0])
        stance_x1 = float(foot_pos_np[stance_samples[-1], 0])
        stance_low_z = float(np.min(foot_pos_np[stance_samples, 2]))
        stance_start_z = float(foot_pos_np[stance_samples[0], 2])
        stance_end_z = float(max(foot_pos_np[stance_samples[-1], 2], stance_low_z + 0.07))
        denom = max(1, len(stance_samples) - 1)
        for order, sample_index in enumerate(stance_samples):
            phase = order / denom
            target_foot_pos[sample_index, 0] = (1.0 - phase) * stance_x0 + phase * stance_x1
            target_foot_pos[sample_index, 1] = (1.0 - phase) * stance_y0 + phase * stance_y1
            if phase < 0.18:
                z_phase = phase / 0.18
                target_foot_pos[sample_index, 2] = (1.0 - z_phase) * stance_start_z + z_phase * stance_low_z
            elif phase > 0.88:
                z_phase = (phase - 0.88) / 0.12
                target_foot_pos[sample_index, 2] = (1.0 - z_phase) * stance_low_z + z_phase * stance_end_z
            else:
                target_foot_pos[sample_index, 2] = stance_low_z
            active_mask[sample_index] = True

    if swing_samples and stance_samples:
        swing_start = target_foot_pos[stance_samples[-1]].copy()
        swing_end = target_foot_pos[stance_samples[0]].copy()
        high_z = max(float(np.max(foot_pos_np[swing_samples, 2])) + abs(swing_extra_height_m), float(swing_start[2]))
        denom = max(1, len(swing_samples) - 1)
        for order, sample_index in enumerate(swing_samples):
            phase = order / denom
            smooth = phase * phase * (3.0 - 2.0 * phase)
            target_foot_pos[sample_index, 0] = (1.0 - smooth) * swing_start[0] + smooth * swing_end[0]
            target_foot_pos[sample_index, 1] = (1.0 - smooth) * swing_start[1] + smooth * swing_end[1]
            target_foot_pos[sample_index, 2] = max(
                (1.0 - smooth) * swing_start[2] + smooth * swing_end[2],
                high_z,
            )
            active_mask[sample_index] = True

    baked_actions, baked_target_joint_pos, action_offsets = solve_bl_foot_path(
        base_target_joint_pos,
        default_joint_pos,
        action_scale,
        target_foot_pos,
        active_mask,
        damping=damping,
        iterations=iterations,
        max_step_rad=max_step_rad,
    )
    achieved_foot_pos, _ = leg_foot_position_and_jacobian(
        "BL_FOOT",
        torch.as_tensor(baked_target_joint_pos[:, bl_indices], dtype=torch.float32),
    )
    achieved_foot_pos_np = achieved_foot_pos.numpy().astype(np.float32)
    metadata = {
        "bl_foot_path_cleanup": True,
        "bl_foot_path_stance_samples": [int(index) for index in stance_samples],
        "bl_foot_path_swing_samples": [int(index) for index in swing_samples],
        "bl_foot_path_push_distance_m": float(abs(push_distance_m)),
        "bl_foot_path_swing_extra_height_m": float(abs(swing_extra_height_m)),
        "bl_foot_path_target_xyz_body": target_foot_pos.astype(float).tolist(),
        "bl_foot_path_achieved_xyz_body": achieved_foot_pos_np.astype(float).tolist(),
        "bl_foot_path_max_abs_action_offset": float(np.max(np.abs(action_offsets))),
    }
    return baked_actions, baked_target_joint_pos, action_offsets, target_foot_pos, metadata


def write_cycle_csv(path: Path, time_s: np.ndarray, values: np.ndarray, header_names: tuple[str, ...] | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("sample", "phase", "time_s", *header_names))
        sample_count = values.shape[0]
        for sample_index, row in enumerate(values):
            phase = sample_index / sample_count
            writer.writerow((sample_index, phase, float(time_s[sample_index]), *[float(value) for value in row]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="model_1798_cycle19_bl_dragfix_v3_clearance")
    parser.add_argument("--lift-height-m", type=float, default=0.08)
    parser.add_argument("--dilate-steps", type=int, default=3)
    parser.add_argument("--lift-joints", choices=tuple(LOCAL_JOINT_SELECTIONS), default="all")
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--action-gain", type=float, default=1.5)
    parser.add_argument("--max-action-delta", type=float, default=1.0)
    parser.add_argument("--bl-swing-pose-blend", type=float, default=0.0)
    parser.add_argument("--bl-swing-pose-dilate-steps", type=int, default=0)
    parser.add_argument("--bl-swing-pose-phase-start", type=float, default=0.35)
    parser.add_argument("--bl-swing-pose-phase-full", type=float, default=0.60)
    parser.add_argument("--bl-swing-pose-phase-fall-start", type=float, default=0.95)
    parser.add_argument("--bl-swing-pose-fall-amount", type=float, default=0.0)
    parser.add_argument("--bl-swing-target-coxa-action", type=float, default=None)
    parser.add_argument("--bl-swing-target-femur-action", type=float, default=-1.0)
    parser.add_argument("--bl-swing-target-tibia-action", type=float, default=-1.0)
    parser.add_argument("--bl-foot-path-cleanup", action="store_true")
    parser.add_argument("--bl-foot-path-push-distance-m", type=float, default=0.12)
    parser.add_argument(
        "--bl-foot-path-stance-samples",
        default="",
        help="Optional comma-separated cyclic BL stance samples. Defaults to the longest BL contact block.",
    )
    parser.add_argument(
        "--bl-foot-path-swing-samples",
        default="",
        help="Optional comma-separated BL swing samples. Defaults to all samples outside the stance block.",
    )
    parser.add_argument("--bl-foot-path-swing-extra-height-m", type=float, default=0.015)
    parser.add_argument("--bl-foot-path-damping", type=float, default=0.015)
    parser.add_argument("--bl-foot-path-iterations", type=int, default=8)
    parser.add_argument("--bl-foot-path-max-step-rad", type=float, default=0.16)
    args = parser.parse_args()

    source = np.load(args.source_npz, allow_pickle=True)
    time_s = source["time_s"].astype(np.float32)
    actions = source["stabilized_actions"].astype(np.float32)
    default_joint_pos = source["default_joint_pos"].astype(np.float32)
    action_scale = float(source["action_scale"])
    foot_contacts = source["foot_contacts"].astype(np.int32)
    joint_names = source["joint_names"]
    foot_names = source["foot_names"]
    step_dt = np.asarray(source["step_dt"]).astype(np.float32)

    if tuple(joint_names.tolist()) != JOINT_NAMES:
        raise RuntimeError(f"Unexpected joint order: {joint_names}")
    if tuple(foot_names.tolist()) != FOOT_NAMES:
        raise RuntimeError(f"Unexpected foot order: {foot_names}")

    base_target_joint_pos = default_joint_pos[None, :] + action_scale * actions
    lift_pulses = compute_cyclic_swing_lift_pulses(foot_contacts, foot_index=0, dilate_steps=args.dilate_steps)
    extra_offsets = compute_bl_lift_offsets(
        base_target_joint_pos,
        lift_pulses,
        action_scale,
        lift_height_m=args.lift_height_m,
        joints=args.lift_joints,
        damping=args.damping,
        action_gain=args.action_gain,
        max_action_delta=args.max_action_delta,
    )
    baked_actions = np.clip(actions + extra_offsets, -1.0, 1.0).astype(np.float32)
    baked_actions, swing_pose_weights = apply_bl_swing_pose_blend(
        baked_actions,
        foot_contacts,
        blend=args.bl_swing_pose_blend,
        phase_start=args.bl_swing_pose_phase_start,
        phase_full=args.bl_swing_pose_phase_full,
        phase_fall_start=args.bl_swing_pose_phase_fall_start,
        fall_amount=args.bl_swing_pose_fall_amount,
        target_coxa_action=args.bl_swing_target_coxa_action,
        target_femur_action=args.bl_swing_target_femur_action,
        target_tibia_action=args.bl_swing_target_tibia_action,
        dilate_steps=max(0, args.bl_swing_pose_dilate_steps),
    )
    foot_path_offsets = np.zeros_like(baked_actions, dtype=np.float32)
    foot_path_metadata: dict[str, object] = {"bl_foot_path_cleanup": False}
    if args.bl_foot_path_cleanup:
        baked_actions, baked_target_joint_pos, foot_path_offsets, _, foot_path_metadata = apply_bl_stance_path_cleanup(
            baked_actions,
            foot_contacts,
            default_joint_pos,
            action_scale,
            push_distance_m=args.bl_foot_path_push_distance_m,
            stance_samples=parse_sample_list(args.bl_foot_path_stance_samples),
            swing_samples=parse_sample_list(args.bl_foot_path_swing_samples),
            swing_extra_height_m=args.bl_foot_path_swing_extra_height_m,
            damping=args.bl_foot_path_damping,
            iterations=args.bl_foot_path_iterations,
            max_step_rad=args.bl_foot_path_max_step_rad,
        )
    else:
        baked_target_joint_pos = (default_joint_pos[None, :] + action_scale * baked_actions).astype(np.float32)
    target_joint_vel = np.gradient(baked_target_joint_pos, float(step_dt), axis=0).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = args.output_dir / f"{args.name}_replay_compat.npz"
    np.savez(
        output_npz,
        time_s=time_s,
        stabilized_actions=baked_actions,
        target_joint_pos=baked_target_joint_pos,
        actual_joint_pos=baked_target_joint_pos,
        actual_joint_vel=target_joint_vel,
        root_lin_vel_b=source["root_lin_vel_b"].astype(np.float32),
        root_ang_vel_b=source["root_ang_vel_b"].astype(np.float32),
        foot_contacts=foot_contacts,
        default_joint_pos=default_joint_pos,
        action_scale=np.asarray(action_scale, dtype=np.float32),
        joint_names=joint_names,
        foot_names=foot_names,
        step_dt=step_dt,
        lift_pulses=lift_pulses.astype(np.float32),
        lift_action_offsets=extra_offsets.astype(np.float32),
        bl_swing_pose_weights=swing_pose_weights.astype(np.float32),
        bl_foot_path_action_offsets=foot_path_offsets.astype(np.float32),
    )

    write_cycle_csv(args.output_dir / f"{args.name}_policy_actions.csv", time_s, baked_actions, joint_names)
    write_cycle_csv(args.output_dir / f"{args.name}_target_joint_positions_rad.csv", time_s, baked_target_joint_pos, joint_names)
    write_cycle_csv(args.output_dir / f"{args.name}_target_joint_velocities_radps.csv", time_s, target_joint_vel, joint_names)
    write_cycle_csv(args.output_dir / f"{args.name}_contacts.csv", time_s, foot_contacts, foot_names)

    summary = {
        "source": str(args.source_npz),
        "output_npz": str(output_npz),
        "method": "Additional BL-only IK swing clearance baked on top of the v2 drag-fix cycle.",
        "extra_lift_height_m": args.lift_height_m,
        "lift_joints": args.lift_joints,
        "swing_lift_dilate_steps": args.dilate_steps,
        "swing_lift_max_action_delta": args.max_action_delta,
        "swing_lift_action_gain": args.action_gain,
        "action_clip": [-1.0, 1.0],
        "max_applied_action_offset": float(np.max(np.abs(extra_offsets))),
        "bl_swing_pose_blend": args.bl_swing_pose_blend,
        "bl_swing_pose_dilate_steps": args.bl_swing_pose_dilate_steps,
        "bl_swing_pose_phase_start": args.bl_swing_pose_phase_start,
        "bl_swing_pose_phase_full": args.bl_swing_pose_phase_full,
        "bl_swing_pose_phase_fall_start": args.bl_swing_pose_phase_fall_start,
        "bl_swing_pose_fall_amount": args.bl_swing_pose_fall_amount,
        "bl_swing_target_coxa_action": args.bl_swing_target_coxa_action,
        "bl_swing_target_femur_action": args.bl_swing_target_femur_action,
        "bl_swing_target_tibia_action": args.bl_swing_target_tibia_action,
        "bl_swing_pose_weights": swing_pose_weights.astype(float).tolist(),
        **foot_path_metadata,
        "active_lift_samples": np.flatnonzero(lift_pulses > 0.0).astype(int).tolist(),
        "lift_pulses": lift_pulses.astype(float).tolist(),
        "joint_order": joint_names.tolist(),
        "foot_order": foot_names.tolist(),
        "primary_table": f"{args.name}_target_joint_positions_rad.csv",
    }
    summary_path = args.output_dir / f"{args.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"OUTPUT_NPZ={output_npz}")
    print(f"SUMMARY={summary_path}")
    print(f"MAX_ACTION_OFFSET={summary['max_applied_action_offset']:.4f}")


if __name__ == "__main__":
    main()
