#!/usr/bin/env python3
"""Render four insectoid robots carrying one payload as a coordinated choreography."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD_PATH = REPO_ROOT / "URDF_description/usd/insectoid_mini_quad.usd"
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
parser.add_argument(
    "--trajectory-npz",
    type=Path,
    default=REPO_ROOT / "data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz",
)
parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/renders/four_robot_payload_carry_primitives.mp4")
parser.add_argument("--summary-output", type=Path, default=None)
parser.add_argument("--duration", type=float, default=11.2)
parser.add_argument("--playback-speed", type=float, default=2.5)
parser.add_argument(
    "--gait-ramp-s",
    type=float,
    default=0.0,
    help="Ease moving segments in/out by blending roots and joints against the static hold pose.",
)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--root-z", type=float, default=0.15)
parser.add_argument("--payload-z", type=float, default=0.23)
parser.add_argument("--payload-size", type=float, nargs=3, default=(1.20, 0.62, 0.18))
parser.add_argument("--robot-side-y", type=float, default=0.60)
parser.add_argument("--robot-pair-half-x", type=float, default=0.54)
parser.add_argument("--segment-speed", type=float, default=0.18)
parser.add_argument("--settle-s", type=float, default=1.0)
parser.add_argument("--move-s", type=float, default=2.0)
parser.add_argument("--pause-s", type=float, default=0.4)
parser.add_argument("--camera-eye", type=float, nargs=3, default=(2.4, -3.8, 2.8))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(0.0, 0.0, 0.16))
parser.add_argument("--env-spacing", type=float, default=1.5)
parser.add_argument("--dynamic-roots", action="store_true", help="Drive joint targets through PhysX and let floor contacts move the robots.")
parser.add_argument("--physics-dt", type=float, default=0.005, help="Physics step size for --dynamic-roots.")
parser.add_argument("--closed-loop-correction", action="store_true", help="Blend corrective primitives and apply low-gain physical formation/yaw stabilization.")
parser.add_argument("--slot-correction-gain", type=float, default=1.35)
parser.add_argument("--center-correction-gain", type=float, default=0.35)
parser.add_argument("--correction-max-blend", type=float, default=0.50)
parser.add_argument("--correction-max-total-blend", type=float, default=0.75)
parser.add_argument("--yaw-coxa-gain", type=float, default=0.10)
parser.add_argument("--yaw-coxa-max-deg", type=float, default=7.0)
parser.add_argument("--formation-force-kp", type=float, default=3.0)
parser.add_argument("--formation-force-kd", type=float, default=0.7)
parser.add_argument("--formation-max-force", type=float, default=1.8)
parser.add_argument("--yaw-torque-kp", type=float, default=0.18)
parser.add_argument("--yaw-torque-kd", type=float, default=0.035)
parser.add_argument("--yaw-max-torque", type=float, default=0.22)
parser.add_argument("--no-video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_video

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import carb  # noqa: E402
import isaacsim.core.utils.prims as prim_utils  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.prims import Articulation as CoreArticulation  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation as LabArticulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

if not args.no_video:
    import omni.replicator.core as rep  # noqa: E402
else:
    rep = None

import insectoid_mini_quad_rl  # noqa: F401, E402
from insectoid_mini_quad_rl.articulation import INSECTOID_MINI_QUAD_CFG  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import omni.usd  # noqa: E402

print("[INFO]: Four-robot payload renderer imports complete.", flush=True)


@dataclass(frozen=True)
class PrimitiveSpec:
    start_s: float
    end_s: float
    reverse: bool = False
    sideways_remap_scale: float = 0.0
    sideways_remap_sign: float = 1.0
    sideways_remap_y_keep_scale: float = 1.0
    sideways_remap_max_action_delta: float = 0.6
    swing_lift_height_m: float = 0.0
    swing_lift_middle_scale: float = 1.0
    swing_lift_dilate_steps: int = 0
    swing_lift_max_action_delta: float = 0.35


@dataclass(frozen=True)
class RobotSlot:
    name: str
    offset_xy: tuple[float, float]
    yaw: float
    opposite_side: bool
    phase_offset: int = 0


@dataclass(frozen=True)
class Segment:
    name: str
    duration_s: float
    command_xy: tuple[float, float]


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
            pulses[sample_index, foot_index] = math.sin(math.pi * ((run_offset + 0.5) / run_length))
    return pulses


def compute_sideways_remap_action_offsets(
    sequence_joint_pos: torch.Tensor,
    action_scale: float,
    scale: float,
    sign: float,
    y_keep_scale: float,
    max_action_delta: float,
) -> torch.Tensor:
    offsets = torch.zeros(sequence_joint_pos.shape[0], len(JOINT_NAMES), device=sequence_joint_pos.device, dtype=sequence_joint_pos.dtype)
    if scale == 0.0:
        return offsets

    damping_sq = 0.01 * 0.01
    eye3 = torch.eye(3, device=sequence_joint_pos.device, dtype=sequence_joint_pos.dtype).expand(sequence_joint_pos.shape[0], 3, 3)
    side_sign = 1.0 if sign >= 0.0 else -1.0
    for foot_name in FOOT_NAMES:
        leg_joint_indices = LEG_JOINT_INDICES[foot_name]
        q_leg = sequence_joint_pos[:, list(leg_joint_indices)]
        foot_position, jacobian = leg_foot_position_and_jacobian(foot_name, q_leg)
        mean_foot_position = torch.mean(foot_position, dim=0, keepdim=True)
        foot_delta = foot_position - mean_foot_position
        target_foot_position = foot_position.clone()
        target_foot_position[:, 0] = mean_foot_position[:, 0] + foot_delta[:, 0] + side_sign * scale * foot_delta[:, 1]
        target_foot_position[:, 1] = mean_foot_position[:, 1] + y_keep_scale * foot_delta[:, 1]
        delta_foot = target_foot_position - foot_position
        if torch.max(torch.abs(delta_foot)).item() <= 1.0e-6:
            continue

        jacobian_selected = jacobian
        lhs = torch.matmul(jacobian_selected, jacobian_selected.transpose(1, 2)) + damping_sq * eye3
        solved = torch.linalg.solve(lhs, delta_foot[:, :, None]).squeeze(-1)
        delta_q = torch.matmul(jacobian_selected.transpose(1, 2), solved[:, :, None]).squeeze(-1)
        delta_action = torch.clamp(delta_q / action_scale, -max_action_delta, max_action_delta)
        for local_column, global_joint_index in enumerate(leg_joint_indices):
            offsets[:, global_joint_index] += delta_action[:, local_column]
    return offsets


def compute_swing_lift_action_offsets(
    sequence_joint_pos: torch.Tensor,
    lift_pulses: torch.Tensor,
    action_scale: float,
    height_m: float,
    middle_scale: float,
    max_action_delta: float,
) -> torch.Tensor:
    offsets = torch.zeros(sequence_joint_pos.shape[0], len(JOINT_NAMES), device=sequence_joint_pos.device, dtype=sequence_joint_pos.dtype)
    if height_m == 0.0:
        return offsets

    damping_sq = 0.01 * 0.01
    eye3 = torch.eye(3, device=sequence_joint_pos.device, dtype=sequence_joint_pos.dtype).expand(sequence_joint_pos.shape[0], 3, 3)
    for foot_index, foot_name in enumerate(FOOT_NAMES):
        leg_scale = 1.0 if foot_index in REAR_FOOT_INDICES else middle_scale
        pulse = lift_pulses[:, foot_index]
        if torch.max(torch.abs(pulse)).item() <= 1.0e-6:
            continue
        leg_joint_indices = LEG_JOINT_INDICES[foot_name]
        q_leg = sequence_joint_pos[:, list(leg_joint_indices)]
        _, jacobian = leg_foot_position_and_jacobian(foot_name, q_leg)
        jacobian_selected = jacobian[:, :, [1, 2]]
        delta_foot = torch.zeros(sequence_joint_pos.shape[0], 3, device=sequence_joint_pos.device, dtype=sequence_joint_pos.dtype)
        delta_foot[:, 2] = height_m * leg_scale * pulse
        lhs = torch.matmul(jacobian_selected, jacobian_selected.transpose(1, 2)) + damping_sq * eye3
        solved = torch.linalg.solve(lhs, delta_foot[:, :, None]).squeeze(-1)
        delta_q = torch.matmul(jacobian_selected.transpose(1, 2), solved[:, :, None]).squeeze(-1)
        delta_action = torch.clamp(delta_q / action_scale, -max_action_delta, max_action_delta)
        for local_column, global_joint_index in enumerate((leg_joint_indices[1], leg_joint_indices[2])):
            offsets[:, global_joint_index] += delta_action[:, local_column]
    return offsets


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_wxyz_to_yaw_np(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def slot_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    right_axis = np.array((math.cos(yaw), -math.sin(yaw)), dtype=np.float32)
    forward_axis = np.array((math.sin(yaw), math.cos(yaw)), dtype=np.float32)
    return right_axis, forward_axis


def clamp_vector_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return torch.zeros_like(vec)
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1.0e-6), max=1.0)
    return vec * scale


def smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def segment_motion_weight(local_frame: int, segment_frames: int, frame_dt: float, ramp_s: float) -> float:
    if ramp_s <= 1.0e-6 or segment_frames <= 1:
        return 1.0
    segment_s = segment_frames * frame_dt
    ramp_s = min(ramp_s, 0.5 * segment_s)
    if ramp_s <= 1.0e-6:
        return 1.0
    t = (local_frame + 0.5) * frame_dt
    ramp_in = smoothstep01(t / ramp_s)
    ramp_out = smoothstep01((segment_s - t) / ramp_s)
    return min(ramp_in, ramp_out)


def build_primitive_actions(data: np.lib.npyio.NpzFile, spec: PrimitiveSpec, device: torch.device) -> np.ndarray:
    time_s = data["time_s"].astype(np.float32)
    mask = (time_s >= spec.start_s) & (time_s < spec.end_s)
    if not np.any(mask):
        raise ValueError(f"No trajectory samples in window [{spec.start_s}, {spec.end_s}).")
    indices = np.flatnonzero(mask)
    if spec.reverse:
        indices = indices[::-1]
    joint_pos = data["actual_joint_pos"][indices].astype(np.float32)
    default_joint_pos = data["default_joint_pos"].astype(np.float32)
    action_scale = float(data["action_scale"])
    actions = torch.as_tensor((joint_pos - default_joint_pos) / action_scale, dtype=torch.float32, device=device)
    sequence_joint_pos = torch.as_tensor(joint_pos, dtype=torch.float32, device=device)

    actions += compute_sideways_remap_action_offsets(
        sequence_joint_pos,
        action_scale,
        spec.sideways_remap_scale,
        spec.sideways_remap_sign,
        spec.sideways_remap_y_keep_scale,
        spec.sideways_remap_max_action_delta,
    )
    if spec.swing_lift_height_m != 0.0:
        foot_contacts = data["foot_contacts"][indices].astype(bool)
        pulses_np = compute_cyclic_swing_lift_pulses(foot_contacts, spec.swing_lift_dilate_steps)
        pulses = torch.as_tensor(pulses_np, dtype=torch.float32, device=device)
        actions += compute_swing_lift_action_offsets(
            sequence_joint_pos,
            pulses,
            action_scale,
            spec.swing_lift_height_m,
            spec.swing_lift_middle_scale,
            spec.swing_lift_max_action_delta,
        )
    return torch.clamp(actions, -1.0, 1.0).detach().cpu().numpy().astype(np.float32)


def build_primitive_joint_positions(data: np.lib.npyio.NpzFile, spec: PrimitiveSpec, device: torch.device) -> np.ndarray:
    actions = build_primitive_actions(data, spec, device)
    default_joint_pos = data["default_joint_pos"].astype(np.float32)
    action_scale = float(data["action_scale"])
    return (default_joint_pos[None, :] + actions * action_scale).astype(np.float32)


def create_payload_cube(size: tuple[float, float, float]):
    stage = omni.usd.get_context().get_stage()
    cube = UsdGeom.Cube.Define(stage, "/World/PayloadBox")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([(0.55, 0.44, 0.30)])
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetScale(Gf.Vec3f(*size))
    return xform


def set_payload_pose(xform: UsdGeom.XformCommonAPI, pos: np.ndarray) -> None:
    xform.SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


def command_to_name(command_xy: np.ndarray) -> str:
    if np.linalg.norm(command_xy) < 1.0e-6:
        return "hold"
    if abs(command_xy[1]) >= abs(command_xy[0]):
        return "forward" if command_xy[1] >= 0.0 else "backward"
    return "right" if command_xy[0] >= 0.0 else "left"


def mapped_primitive(reference_name: str, opposite_side: bool) -> str:
    if not opposite_side:
        return {
            "hold": "forward",
            "forward": "forward",
            "backward": "backward",
            "left": "xneg",
            "right": "xpos",
        }[reference_name]
    return {
        "hold": "backward",
        "forward": "backward",
        "backward": "forward",
        "left": "xpos",
        "right": "xneg",
    }[reference_name]


def main() -> None:
    data = np.load(args.trajectory_npz, allow_pickle=False)

    env_cfg = parse_env_cfg(args.task, num_envs=4)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 4
    env_cfg.scene.env_spacing = args.env_spacing
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.viewer.eye = tuple(args.camera_eye)
    env_cfg.viewer.lookat = tuple(args.camera_lookat)
    env_cfg.viewer.origin_type = "world"
    if hasattr(env_cfg, "events"):
        env_cfg.events.physics_material = None

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if not args.no_video else None)
    env.reset()
    raw = env.unwrapped
    raw.episode_length_buf.zero_()

    primitive_specs = {
        "forward": PrimitiveSpec(0.72, 1.30),
        "backward": PrimitiveSpec(
            1.30,
            1.88,
            reverse=True,
            swing_lift_height_m=0.03,
            swing_lift_middle_scale=2.0,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.55,
        ),
        "xneg": PrimitiveSpec(
            0.72,
            1.30,
            sideways_remap_scale=1.0,
            sideways_remap_sign=-1.0,
            sideways_remap_y_keep_scale=0.10,
            sideways_remap_max_action_delta=0.65,
            swing_lift_height_m=0.02,
            swing_lift_middle_scale=1.5,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.45,
        ),
        "xpos": PrimitiveSpec(
            0.72,
            1.30,
            sideways_remap_scale=1.5,
            sideways_remap_sign=1.0,
            sideways_remap_y_keep_scale=0.0,
            sideways_remap_max_action_delta=0.85,
            swing_lift_height_m=0.02,
            swing_lift_middle_scale=1.5,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.45,
        ),
    }
    primitives = {name: build_primitive_actions(data, spec, raw.device) for name, spec in primitive_specs.items()}

    slots = [
        RobotSlot("near_left", (-args.robot_pair_half_x, -args.robot_side_y), 0.0, False, 0),
        RobotSlot("near_right", (args.robot_pair_half_x, -args.robot_side_y), 0.0, False, 4),
        RobotSlot("far_left", (-args.robot_pair_half_x, args.robot_side_y), math.pi, True, 4),
        RobotSlot("far_right", (args.robot_pair_half_x, args.robot_side_y), math.pi, True, 0),
    ]
    segments = [
        Segment("settle", args.settle_s, (0.0, 0.0)),
        Segment("forward", args.move_s, (0.0, args.segment_speed)),
        Segment("pause_after_forward", args.pause_s, (0.0, 0.0)),
        Segment("backward", args.move_s, (0.0, -args.segment_speed)),
        Segment("pause_after_backward", args.pause_s, (0.0, 0.0)),
        Segment("left", args.move_s, (-args.segment_speed, 0.0)),
        Segment("pause_after_left", args.pause_s, (0.0, 0.0)),
        Segment("right", args.move_s, (args.segment_speed, 0.0)),
    ]

    payload_pos = np.array((0.0, 0.0, args.payload_z), dtype=np.float32)
    payload_xform = create_payload_cube(tuple(args.payload_size))
    set_payload_pose(payload_xform, payload_pos)

    env_ids = torch.arange(4, device=raw.device, dtype=torch.int64)
    action_tensor = torch.zeros(4, raw.single_action_space.shape[0], device=raw.device)
    root_pose = torch.zeros(4, 7, device=raw.device)
    root_vel = torch.zeros(4, 6, device=raw.device)

    def write_formation(center_xy: np.ndarray, command_xy: np.ndarray) -> None:
        for robot_index, slot in enumerate(slots):
            root_pose[robot_index, 0] = float(center_xy[0] + slot.offset_xy[0])
            root_pose[robot_index, 1] = float(center_xy[1] + slot.offset_xy[1])
            root_pose[robot_index, 2] = args.root_z
            root_pose[robot_index, 3:7] = torch.tensor(yaw_to_quat_wxyz(slot.yaw), dtype=torch.float32, device=raw.device)
            root_vel[robot_index, 0] = float(command_xy[0])
            root_vel[robot_index, 1] = float(command_xy[1])
            root_vel[robot_index, 2:] = 0.0
        raw._robot.write_root_pose_to_sim(root_pose, env_ids)
        raw._robot.write_root_velocity_to_sim(root_vel, env_ids)

    center_xy = np.zeros(2, dtype=np.float32)
    write_formation(center_xy, np.zeros(2, dtype=np.float32))
    default_joint_pos = raw._robot.data.default_joint_pos.clone()
    raw._robot.write_joint_state_to_sim(default_joint_pos, torch.zeros_like(default_joint_pos), None, env_ids)
    raw._actions.zero_()
    raw._previous_actions.zero_()
    raw._processed_actions[:] = default_joint_pos

    frames = []
    payload_positions = []
    segment_log = []
    root_positions = []
    primitive_counts = {name: 0 for name in primitives}
    total_steps = min(max(1, int(args.duration / raw.step_dt)), int(sum(segment.duration_s for segment in segments) / raw.step_dt))
    capture_every = max(1, round(1.0 / (args.fps * raw.step_dt)))
    step_cursor = 0

    for segment in segments:
        segment_steps = int(segment.duration_s / raw.step_dt)
        command_xy = np.asarray(segment.command_xy, dtype=np.float32)
        for local_step in range(segment_steps):
            if step_cursor >= total_steps:
                break
            reference_name = command_to_name(command_xy)
            for robot_index, slot in enumerate(slots):
                primitive_name = mapped_primitive(reference_name, slot.opposite_side)
                primitive = primitives[primitive_name]
                sequence_index = (int(local_step * args.playback_speed) + slot.phase_offset) % primitive.shape[0]
                action_tensor[robot_index] = torch.as_tensor(primitive[sequence_index], dtype=torch.float32, device=raw.device)
                primitive_counts[primitive_name] += 1

            with torch.inference_mode():
                env.step(action_tensor)
                raw.episode_length_buf.zero_()
                center_xy += command_xy * raw.step_dt
                payload_pos[:2] = center_xy
                set_payload_pose(payload_xform, payload_pos)
                write_formation(center_xy, command_xy)

            payload_positions.append(payload_pos.copy())
            root_positions.append(raw._robot.data.root_link_pos_w.detach().cpu().numpy().copy())
            segment_log.append(segment.name)
            if not args.no_video and (step_cursor % capture_every == 0 or step_cursor == total_steps - 1):
                frame = env.render()
                if frame is not None and frame.size:
                    frames.append(frame)
            step_cursor += 1
        if step_cursor >= total_steps:
            break

    payload_positions_np = np.asarray(payload_positions)
    root_positions_np = np.asarray(root_positions)
    payload_displacement = payload_positions_np[-1] - payload_positions_np[0]
    root_displacements = root_positions_np[-1] - root_positions_np[0]
    summary = {
        "type": "four_robot_payload_choreography",
        "physics_contact_payload": False,
        "trajectory_npz": str(args.trajectory_npz),
        "duration_s": float(step_cursor * raw.step_dt),
        "payload_size_m": list(args.payload_size),
        "payload_initial_pos_w_m": payload_positions_np[0].tolist(),
        "payload_final_pos_w_m": payload_positions_np[-1].tolist(),
        "payload_displacement_w_m": payload_displacement.tolist(),
        "robot_slots": [
            {
                "name": slot.name,
                "offset_xy_m": list(slot.offset_xy),
                "yaw_rad": slot.yaw,
                "opposite_side": slot.opposite_side,
            }
            for slot in slots
        ],
        "segments": [
            {"name": segment.name, "duration_s": segment.duration_s, "command_xy_mps": list(segment.command_xy)}
            for segment in segments
        ],
        "primitive_counts": primitive_counts,
        "mean_robot_root_displacement_w_m": np.mean(root_displacements, axis=0).tolist(),
        "max_robot_root_displacement_error_m": float(np.max(np.linalg.norm(root_displacements[:, :2] - payload_displacement[:2], axis=1))),
        "notes": [
            "Hero render uses kinematic robot root formation and kinematic payload pose.",
            "The payload is visually mounted on the front limb area; this is not a fully dynamic box-contact carry.",
        ],
    }

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
    print(f"PAYLOAD_DISP={payload_displacement.tolist()}", flush=True)
    print(f"MAX_FORMATION_ERROR={summary['max_robot_root_displacement_error_m']:.4f}", flush=True)
    env.close()


def main_stage_reference() -> None:
    data = np.load(args.trajectory_npz, allow_pickle=False)

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 0.0
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.viewer.eye = tuple(args.camera_eye)
    env_cfg.viewer.lookat = tuple(args.camera_lookat)
    env_cfg.viewer.origin_type = "world"
    if hasattr(env_cfg, "events"):
        env_cfg.events.physics_material = None

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if not args.no_video else None)
    env.reset()
    raw = env.unwrapped
    raw.episode_length_buf.zero_()

    stage = omni.usd.get_context().get_stage()
    carrier_root = UsdGeom.Xform.Define(stage, "/World/CarrierRobots")
    UsdGeom.Imageable(carrier_root.GetPrim()).MakeVisible()

    env_robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if env_robot_prim.IsValid():
        UsdGeom.Imageable(env_robot_prim).MakeInvisible()

    hidden_env_ids = torch.tensor([0], dtype=torch.int64, device=raw.device)
    hidden_root_pose = torch.tensor([[25.0, 25.0, args.root_z, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=raw.device)
    hidden_root_vel = torch.zeros(1, 6, device=raw.device)

    def hide_env_robot() -> None:
        raw._robot.write_root_pose_to_sim(hidden_root_pose, hidden_env_ids)
        raw._robot.write_root_velocity_to_sim(hidden_root_vel, hidden_env_ids)
        raw.episode_length_buf.zero_()

    hide_env_robot()

    primitive_specs = {
        "forward": PrimitiveSpec(0.72, 1.30),
        "backward": PrimitiveSpec(
            1.30,
            1.88,
            reverse=True,
            swing_lift_height_m=0.03,
            swing_lift_middle_scale=2.0,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.55,
        ),
        "xneg": PrimitiveSpec(
            0.72,
            1.30,
            sideways_remap_scale=1.0,
            sideways_remap_sign=-1.0,
            sideways_remap_y_keep_scale=0.10,
            sideways_remap_max_action_delta=0.65,
            swing_lift_height_m=0.02,
            swing_lift_middle_scale=1.5,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.45,
        ),
        "xpos": PrimitiveSpec(
            0.72,
            1.30,
            sideways_remap_scale=1.5,
            sideways_remap_sign=1.0,
            sideways_remap_y_keep_scale=0.0,
            sideways_remap_max_action_delta=0.85,
            swing_lift_height_m=0.02,
            swing_lift_middle_scale=1.5,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.45,
        ),
    }
    primitives = {name: build_primitive_joint_positions(data, spec, raw.device) for name, spec in primitive_specs.items()}

    slots = [
        RobotSlot("near_left", (-args.robot_pair_half_x, -args.robot_side_y), 0.0, False, 0),
        RobotSlot("near_right", (args.robot_pair_half_x, -args.robot_side_y), 0.0, False, 4),
        RobotSlot("far_left", (-args.robot_pair_half_x, args.robot_side_y), math.pi, True, 4),
        RobotSlot("far_right", (args.robot_pair_half_x, args.robot_side_y), math.pi, True, 0),
    ]
    segments = [
        Segment("settle", args.settle_s, (0.0, 0.0)),
        Segment("forward", args.move_s, (0.0, args.segment_speed)),
        Segment("pause_after_forward", args.pause_s, (0.0, 0.0)),
        Segment("backward", args.move_s, (0.0, -args.segment_speed)),
        Segment("pause_after_backward", args.pause_s, (0.0, 0.0)),
        Segment("left", args.move_s, (-args.segment_speed, 0.0)),
        Segment("pause_after_left", args.pause_s, (0.0, 0.0)),
        Segment("right", args.move_s, (args.segment_speed, 0.0)),
    ]

    def create_robot_reference(slot: RobotSlot) -> dict[str, object]:
        prim_path = f"/World/CarrierRobots/{slot.name}"
        robot_prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        robot_prim.GetReferences().AddReference(str(ROBOT_USD_PATH), "/insectoid_mini_quad")
        xform = UsdGeom.XformCommonAPI(robot_prim)
        xform.SetScale(Gf.Vec3d(1.0, 1.0, 1.0))

        joint_attrs = {}
        for prim in Usd.PrimRange(robot_prim):
            name = prim.GetName()
            if name not in JOINT_NAMES:
                continue
            joint_attrs[name] = {
                "target": prim.GetAttribute("drive:angular:physics:targetPosition"),
                "state": prim.GetAttribute("state:angular:physics:position"),
            }
        missing = sorted(set(JOINT_NAMES) - set(joint_attrs))
        if missing:
            raise RuntimeError(f"{slot.name} reference is missing joint prims: {missing}")
        return {"slot": slot, "prim": robot_prim, "xform": xform, "joint_attrs": joint_attrs}

    robots = [create_robot_reference(slot) for slot in slots]

    payload_pos = np.array((0.0, 0.0, args.payload_z), dtype=np.float32)
    payload_xform = create_payload_cube(tuple(args.payload_size))
    set_payload_pose(payload_xform, payload_pos)

    def set_robot_pose(robot: dict[str, object], center_xy: np.ndarray) -> np.ndarray:
        slot = robot["slot"]
        assert isinstance(slot, RobotSlot)
        pos = np.array(
            (
                center_xy[0] + slot.offset_xy[0],
                center_xy[1] + slot.offset_xy[1],
                args.root_z,
            ),
            dtype=np.float32,
        )
        xform = robot["xform"]
        assert isinstance(xform, UsdGeom.XformCommonAPI)
        xform.SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        xform.SetRotate((0.0, 0.0, math.degrees(slot.yaw)), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        return pos

    def set_robot_joints(robot: dict[str, object], joint_pos_rad: np.ndarray) -> None:
        joint_attrs = robot["joint_attrs"]
        assert isinstance(joint_attrs, dict)
        for joint_name, joint_pos in zip(JOINT_NAMES, joint_pos_rad, strict=True):
            value_deg = float(np.rad2deg(joint_pos))
            attrs = joint_attrs[joint_name]
            for attr_name in ("target", "state"):
                attr = attrs[attr_name]
                if attr is not None and attr.IsValid():
                    attr.Set(value_deg)

    frames = []
    payload_positions = []
    formation_positions = []
    segment_log = []
    primitive_counts = {name: 0 for name in primitives}
    total_steps = min(max(1, int(args.duration / raw.step_dt)), int(sum(segment.duration_s for segment in segments) / raw.step_dt))
    capture_every = max(1, round(1.0 / (args.fps * raw.step_dt)))
    step_cursor = 0
    center_xy = np.zeros(2, dtype=np.float32)
    zero_action = torch.zeros(1, raw.single_action_space.shape[0], device=raw.device)

    for segment in segments:
        segment_steps = int(segment.duration_s / raw.step_dt)
        command_xy = np.asarray(segment.command_xy, dtype=np.float32)
        for local_step in range(segment_steps):
            if step_cursor >= total_steps:
                break

            reference_name = command_to_name(command_xy)
            robot_positions = []
            for robot in robots:
                slot = robot["slot"]
                assert isinstance(slot, RobotSlot)
                primitive_name = mapped_primitive(reference_name, slot.opposite_side)
                primitive = primitives[primitive_name]
                sequence_index = (int(local_step * args.playback_speed) + slot.phase_offset) % primitive.shape[0]
                set_robot_joints(robot, primitive[sequence_index])
                robot_positions.append(set_robot_pose(robot, center_xy))
                primitive_counts[primitive_name] += 1

            payload_pos[:2] = center_xy
            set_payload_pose(payload_xform, payload_pos)

            with torch.inference_mode():
                env.step(zero_action)
                hide_env_robot()

            for robot in robots:
                set_robot_pose(robot, center_xy)
            set_payload_pose(payload_xform, payload_pos)

            payload_positions.append(payload_pos.copy())
            formation_positions.append(np.asarray(robot_positions, dtype=np.float32))
            segment_log.append(segment.name)
            if not args.no_video and (step_cursor % capture_every == 0 or step_cursor == total_steps - 1):
                frame = env.render()
                if frame is not None and frame.size:
                    frames.append(frame)

            center_xy += command_xy * raw.step_dt
            step_cursor += 1
        if step_cursor >= total_steps:
            break

    payload_positions_np = np.asarray(payload_positions)
    formation_positions_np = np.asarray(formation_positions)
    payload_displacement = payload_positions_np[-1] - payload_positions_np[0]
    robot_root_displacements = formation_positions_np[-1] - formation_positions_np[0]
    summary = {
        "type": "four_robot_payload_choreography",
        "render_mode": "single_env_stage_references",
        "physics_contact_payload": False,
        "trajectory_npz": str(args.trajectory_npz),
        "robot_usd": str(ROBOT_USD_PATH),
        "duration_s": float(step_cursor * raw.step_dt),
        "payload_size_m": list(args.payload_size),
        "payload_initial_pos_w_m": payload_positions_np[0].tolist(),
        "payload_final_pos_w_m": payload_positions_np[-1].tolist(),
        "payload_displacement_w_m": payload_displacement.tolist(),
        "robot_slots": [
            {
                "name": slot.name,
                "offset_xy_m": list(slot.offset_xy),
                "yaw_rad": slot.yaw,
                "opposite_side": slot.opposite_side,
                "phase_offset_samples": slot.phase_offset,
            }
            for slot in slots
        ],
        "segments": [
            {"name": segment.name, "duration_s": segment.duration_s, "command_xy_mps": list(segment.command_xy)}
            for segment in segments
        ],
        "primitive_counts": primitive_counts,
        "mean_robot_root_displacement_w_m": np.mean(robot_root_displacements, axis=0).tolist(),
        "max_robot_root_displacement_error_m": float(
            np.max(np.linalg.norm(robot_root_displacements[:, :2] - payload_displacement[:2], axis=1))
        ),
        "notes": [
            "Four robot USD references are animated in one stage from the model-1399 canonical trajectory primitives.",
            "The robots and payload are kinematic visual choreography; this is not a full dynamic payload-contact simulation.",
            "Robots on the far side face the near pair, so forward/backward and lateral primitive mappings are inverted for that side.",
        ],
    }

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
    print(f"PAYLOAD_DISP={payload_displacement.tolist()}", flush=True)
    print(f"MAX_FORMATION_ERROR={summary['max_robot_root_displacement_error_m']:.4f}", flush=True)
    env.close()


def main_world_reference() -> None:
    print("[INFO]: Starting Isaac Core four-robot payload renderer.", flush=True)
    data = np.load(args.trajectory_npz, allow_pickle=False)
    device = torch.device("cpu")

    world = World(physics_dt=0.005, rendering_dt=1.0 / args.fps, backend="torch", device="cpu")
    world.scene.add_default_ground_plane()
    prim_utils.create_prim("/World/Light/Key", "SphereLight", translation=(3.0, -4.0, 5.5), attributes={"inputs:intensity": 4500.0})
    prim_utils.create_prim("/World/Light/Fill", "SphereLight", translation=(-3.0, 2.5, 4.0), attributes={"inputs:intensity": 1200.0})
    prim_utils.create_prim("/World/CarrierRobots", "Xform")

    if not args.no_video:
        set_camera_view(eye=list(args.camera_eye), target=list(args.camera_lookat))
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        camera = rep.create.camera(position=tuple(args.camera_eye), look_at=tuple(args.camera_lookat), focal_length=35.0)
        render_product = rep.create.render_product(camera, resolution=(args.width, args.height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach(render_product)
    else:
        rgb_annotator = None

    primitive_specs = {
        "forward": PrimitiveSpec(0.72, 1.30),
        "backward": PrimitiveSpec(
            1.30,
            1.88,
            reverse=True,
            swing_lift_height_m=0.03,
            swing_lift_middle_scale=2.0,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.55,
        ),
        "xneg": PrimitiveSpec(
            0.72,
            1.30,
            sideways_remap_scale=1.0,
            sideways_remap_sign=-1.0,
            sideways_remap_y_keep_scale=0.10,
            sideways_remap_max_action_delta=0.65,
            swing_lift_height_m=0.02,
            swing_lift_middle_scale=1.5,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.45,
        ),
        "xpos": PrimitiveSpec(
            0.72,
            1.30,
            sideways_remap_scale=1.5,
            sideways_remap_sign=1.0,
            sideways_remap_y_keep_scale=0.0,
            sideways_remap_max_action_delta=0.85,
            swing_lift_height_m=0.02,
            swing_lift_middle_scale=1.5,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.45,
        ),
    }
    primitives = {name: build_primitive_joint_positions(data, spec, device) for name, spec in primitive_specs.items()}

    slots = [
        RobotSlot("near_left", (-args.robot_pair_half_x, -args.robot_side_y), 0.0, False, 0),
        RobotSlot("near_right", (args.robot_pair_half_x, -args.robot_side_y), 0.0, False, 4),
        RobotSlot("far_left", (-args.robot_pair_half_x, args.robot_side_y), math.pi, True, 4),
        RobotSlot("far_right", (args.robot_pair_half_x, args.robot_side_y), math.pi, True, 0),
    ]
    robot_paths = [f"/World/CarrierRobots/{slot.name}" for slot in slots]
    for slot, robot_path in zip(slots, robot_paths, strict=True):
        prim_utils.create_prim(
            robot_path,
            usd_path=str(ROBOT_USD_PATH),
            translation=(slot.offset_xy[0], slot.offset_xy[1], args.root_z),
            orientation=yaw_to_quat_wxyz(slot.yaw),
        )

    robots = world.scene.add(CoreArticulation(robot_paths, name="carrier_robot_view"))

    payload_pos = np.array((0.0, 0.0, args.payload_z), dtype=np.float32)
    payload_xform = create_payload_cube(tuple(args.payload_size))
    set_payload_pose(payload_xform, payload_pos)

    world.reset()
    print("[INFO]: World/articulation setup complete.", flush=True)

    missing_joints = sorted(set(JOINT_NAMES) - set(robots.dof_names))
    if missing_joints:
        raise RuntimeError(f"Carrier articulation is missing expected joints: {missing_joints}. Available: {robots.dof_names}")

    segments = [
        Segment("settle", args.settle_s, (0.0, 0.0)),
        Segment("forward", args.move_s, (0.0, args.segment_speed)),
        Segment("pause_after_forward", args.pause_s, (0.0, 0.0)),
        Segment("backward", args.move_s, (0.0, -args.segment_speed)),
        Segment("pause_after_backward", args.pause_s, (0.0, 0.0)),
        Segment("left", args.move_s, (-args.segment_speed, 0.0)),
        Segment("pause_after_left", args.pause_s, (0.0, 0.0)),
        Segment("right", args.move_s, (args.segment_speed, 0.0)),
    ]

    frame_dt = 1.0 / args.fps
    total_frames = min(
        max(1, int(args.duration * args.fps)),
        int(sum(segment.duration_s for segment in segments) * args.fps),
    )
    center_xy = np.zeros(2, dtype=np.float32)
    frames = []
    payload_positions = []
    formation_positions = []
    segment_log = []
    primitive_counts = {name: 0 for name in primitives}

    def robot_pose_batch(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.zeros((len(slots), 3), dtype=np.float32)
        orientations = np.zeros((len(slots), 4), dtype=np.float32)
        for robot_index, slot in enumerate(slots):
            positions[robot_index] = (
                center[0] + slot.offset_xy[0],
                center[1] + slot.offset_xy[1],
                args.root_z,
            )
            orientations[robot_index] = yaw_to_quat_wxyz(slot.yaw)
        return positions, orientations

    def set_robot_states(center: np.ndarray, q_batch: np.ndarray) -> np.ndarray:
        positions, orientations = robot_pose_batch(center)
        robots.set_world_poses(positions=positions, orientations=orientations)
        robots.set_joint_positions(q_batch, joint_names=list(JOINT_NAMES))
        robots.set_joint_position_targets(q_batch, joint_names=list(JOINT_NAMES))
        robots.set_joint_velocities(np.zeros_like(q_batch), joint_names=list(JOINT_NAMES))
        return positions

    default_reference = primitives["forward"][0]
    default_q_batch = np.tile(default_reference[None, :], (len(slots), 1)).astype(np.float32)
    set_robot_states(center_xy, default_q_batch)
    for _ in range(8):
        world.step(render=not args.no_video)

    step_cursor = 0
    for segment in segments:
        segment_frames = int(segment.duration_s * args.fps)
        command_xy = np.asarray(segment.command_xy, dtype=np.float32)
        for local_frame in range(segment_frames):
            if step_cursor >= total_frames:
                break

            reference_name = command_to_name(command_xy)
            q_batch = np.zeros((len(slots), len(JOINT_NAMES)), dtype=np.float32)
            for robot_index, slot in enumerate(slots):
                primitive_name = mapped_primitive(reference_name, slot.opposite_side)
                primitive = primitives[primitive_name]
                sequence_index = (int(local_frame * args.playback_speed) + slot.phase_offset) % primitive.shape[0]
                q_batch[robot_index] = primitive[sequence_index]
                primitive_counts[primitive_name] += 1

            robot_positions = set_robot_states(center_xy, q_batch)
            payload_pos[:2] = center_xy
            set_payload_pose(payload_xform, payload_pos)

            world.step(render=not args.no_video)
            robot_positions = set_robot_states(center_xy, q_batch)
            set_payload_pose(payload_xform, payload_pos)
            if not args.no_video:
                world.render()

            payload_positions.append(payload_pos.copy())
            formation_positions.append(robot_positions.copy())
            segment_log.append(segment.name)

            if not args.no_video and rgb_annotator is not None:
                rgb = rgb_annotator.get_data()
                if rgb is not None and getattr(rgb, "size", 0):
                    rgb = np.asarray(rgb)
                    if rgb.ndim == 3 and rgb.shape[-1] == 4:
                        rgb = rgb[..., :3]
                    if rgb.dtype != np.uint8:
                        rgb = np.clip(rgb, 0, 255 if np.max(rgb) > 1.0 else 1.0)
                        rgb = (rgb * 255.0).astype(np.uint8) if np.max(rgb) <= 1.0 else rgb.astype(np.uint8)
                    frames.append(np.ascontiguousarray(rgb))

            center_xy += command_xy * frame_dt
            step_cursor += 1
        if step_cursor >= total_frames:
            break

    payload_positions_np = np.asarray(payload_positions)
    formation_positions_np = np.asarray(formation_positions)
    payload_displacement = payload_positions_np[-1] - payload_positions_np[0]
    robot_root_displacements = formation_positions_np[-1] - formation_positions_np[0]
    summary = {
        "type": "four_robot_payload_choreography",
        "render_mode": "isaac_core_world_four_articulations",
        "physics_contact_payload": False,
        "trajectory_npz": str(args.trajectory_npz),
        "robot_usd": str(ROBOT_USD_PATH),
        "duration_s": float(step_cursor * frame_dt),
        "fps": args.fps,
        "payload_size_m": list(args.payload_size),
        "payload_initial_pos_w_m": payload_positions_np[0].tolist(),
        "payload_final_pos_w_m": payload_positions_np[-1].tolist(),
        "payload_displacement_w_m": payload_displacement.tolist(),
        "robot_slots": [
            {
                "name": slot.name,
                "prim_path": robot_path,
                "offset_xy_m": list(slot.offset_xy),
                "yaw_rad": slot.yaw,
                "opposite_side": slot.opposite_side,
                "phase_offset_samples": slot.phase_offset,
            }
            for slot, robot_path in zip(slots, robot_paths, strict=True)
        ],
        "segments": [
            {"name": segment.name, "duration_s": segment.duration_s, "command_xy_mps": list(segment.command_xy)}
            for segment in segments
        ],
        "primitive_counts": primitive_counts,
        "dof_names": list(robots.dof_names),
        "mean_robot_root_displacement_w_m": np.mean(robot_root_displacements, axis=0).tolist(),
        "max_robot_root_displacement_error_m": float(
            np.max(np.linalg.norm(robot_root_displacements[:, :2] - payload_displacement[:2], axis=1))
        ),
        "notes": [
            "Four real robot USD articulations are posed by name from model-1399 canonical trajectory primitives.",
            "The payload is a kinematic visual cube carried in formation; this is not a full dynamic payload-contact simulation.",
            "Robots on the far side face the near pair, so forward/backward and lateral primitive mappings are inverted for that side.",
        ],
    }

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
    print(f"PAYLOAD_DISP={payload_displacement.tolist()}", flush=True)
    print(f"MAX_FORMATION_ERROR={summary['max_robot_root_displacement_error_m']:.4f}", flush=True)


def main_lab_reference() -> None:
    print("[INFO]: Starting IsaacLab articulation four-robot payload renderer.", flush=True)
    data = np.load(args.trajectory_npz, allow_pickle=False)
    device = torch.device("cpu")

    sim_dt = args.physics_dt if args.dynamic_roots else min(1.0 / args.fps, 0.02)
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, render_interval=1, device=args.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(args.camera_eye), target=list(args.camera_lookat))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3200.0, color=(0.78, 0.78, 0.76))
    light_cfg.func("/World/Light", light_cfg)
    sim_utils.create_prim("/World/CarrierRobots", "Xform")

    primitive_specs = {
        "forward": PrimitiveSpec(0.72, 1.30),
        "backward": PrimitiveSpec(
            1.30,
            1.88,
            reverse=True,
            swing_lift_height_m=0.03,
            swing_lift_middle_scale=2.0,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.55,
        ),
        "xneg": PrimitiveSpec(
            0.72,
            1.30,
            sideways_remap_scale=1.0,
            sideways_remap_sign=-1.0,
            sideways_remap_y_keep_scale=0.10,
            sideways_remap_max_action_delta=0.65,
            swing_lift_height_m=0.02,
            swing_lift_middle_scale=1.5,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.45,
        ),
        "xpos": PrimitiveSpec(
            0.72,
            1.30,
            sideways_remap_scale=1.5,
            sideways_remap_sign=1.0,
            sideways_remap_y_keep_scale=0.0,
            sideways_remap_max_action_delta=0.85,
            swing_lift_height_m=0.02,
            swing_lift_middle_scale=1.5,
            swing_lift_dilate_steps=1,
            swing_lift_max_action_delta=0.45,
        ),
    }
    primitives = {name: build_primitive_joint_positions(data, spec, device) for name, spec in primitive_specs.items()}

    slots = [
        RobotSlot("near_left", (-args.robot_pair_half_x, -args.robot_side_y), 0.0, False, 0),
        RobotSlot("near_right", (args.robot_pair_half_x, -args.robot_side_y), 0.0, False, 4),
        RobotSlot("far_left", (-args.robot_pair_half_x, args.robot_side_y), math.pi, True, 4),
        RobotSlot("far_right", (args.robot_pair_half_x, args.robot_side_y), math.pi, True, 0),
    ]
    origin_paths = [f"/World/CarrierRobots/slot_{index:02d}_{slot.name}" for index, slot in enumerate(slots)]
    robot_paths = [f"{origin_path}/Robot" for origin_path in origin_paths]
    for origin_path in origin_paths:
        sim_utils.create_prim(origin_path, "Xform")

    robot_cfg = INSECTOID_MINI_QUAD_CFG.replace(prim_path="/World/CarrierRobots/slot_.*/Robot")
    robots = LabArticulation(cfg=robot_cfg)

    payload_pos = np.array((0.0, 0.0, args.payload_z), dtype=np.float32)
    payload_xform = create_payload_cube(tuple(args.payload_size))
    set_payload_pose(payload_xform, payload_pos)

    if not args.no_video:
        if rep is None:
            raise RuntimeError("Replicator is not available; rerun without --no-video only when camera extensions are loaded.")
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        camera = rep.create.camera(position=tuple(args.camera_eye), look_at=tuple(args.camera_lookat), focal_length=35.0)
        render_product = rep.create.render_product(camera, resolution=(args.width, args.height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach(render_product)
    else:
        rgb_annotator = None

    sim.reset()
    print("[INFO]: IsaacLab articulation setup complete.", flush=True)

    if robots.data.default_joint_pos.shape[0] != len(slots):
        raise RuntimeError(
            f"Expected {len(slots)} carrier robots, got {robots.data.default_joint_pos.shape[0]} from prim path {robot_cfg.prim_path}."
        )

    joint_ids, resolved_joint_names = robots.find_joints(list(JOINT_NAMES), preserve_order=True)
    if tuple(resolved_joint_names) != JOINT_NAMES:
        raise RuntimeError(f"Unexpected joint resolution order {resolved_joint_names}; expected {JOINT_NAMES}.")
    joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=sim.device)
    base_body_ids_tensor = None
    if args.closed_loop_correction:
        base_body_ids, base_body_names = robots.find_bodies("base_link", preserve_order=True)
        if base_body_names != ["base_link"]:
            raise RuntimeError(f"Unexpected base body resolution {base_body_names}; expected ['base_link'].")
        base_body_ids_tensor = torch.tensor(base_body_ids, dtype=torch.long, device=sim.device)

    segments = [
        Segment("settle", args.settle_s, (0.0, 0.0)),
        Segment("forward", args.move_s, (0.0, args.segment_speed)),
        Segment("pause_after_forward", args.pause_s, (0.0, 0.0)),
        Segment("backward", args.move_s, (0.0, -args.segment_speed)),
        Segment("pause_after_backward", args.pause_s, (0.0, 0.0)),
        Segment("left", args.move_s, (-args.segment_speed, 0.0)),
        Segment("pause_after_left", args.pause_s, (0.0, 0.0)),
        Segment("right", args.move_s, (args.segment_speed, 0.0)),
    ]

    frame_dt = 1.0 / args.fps
    physics_steps_per_frame = max(1, round(frame_dt / sim.get_physics_dt())) if args.dynamic_roots else 1
    total_frames = min(
        max(1, int(args.duration * args.fps)),
        int(sum(segment.duration_s for segment in segments) * args.fps),
    )
    center_xy = np.zeros(2, dtype=np.float32)
    frames = []
    payload_positions = []
    formation_positions = []
    segment_log = []
    primitive_counts = {name: 0 for name in primitives}
    correction_stats = {
        "samples": 0,
        "mean_slot_error_m": 0.0,
        "max_slot_error_m": 0.0,
        "mean_center_error_m": 0.0,
        "max_center_error_m": 0.0,
        "mean_yaw_error_rad": 0.0,
        "max_yaw_error_rad": 0.0,
        "max_forward_blend": 0.0,
        "max_side_blend": 0.0,
        "max_total_blend": 0.0,
        "max_force_n": 0.0,
        "max_torque_nm": 0.0,
    }

    def robot_pose_batch(center: np.ndarray) -> torch.Tensor:
        root_pose = torch.zeros((len(slots), 7), dtype=torch.float32, device=sim.device)
        for robot_index, slot in enumerate(slots):
            root_pose[robot_index, 0] = float(center[0] + slot.offset_xy[0])
            root_pose[robot_index, 1] = float(center[1] + slot.offset_xy[1])
            root_pose[robot_index, 2] = args.root_z
            root_pose[robot_index, 3:7] = torch.tensor(yaw_to_quat_wxyz(slot.yaw), dtype=torch.float32, device=sim.device)
        return root_pose

    def write_robot_states(
        center: np.ndarray,
        q_batch_np: np.ndarray,
        write_roots: bool,
        force_np: np.ndarray | None = None,
        torque_np: np.ndarray | None = None,
    ) -> torch.Tensor:
        root_pose = robot_pose_batch(center)
        q_batch = torch.as_tensor(q_batch_np, dtype=torch.float32, device=sim.device)
        zero_joint_vel = torch.zeros_like(q_batch)
        if write_roots:
            robots.write_root_pose_to_sim(root_pose)
            robots.write_root_velocity_to_sim(torch.zeros((len(slots), 6), dtype=torch.float32, device=sim.device))
            robots.write_joint_state_to_sim(q_batch, zero_joint_vel, joint_ids=joint_ids_tensor)
        robots.set_joint_position_target(q_batch, joint_ids=joint_ids_tensor)
        if args.closed_loop_correction and args.dynamic_roots and base_body_ids_tensor is not None:
            robots.permanent_wrench_composer.reset()
            if force_np is not None or torque_np is not None:
                forces = torch.as_tensor(
                    np.zeros((len(slots), 1, 3), dtype=np.float32) if force_np is None else force_np[:, None, :],
                    dtype=torch.float32,
                    device=sim.device,
                )
                torques = torch.as_tensor(
                    np.zeros((len(slots), 1, 3), dtype=np.float32) if torque_np is None else torque_np[:, None, :],
                    dtype=torch.float32,
                    device=sim.device,
                )
                robots.permanent_wrench_composer.set_forces_and_torques(
                    forces=forces,
                    torques=torques,
                    body_ids=base_body_ids_tensor,
                    is_global=True,
                )
        robots.write_data_to_sim()
        if args.dynamic_roots and not write_roots:
            return robots.data.root_link_pos_w.detach().cpu()
        return root_pose[:, :3].detach().cpu()

    def primitive_sample(name: str, local_frame: int, slot: RobotSlot) -> np.ndarray:
        primitive = primitives[name]
        sequence_index = (int(local_frame * args.playback_speed) + slot.phase_offset) % primitive.shape[0]
        return primitive[sequence_index]

    def gait_sample(name: str, local_frame: int, slot: RobotSlot, motion_weight: float) -> np.ndarray:
        sample = primitive_sample(name, local_frame, slot)
        if motion_weight >= 1.0:
            return sample.copy()
        return (static_q + motion_weight * (sample - static_q)).astype(np.float32)

    def blend_primitives(base_q: np.ndarray, components: list[tuple[float, np.ndarray]]) -> np.ndarray:
        active_components = [(float(weight), q) for weight, q in components if weight > 1.0e-6]
        total_weight = sum(weight for weight, _ in active_components)
        if total_weight > args.correction_max_total_blend:
            scale = args.correction_max_total_blend / total_weight
            active_components = [(weight * scale, q) for weight, q in active_components]
            total_weight = args.correction_max_total_blend
        blended = (1.0 - total_weight) * base_q
        for weight, q in active_components:
            blended += weight * q
        return blended.astype(np.float32)

    def current_root_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        root_pos = robots.data.root_link_pos_w.detach().cpu().numpy()
        root_quat = robots.data.root_link_quat_w.detach().cpu().numpy()
        root_vel = robots.data.root_link_lin_vel_w.detach().cpu().numpy()
        root_ang_vel = robots.data.root_link_ang_vel_w.detach().cpu().numpy()
        return root_pos, root_quat, root_vel, root_ang_vel

    def correction_inputs(center: np.ndarray) -> dict[str, np.ndarray]:
        root_pos, root_quat, root_vel, root_ang_vel = current_root_arrays()
        yaw = quat_wxyz_to_yaw_np(root_quat)
        actual_center = root_pos[:, :2].mean(axis=0)
        center_error = center - actual_center
        desired_xy = np.zeros((len(slots), 2), dtype=np.float32)
        slot_error = np.zeros((len(slots), 2), dtype=np.float32)
        yaw_error = np.zeros(len(slots), dtype=np.float32)
        for robot_index, slot in enumerate(slots):
            desired_xy[robot_index] = (
                actual_center
                + np.asarray(slot.offset_xy, dtype=np.float32)
                + args.center_correction_gain * center_error
            )
            slot_error[robot_index] = desired_xy[robot_index] - root_pos[robot_index, :2]
            yaw_error[robot_index] = float(wrap_to_pi(slot.yaw - yaw[robot_index]))
        return {
            "root_pos": root_pos,
            "root_vel": root_vel,
            "root_ang_vel": root_ang_vel,
            "actual_center": actual_center,
            "center_error": center_error,
            "slot_error": slot_error,
            "yaw": yaw,
            "yaw_error": yaw_error,
        }

    def correction_wrenches(inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        forces = np.zeros((len(slots), 3), dtype=np.float32)
        torques = np.zeros((len(slots), 3), dtype=np.float32)
        for robot_index in range(len(slots)):
            force_xy = (
                args.formation_force_kp * inputs["slot_error"][robot_index]
                - args.formation_force_kd * inputs["root_vel"][robot_index, :2]
            )
            force = torch.as_tensor(force_xy[None, :], dtype=torch.float32)
            force_xy = clamp_vector_norm(force, args.formation_max_force).numpy()[0]
            forces[robot_index, :2] = force_xy

            torque_z = (
                args.yaw_torque_kp * inputs["yaw_error"][robot_index]
                - args.yaw_torque_kd * inputs["root_ang_vel"][robot_index, 2]
            )
            torques[robot_index, 2] = float(np.clip(torque_z, -args.yaw_max_torque, args.yaw_max_torque))
        return forces, torques

    def update_correction_stats(
        inputs: dict[str, np.ndarray],
        forward_blends: np.ndarray,
        side_blends: np.ndarray,
        total_blends: np.ndarray,
        forces: np.ndarray | None,
        torques: np.ndarray | None,
    ) -> None:
        slot_error_norm = np.linalg.norm(inputs["slot_error"], axis=1)
        center_error_norm = float(np.linalg.norm(inputs["center_error"]))
        yaw_error_abs = np.abs(inputs["yaw_error"])
        sample_count = correction_stats["samples"]
        robot_count = len(slots)
        correction_stats["samples"] += robot_count
        correction_stats["mean_slot_error_m"] += float(np.sum(slot_error_norm))
        correction_stats["mean_center_error_m"] += center_error_norm * robot_count
        correction_stats["mean_yaw_error_rad"] += float(np.sum(yaw_error_abs))
        correction_stats["max_slot_error_m"] = max(correction_stats["max_slot_error_m"], float(np.max(slot_error_norm)))
        correction_stats["max_center_error_m"] = max(correction_stats["max_center_error_m"], center_error_norm)
        correction_stats["max_yaw_error_rad"] = max(correction_stats["max_yaw_error_rad"], float(np.max(yaw_error_abs)))
        correction_stats["max_forward_blend"] = max(correction_stats["max_forward_blend"], float(np.max(np.abs(forward_blends))))
        correction_stats["max_side_blend"] = max(correction_stats["max_side_blend"], float(np.max(np.abs(side_blends))))
        correction_stats["max_total_blend"] = max(correction_stats["max_total_blend"], float(np.max(total_blends)))
        if forces is not None:
            correction_stats["max_force_n"] = max(correction_stats["max_force_n"], float(np.max(np.linalg.norm(forces, axis=1))))
        if torques is not None:
            correction_stats["max_torque_nm"] = max(correction_stats["max_torque_nm"], float(np.max(np.linalg.norm(torques, axis=1))))

    def build_q_batch(
        reference_name: str,
        local_frame: int,
        inputs: dict[str, np.ndarray] | None,
        motion_weight: float = 1.0,
        count_primitives: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q_batch = np.zeros((len(slots), len(JOINT_NAMES)), dtype=np.float32)
        forward_blends = np.zeros(len(slots), dtype=np.float32)
        side_blends = np.zeros(len(slots), dtype=np.float32)
        total_blends = np.zeros(len(slots), dtype=np.float32)
        for robot_index, slot in enumerate(slots):
            if reference_name == "hold":
                primitive_name = "hold"
                base_q = static_q.copy()
            else:
                primitive_name = mapped_primitive(reference_name, slot.opposite_side)
                base_q = gait_sample(primitive_name, local_frame, slot, motion_weight)
            if count_primitives:
                primitive_counts.setdefault(primitive_name, 0)
                primitive_counts[primitive_name] += 1

            if args.closed_loop_correction and args.dynamic_roots and inputs is not None:
                right_axis, forward_axis = slot_axes(slot.yaw)
                err_xy = inputs["slot_error"][robot_index]
                err_forward = float(np.dot(err_xy, forward_axis))
                err_right = float(np.dot(err_xy, right_axis))
                forward_blend = float(
                    np.clip(
                        args.slot_correction_gain * err_forward,
                        -args.correction_max_blend,
                        args.correction_max_blend,
                    )
                )
                side_blend = float(
                    np.clip(
                        args.slot_correction_gain * err_right,
                        -args.correction_max_blend,
                        args.correction_max_blend,
                    )
                )
                components: list[tuple[float, np.ndarray]] = []
                if abs(forward_blend) > 1.0e-6:
                    corrective_name = "forward" if forward_blend > 0.0 else "backward"
                    components.append((abs(forward_blend), gait_sample(corrective_name, local_frame, slot, motion_weight)))
                if abs(side_blend) > 1.0e-6:
                    corrective_name = "xpos" if side_blend > 0.0 else "xneg"
                    components.append((abs(side_blend), gait_sample(corrective_name, local_frame, slot, motion_weight)))
                q = blend_primitives(base_q, components)
                yaw_delta = float(
                    np.clip(
                        args.yaw_coxa_gain * inputs["yaw_error"][robot_index],
                        -math.radians(args.yaw_coxa_max_deg),
                        math.radians(args.yaw_coxa_max_deg),
                    )
                )
                q[0] += yaw_delta
                q[2] += yaw_delta
                q[1] -= yaw_delta
                q[3] -= yaw_delta
                q_batch[robot_index] = q
                forward_blends[robot_index] = forward_blend
                side_blends[robot_index] = side_blend
                total_blends[robot_index] = min(abs(forward_blend) + abs(side_blend), args.correction_max_total_blend)
            else:
                q_batch[robot_index] = base_q
        return q_batch, forward_blends, side_blends, total_blends

    static_q = data["default_joint_pos"].astype(np.float32)
    default_q = np.tile(static_q[None, :], (len(slots), 1)).astype(np.float32)
    write_robot_states(center_xy, default_q, write_roots=True)
    robots.reset()
    for warmup_index in range(5 if args.dynamic_roots else 3):
        write_robot_states(center_xy, default_q, write_roots=not args.dynamic_roots)
        sim.step(render=not args.no_video and warmup_index == 4)
        robots.update(sim.get_physics_dt())

    step_cursor = 0
    for segment in segments:
        segment_frames = int(segment.duration_s * args.fps)
        command_xy = np.asarray(segment.command_xy, dtype=np.float32)
        for local_frame in range(segment_frames):
            if step_cursor >= total_frames:
                break

            reference_name = command_to_name(command_xy)
            motion_weight = (
                segment_motion_weight(local_frame, segment_frames, frame_dt, args.gait_ramp_s)
                if reference_name != "hold"
                else 0.0
            )
            inputs = correction_inputs(center_xy) if args.closed_loop_correction and args.dynamic_roots else None
            q_batch, forward_blends, side_blends, total_blends = build_q_batch(
                reference_name,
                local_frame,
                inputs,
                motion_weight=motion_weight,
            )
            force_np, torque_np = (correction_wrenches(inputs) if inputs is not None else (None, None))
            robot_positions = write_robot_states(
                center_xy,
                q_batch,
                write_roots=not args.dynamic_roots,
                force_np=force_np,
                torque_np=torque_np,
            )
            payload_pos[:2] = robot_positions.numpy()[:, :2].mean(axis=0) if args.dynamic_roots else center_xy
            set_payload_pose(payload_xform, payload_pos)

            for physics_step_index in range(physics_steps_per_frame):
                sim.step(render=not args.no_video and physics_step_index == physics_steps_per_frame - 1)
                robots.update(sim.get_physics_dt())
                if args.closed_loop_correction and args.dynamic_roots:
                    inputs = correction_inputs(center_xy)
                    q_batch, forward_blends, side_blends, total_blends = build_q_batch(
                        reference_name,
                        local_frame,
                        inputs,
                        motion_weight=motion_weight,
                        count_primitives=False,
                    )
                    force_np, torque_np = correction_wrenches(inputs)
                robot_positions = write_robot_states(
                    center_xy,
                    q_batch,
                    write_roots=not args.dynamic_roots,
                    force_np=force_np,
                    torque_np=torque_np,
                )
                if args.dynamic_roots:
                    payload_pos[:2] = robot_positions.numpy()[:, :2].mean(axis=0)
                set_payload_pose(payload_xform, payload_pos)

            if inputs is not None:
                update_correction_stats(inputs, forward_blends, side_blends, total_blends, force_np, torque_np)

            payload_positions.append(payload_pos.copy())
            formation_positions.append(robot_positions.numpy().copy())
            segment_log.append(segment.name)

            if not args.no_video and rgb_annotator is not None:
                rgb = rgb_annotator.get_data()
                if rgb is not None and getattr(rgb, "size", 0):
                    rgb = np.asarray(rgb)
                    if rgb.ndim == 3 and rgb.shape[-1] == 4:
                        rgb = rgb[..., :3]
                    if rgb.dtype != np.uint8:
                        max_value = float(np.max(rgb)) if rgb.size else 1.0
                        rgb = np.clip(rgb, 0, 255 if max_value > 1.0 else 1.0)
                        rgb = (rgb * 255.0).astype(np.uint8) if max_value <= 1.0 else rgb.astype(np.uint8)
                    frames.append(np.ascontiguousarray(rgb))

            center_xy += command_xy * motion_weight * frame_dt
            step_cursor += 1
        if step_cursor >= total_frames:
            break

    payload_positions_np = np.asarray(payload_positions)
    formation_positions_np = np.asarray(formation_positions)
    payload_displacement = payload_positions_np[-1] - payload_positions_np[0]
    robot_root_displacements = formation_positions_np[-1] - formation_positions_np[0]
    correction_summary = dict(correction_stats)
    if correction_summary["samples"] > 0:
        samples = float(correction_summary["samples"])
        correction_summary["mean_slot_error_m"] /= samples
        correction_summary["mean_center_error_m"] /= samples
        correction_summary["mean_yaw_error_rad"] /= samples
    prim_paths = getattr(robots.root_physx_view, "prim_paths", robot_paths)
    summary = {
        "type": "four_robot_payload_choreography",
        "render_mode": (
            "isaaclab_simulation_context_dynamic_floor_contacts"
            if args.dynamic_roots
            else "isaaclab_simulation_context_kinematic_formation"
        ),
        "dynamic_roots": bool(args.dynamic_roots),
        "physics_floor_motion": bool(args.dynamic_roots),
        "closed_loop_correction": bool(args.closed_loop_correction),
        "physics_contact_payload": False,
        "trajectory_npz": str(args.trajectory_npz),
        "robot_usd": str(ROBOT_USD_PATH),
        "robot_cfg_prim_path": robot_cfg.prim_path,
        "resolved_robot_prim_paths": list(prim_paths),
        "duration_s": float(step_cursor * frame_dt),
        "fps": args.fps,
        "playback_speed": args.playback_speed,
        "gait_ramp_s": args.gait_ramp_s,
        "physics_dt": float(sim.get_physics_dt()),
        "physics_steps_per_frame": int(physics_steps_per_frame),
        "payload_follow_mode": "mean_simulated_robot_root_xy" if args.dynamic_roots else "scripted_choreography_center_xy",
        "payload_size_m": list(args.payload_size),
        "payload_initial_pos_w_m": payload_positions_np[0].tolist(),
        "payload_final_pos_w_m": payload_positions_np[-1].tolist(),
        "payload_displacement_w_m": payload_displacement.tolist(),
        "robot_slots": [
            {
                "name": slot.name,
                "origin_path": origin_path,
                "robot_path": robot_path,
                "offset_xy_m": list(slot.offset_xy),
                "yaw_rad": slot.yaw,
                "opposite_side": slot.opposite_side,
                "phase_offset_samples": slot.phase_offset,
            }
            for slot, origin_path, robot_path in zip(slots, origin_paths, robot_paths, strict=True)
        ],
        "segments": [
            {"name": segment.name, "duration_s": segment.duration_s, "command_xy_mps": list(segment.command_xy)}
            for segment in segments
        ],
        "primitive_counts": primitive_counts,
        "correction_gains": {
            "slot_correction_gain": args.slot_correction_gain,
            "center_correction_gain": args.center_correction_gain,
            "correction_max_blend": args.correction_max_blend,
            "correction_max_total_blend": args.correction_max_total_blend,
            "yaw_coxa_gain": args.yaw_coxa_gain,
            "yaw_coxa_max_deg": args.yaw_coxa_max_deg,
            "formation_force_kp": args.formation_force_kp,
            "formation_force_kd": args.formation_force_kd,
            "formation_max_force": args.formation_max_force,
            "yaw_torque_kp": args.yaw_torque_kp,
            "yaw_torque_kd": args.yaw_torque_kd,
            "yaw_max_torque": args.yaw_max_torque,
        },
        "correction_stats": correction_summary,
        "joint_ids": joint_ids,
        "joint_names": resolved_joint_names,
        "mean_robot_root_displacement_w_m": np.mean(robot_root_displacements, axis=0).tolist(),
        "max_robot_root_displacement_error_m": float(
            np.max(np.linalg.norm(robot_root_displacements[:, :2] - payload_displacement[:2], axis=1))
        ),
        "notes": [
            "Four robot articulations use the same INSECTOID_MINI_QUAD_CFG as the training/play environments.",
            (
                "Dynamic-root mode initializes each root once, then drives model-1399 joint position targets through PhysX floor contact."
                if args.dynamic_roots
                else "Kinematic mode writes root poses and model-1399 primitive joint states directly each frame for a controlled choreography."
            ),
            "The payload is still a kinematic visual cube; it follows the mean robot root pose in dynamic-root mode and is not a full dynamic payload-contact simulation.",
            "Robots on the far side face the near pair, so forward/backward and lateral primitive mappings are inverted for that side.",
            (
                f"Moving segments use a {args.gait_ramp_s:.3f}s shared root/gait ramp; hold segments command the static default joint pose."
                if args.gait_ramp_s > 0.0
                else "Hold segments command the static default joint pose; moving segments use no additional gait/root ramp."
            ),
            (
                "Closed-loop correction blends small forward/backward/sideways primitive components from slot error and applies low-gain physical base forces plus yaw torque."
                if args.closed_loop_correction
                else "Closed-loop primitive blending and physical formation/yaw stabilization are disabled."
            ),
        ],
    }

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
    print(f"PAYLOAD_DISP={payload_displacement.tolist()}", flush=True)
    print(f"MAX_FORMATION_ERROR={summary['max_robot_root_displacement_error_m']:.4f}", flush=True)


if __name__ == "__main__":
    try:
        main_lab_reference()
    except BaseException as exc:
        print(f"[ERROR]: Renderer aborted with {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        try:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        except TypeError:
            simulation_app.close()
