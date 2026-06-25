#!/usr/bin/env python3
"""Render a scripted multi-agent material relay / handoff primitive."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD_PATH = REPO_ROOT / "URDF_description/usd/insectoid_mini_quad.usd"
TRAJECTORY_PATH = REPO_ROOT / "data/trajectories/reversed_trajectory/model_1399_forward_canonical_capture.trajectory.npz"

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
LEG_LOWER_LIMITS = (-1.047198, -1.221730, 0.0)
LEG_UPPER_LIMITS = (1.047198, 0.0, 2.617994)

STATE_WINDOWS = (
    ("INIT", 1.0),
    ("TEAM_A_CARRY_TO_HANDOFF", 5.8),
    ("TEAM_A_STABILIZE", 1.6),
    ("TEAM_B_SLOT_INTO_GAPS", 4.6),
    ("OVERLAP_SUPPORT", 0.24),
    ("VERIFY_HANDOFF", 0.16),
    ("TRANSFER_LOAD", 0.32),
    ("TEAM_A_RELEASE_AND_EXIT", 3.2),
    ("TEAM_B_CARRY_TO_DESTINATION", 5.8),
    ("COMPLETE", 1.8),
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trajectory-npz", type=Path, default=TRAJECTORY_PATH)
parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/renders/material_relay_handoff_v1.mp4")
parser.add_argument("--summary-output", type=Path, default=None)
parser.add_argument("--duration", type=float, default=sum(duration for _, duration in STATE_WINDOWS))
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--cat-cycle-distance", type=float, default=0.34, help="Root travel distance per full hardcoded cat-style diagonal gait cycle.")
parser.add_argument("--cat-stance-line-scale", type=float, default=1.0, help="Scale for stance foot line length relative to cycle_distance * stance_duty.")
parser.add_argument("--cat-foot-clearance", type=float, default=0.050, help="Swing foot clearance above the stance line in meters.")
parser.add_argument("--cat-stance-duty", type=float, default=0.62, help="Fraction of each leg cycle spent planted and driving.")
parser.add_argument("--cat-speed-blend-mps", type=float, default=0.08)
parser.add_argument("--cat-activity-rise-tau", type=float, default=0.12, help="Seconds for gait activity to fade in when a robot starts moving.")
parser.add_argument("--cat-activity-fall-tau", type=float, default=0.45, help="Seconds for gait activity to fade out when a robot stops.")
parser.add_argument("--cat-ik-iterations", type=int, default=10)
parser.add_argument("--cat-ik-damping", type=float, default=0.015)
parser.add_argument("--cat-ik-gain", type=float, default=0.9)
parser.add_argument("--cat-ik-max-joint-step-deg", type=float, default=9.0)
parser.add_argument("--root-z", type=float, default=0.15)
parser.add_argument("--beam-z", type=float, default=0.36)
parser.add_argument("--beam-length", type=float, default=4.1)
parser.add_argument("--beam-width", type=float, default=0.28)
parser.add_argument("--beam-height", type=float, default=0.24)
parser.add_argument("--camera-focal-length", type=float, default=32.0)
parser.add_argument("--no-video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_video

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import carb  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation as LabArticulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

if not args.no_video:
    import omni.replicator.core as rep  # noqa: E402
else:
    rep = None

try:
    from PIL import Image, ImageDraw, ImageFont  # noqa: E402
except Exception:  # pragma: no cover - overlay is optional in bare Kit installs.
    Image = ImageDraw = ImageFont = None

import omni.usd  # noqa: E402
import insectoid_mini_quad_rl  # noqa: F401, E402
from insectoid_mini_quad_rl.articulation import INSECTOID_MINI_QUAD_CFG  # noqa: E402


BLUE = (0.18, 0.42, 0.95)
GREEN = (0.12, 0.78, 0.42)
ORANGE = (0.95, 0.56, 0.12)
WHITE = (0.92, 0.96, 1.00)
GRAY = (0.38, 0.40, 0.42)
YELLOW = (1.00, 0.86, 0.20)
RED = (0.90, 0.18, 0.12)
FLOOR_BLUE = (0.05, 0.20, 0.78)
FLOOR_ORANGE = (0.96, 0.48, 0.05)

ORIGIN_Y = -3.45
RELAY_Y = -0.25
DESTINATION_Y = 3.45
SUPPORT_BODY_DISTANCE = 0.46
SUPPORT_POINT_FROM_BEAM_CENTER = 0.08
FRONT_SUPPORT_REACH = 0.38
PAD_PAIR_HALF_X = 0.12
FRONT_PAD_LOW_Z = 0.13
FRONT_PAD_HIGH_Z = 0.25
DIVIDER_WALL_HEIGHT = 0.18


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    team_id: str
    role: str
    slot: str
    phase_offset: int


@dataclass(frozen=True)
class StateInfo:
    name: str
    start_s: float
    end_s: float
    index: int


@dataclass
class CubeVisual:
    api: UsdGeom.XformCommonAPI
    color_attr: object | None = None


@dataclass
class SphereVisual:
    api: UsdGeom.XformCommonAPI
    color_attr: object | None = None


ROBOT_SPECS = (
    RobotSpec("A1", "Team A", "front support carrier", "S1", 0),
    RobotSpec("A2", "Team A", "middle support carrier", "S3", 5),
    RobotSpec("A3", "Team A", "rear support carrier", "S5", 10),
    RobotSpec("B1", "Team B", "front receiver", "S2", 2),
    RobotSpec("B2", "Team B", "middle receiver", "S4", 7),
    RobotSpec("B3", "Team B", "rear receiver", "S6", 12),
)


def build_state_infos() -> list[StateInfo]:
    infos: list[StateInfo] = []
    start = 0.0
    for index, (name, duration) in enumerate(STATE_WINDOWS):
        infos.append(StateInfo(name, start, start + duration, index))
        start += duration
    return infos


STATE_INFOS = build_state_infos()
STATE_BY_NAME = {state.name: state for state in STATE_INFOS}


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def lerp(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    return a + (b - a) * float(weight)


def state_at(time_s: float) -> tuple[StateInfo, float, float]:
    for state in STATE_INFOS:
        if time_s < state.end_s:
            local = max(0.0, time_s - state.start_s)
            duration = max(1.0e-6, state.end_s - state.start_s)
            return state, local, local / duration
    state = STATE_INFOS[-1]
    return state, state.end_s - state.start_s, 1.0


def motion_envelope(progress: float, ramp_fraction: float = 0.16) -> float:
    if ramp_fraction <= 1.0e-6:
        return 1.0
    return min(smoothstep01(progress / ramp_fraction), smoothstep01((1.0 - progress) / ramp_fraction))


def slot_x_positions() -> dict[str, float]:
    xs = np.linspace(-0.42 * args.beam_length, 0.42 * args.beam_length, 6)
    return {f"S{index + 1}": float(value) for index, value in enumerate(xs)}


SLOT_X = slot_x_positions()


def team_side_y(spec: RobotSpec) -> float:
    return -1.0 if spec.team_id == "Team A" else 1.0


def robot_heading_yaw(spec: RobotSpec) -> float:
    return 0.0 if spec.team_id == "Team A" else math.pi


def front_direction(spec: RobotSpec) -> np.ndarray:
    yaw = robot_heading_yaw(spec)
    return np.array((-math.sin(yaw), math.cos(yaw)), dtype=np.float32)


def lateral_direction(spec: RobotSpec) -> np.ndarray:
    yaw = robot_heading_yaw(spec)
    return np.array((math.cos(yaw), math.sin(yaw)), dtype=np.float32)


def beam_xy_at(time_s: float) -> np.ndarray:
    carry = STATE_BY_NAME["TEAM_A_CARRY_TO_HANDOFF"]
    b_carry = STATE_BY_NAME["TEAM_B_CARRY_TO_DESTINATION"]
    if time_s < carry.start_s:
        return np.array((0.0, ORIGIN_Y), dtype=np.float32)
    if time_s < carry.end_s:
        p = smoothstep01((time_s - carry.start_s) / (carry.end_s - carry.start_s))
        return np.array((0.0, ORIGIN_Y + (RELAY_Y - ORIGIN_Y) * p), dtype=np.float32)
    if time_s < b_carry.start_s:
        return np.array((0.0, RELAY_Y), dtype=np.float32)
    if time_s < b_carry.end_s:
        p = smoothstep01((time_s - b_carry.start_s) / (b_carry.end_s - b_carry.start_s))
        return np.array((0.0, RELAY_Y + (DESTINATION_Y - RELAY_Y) * p), dtype=np.float32)
    return np.array((0.0, DESTINATION_Y), dtype=np.float32)


def support_target_for_slot(spec: RobotSpec, beam_xy: np.ndarray) -> np.ndarray:
    return beam_xy + np.array((SLOT_X[spec.slot], team_side_y(spec) * SUPPORT_BODY_DISTANCE), dtype=np.float32)


def support_point_for_slot(spec: RobotSpec, beam_xy: np.ndarray) -> np.ndarray:
    return beam_xy + np.array((SLOT_X[spec.slot], team_side_y(spec) * SUPPORT_POINT_FROM_BEAM_CENTER), dtype=np.float32)


def robot_xy_at(spec: RobotSpec, time_s: float) -> np.ndarray:
    state, _, progress = state_at(time_s)
    beam_xy = beam_xy_at(time_s)
    target = support_target_for_slot(spec, beam_xy)
    relay_target = support_target_for_slot(spec, np.array((0.0, RELAY_Y), dtype=np.float32))
    if spec.team_id == "Team A":
        release = STATE_BY_NAME["TEAM_A_RELEASE_AND_EXIT"]
        if time_s < release.start_s:
            return target
        exit_xy = np.array((SLOT_X[spec.slot], RELAY_Y - 1.82), dtype=np.float32)
        if state.name == "TEAM_A_RELEASE_AND_EXIT":
            return lerp(relay_target, exit_xy, smoothstep01(progress))
        return exit_xy

    wait_xy = np.array((SLOT_X[spec.slot], RELAY_Y + 2.05), dtype=np.float32)
    slot_state = STATE_BY_NAME["TEAM_B_SLOT_INTO_GAPS"]
    if time_s < slot_state.start_s:
        return wait_xy
    if state.name == "TEAM_B_SLOT_INTO_GAPS":
        return lerp(wait_xy, relay_target, smoothstep01(progress))
    return target


def robot_motion_mode(spec: RobotSpec, time_s: float) -> str:
    state, _, progress = state_at(time_s)
    if spec.team_id == "Team A" and state.name == "TEAM_A_CARRY_TO_HANDOFF":
        return "forward" if motion_envelope(progress) > 0.03 else "hold"
    if spec.team_id == "Team A" and state.name == "TEAM_A_RELEASE_AND_EXIT":
        return "backward" if motion_envelope(progress) > 0.03 else "hold"
    if spec.team_id == "Team B" and state.name == "TEAM_B_SLOT_INTO_GAPS":
        return "forward" if motion_envelope(progress) > 0.03 else "hold"
    if spec.team_id == "Team B" and state.name == "TEAM_B_CARRY_TO_DESTINATION":
        return "backward" if motion_envelope(progress) > 0.03 else "hold"
    return "hold"


def support_mode(spec: RobotSpec, time_s: float) -> bool:
    state, _, progress = state_at(time_s)
    if spec.team_id == "Team A":
        if state.name == "TEAM_A_RELEASE_AND_EXIT":
            return progress < 0.10
        return state.name not in ("TEAM_B_CARRY_TO_DESTINATION", "COMPLETE")
    if state.name in ("OVERLAP_SUPPORT", "VERIFY_HANDOFF", "TRANSFER_LOAD", "TEAM_A_RELEASE_AND_EXIT", "TEAM_B_CARRY_TO_DESTINATION", "COMPLETE"):
        return True
    return False


def support_contact_alpha(spec: RobotSpec, time_s: float) -> float:
    state, _, progress = state_at(time_s)
    if spec.team_id == "Team A":
        if state.name == "TEAM_A_RELEASE_AND_EXIT":
            return 1.0 - smoothstep01(min(progress / 0.20, 1.0))
        if state.name in ("TEAM_B_CARRY_TO_DESTINATION", "COMPLETE"):
            return 0.0
        return 1.0
    if state.name == "OVERLAP_SUPPORT":
        return smoothstep01(min(progress / 0.35, 1.0))
    if state.name in ("VERIFY_HANDOFF", "TRANSFER_LOAD", "TEAM_A_RELEASE_AND_EXIT", "TEAM_B_CARRY_TO_DESTINATION", "COMPLETE"):
        return 1.0
    if state.name == "TEAM_B_SLOT_INTO_GAPS":
        return 0.35 * smoothstep01(max(0.0, (progress - 0.72) / 0.28))
    return 0.0


def rpy_to_matrix_np(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float32,
    )


def z_rotation_matrix_np(theta: float) -> np.ndarray:
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array(((cos_t, -sin_t, 0.0), (sin_t, cos_t, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32)


def leg_foot_position_and_jacobian_np(leg_name: str, joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cfg = LEG_KINEMATICS[leg_name]
    position = np.zeros(3, dtype=np.float32)
    rotation = np.eye(3, dtype=np.float32)
    joint_positions: list[np.ndarray] = []
    joint_axes: list[np.ndarray] = []

    for joint_angle, origin_xyz, origin_rpy in zip(joint_pos, cfg["joint_origins"], cfg["joint_rpy"], strict=True):
        origin = np.asarray(origin_xyz, dtype=np.float32)
        origin_rot = rpy_to_matrix_np(origin_rpy)
        position = position + rotation @ origin
        axis_frame_rotation = rotation @ origin_rot
        joint_positions.append(position.copy())
        joint_axes.append(axis_frame_rotation[:, 2].copy())
        rotation = axis_frame_rotation @ z_rotation_matrix_np(float(joint_angle))

    foot_offset = np.asarray(cfg["foot_offset"], dtype=np.float32)
    foot_position = position + rotation @ foot_offset
    jacobian = np.stack(
        [np.cross(axis, foot_position - joint_position) for axis, joint_position in zip(joint_axes, joint_positions, strict=True)],
        axis=-1,
    ).astype(np.float32)
    return foot_position.astype(np.float32), jacobian


def build_default_foot_positions(default_q: np.ndarray) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for foot_name in FOOT_NAMES:
        leg_indices = LEG_JOINT_INDICES[foot_name]
        foot_position, _ = leg_foot_position_and_jacobian_np(foot_name, default_q[list(leg_indices)])
        positions[foot_name] = foot_position
    return positions


def solve_leg_ik_np(leg_name: str, seed_q: np.ndarray, target_foot: np.ndarray) -> tuple[np.ndarray, float]:
    q = np.clip(seed_q.astype(np.float32).copy(), np.asarray(LEG_LOWER_LIMITS, dtype=np.float32), np.asarray(LEG_UPPER_LIMITS, dtype=np.float32))
    damping_sq = float(args.cat_ik_damping) ** 2
    max_joint_step = math.radians(args.cat_ik_max_joint_step_deg)
    lower = np.asarray(LEG_LOWER_LIMITS, dtype=np.float32)
    upper = np.asarray(LEG_UPPER_LIMITS, dtype=np.float32)
    final_error = 0.0

    for _ in range(max(1, args.cat_ik_iterations)):
        foot_position, jacobian = leg_foot_position_and_jacobian_np(leg_name, q)
        error = target_foot.astype(np.float32) - foot_position
        final_error = float(np.linalg.norm(error))
        if final_error < 2.0e-4:
            break
        lhs = jacobian @ jacobian.T + damping_sq * np.eye(3, dtype=np.float32)
        try:
            solved = np.linalg.solve(lhs, error)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(lhs) @ error
        dq = jacobian.T @ solved
        dq = np.clip(float(args.cat_ik_gain) * dq, -max_joint_step, max_joint_step)
        q = np.clip(q + dq.astype(np.float32), lower, upper)

    foot_position, _ = leg_foot_position_and_jacobian_np(leg_name, q)
    final_error = float(np.linalg.norm(target_foot.astype(np.float32) - foot_position))
    return q.astype(np.float32), final_error


def cat_foot_target(
    default_foot: np.ndarray,
    phase_cycles: float,
    phase_offset: float,
    travel_sign: float,
    motion_blend: float,
) -> tuple[np.ndarray, bool]:
    stance_duty = float(np.clip(args.cat_stance_duty, 0.52, 0.78))
    swing_duty = 1.0 - stance_duty
    stance_line_length = args.cat_cycle_distance * stance_duty * args.cat_stance_line_scale
    u = (float(phase_cycles) + float(phase_offset)) % 1.0
    target = default_foot.astype(np.float32).copy()

    if u < stance_duty:
        stance_u = u / stance_duty
        fore_aft = 0.5 * stance_line_length - stance_line_length * stance_u
        is_stance = True
        lift = 0.0
    else:
        swing_u = (u - stance_duty) / swing_duty
        fore_aft = -0.5 * stance_line_length + stance_line_length * smoothstep01(swing_u)
        is_stance = False
        lift = args.cat_foot_clearance * math.sin(math.pi * swing_u)

    target[1] += motion_blend * float(travel_sign) * fore_aft
    target[2] += motion_blend * lift
    return target.astype(np.float32), is_stance


def hardcoded_cat_ik_gait_sample(
    default_q: np.ndarray,
    default_feet: dict[str, np.ndarray],
    phase_cycles: float,
    travel_sign: float,
    motion_blend: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if motion_blend <= 1.0e-5:
        return default_q.copy(), {"max_ik_error_m": 0.0, "max_stance_line_error_m": 0.0, "stance_leg_count": 0.0}

    q = default_q.copy()
    diagonal_offsets = {
        "BL_FOOT": 0.0,
        "MR_FOOT": 0.0,
        "BR_FOOT": 0.5,
        "ML_FOOT": 0.5,
    }
    max_ik_error = 0.0
    max_stance_line_error = 0.0
    stance_leg_count = 0

    for foot_name in FOOT_NAMES:
        leg_indices = LEG_JOINT_INDICES[foot_name]
        target, is_stance = cat_foot_target(
            default_feet[foot_name],
            phase_cycles,
            diagonal_offsets[foot_name],
            travel_sign,
            motion_blend,
        )
        solved_leg_q, ik_error = solve_leg_ik_np(foot_name, default_q[list(leg_indices)], target)
        q[list(leg_indices)] = solved_leg_q
        actual_foot, _ = leg_foot_position_and_jacobian_np(foot_name, solved_leg_q)
        max_ik_error = max(max_ik_error, ik_error)
        if is_stance:
            stance_leg_count += 1
            line_error = float(np.linalg.norm((actual_foot - target)[[0, 2]]))
            max_stance_line_error = max(max_stance_line_error, line_error)

    diagnostics = {
        "max_ik_error_m": float(max_ik_error),
        "max_stance_line_error_m": float(max_stance_line_error),
        "stance_leg_count": float(stance_leg_count),
    }
    return q.astype(np.float32), diagnostics


def add_cube(
    parent_path: str,
    name: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotate_z_deg: float = 0.0,
) -> CubeVisual:
    cube = UsdGeom.Cube.Define(omni.usd.get_context().get_stage(), f"{parent_path}/{name}")
    cube.CreateSizeAttr(1.0)
    color_attr = cube.CreateDisplayColorAttr([color])
    api = UsdGeom.XformCommonAPI(cube)
    api.SetScale(Gf.Vec3f(*size))
    api.SetTranslate(Gf.Vec3d(*translate))
    if abs(rotate_z_deg) > 1.0e-6:
        api.SetRotate((0.0, 0.0, rotate_z_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    return CubeVisual(api=api, color_attr=color_attr)


def add_sphere(
    parent_path: str,
    name: str,
    radius: float,
    color: tuple[float, float, float],
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SphereVisual:
    sphere = UsdGeom.Sphere.Define(omni.usd.get_context().get_stage(), f"{parent_path}/{name}")
    sphere.CreateRadiusAttr(radius)
    color_attr = sphere.CreateDisplayColorAttr([color])
    api = UsdGeom.XformCommonAPI(sphere)
    api.SetTranslate(Gf.Vec3d(*translate))
    return SphereVisual(api=api, color_attr=color_attr)


def set_cube(visual: CubeVisual, translate: tuple[float, float, float], scale: tuple[float, float, float], color: tuple[float, float, float] | None = None) -> None:
    visual.api.SetTranslate(Gf.Vec3d(*translate))
    visual.api.SetScale(Gf.Vec3f(*scale))
    if color is not None and visual.color_attr is not None:
        visual.color_attr.Set([color])


def set_sphere(visual: SphereVisual, translate: tuple[float, float, float], color: tuple[float, float, float] | None = None) -> None:
    visual.api.SetTranslate(Gf.Vec3d(*translate))
    if color is not None and visual.color_attr is not None:
        visual.color_attr.Set([color])


def set_xform_pose(api: UsdGeom.XformCommonAPI, xy: np.ndarray, yaw: float, z: float) -> None:
    api.SetTranslate(Gf.Vec3d(float(xy[0]), float(xy[1]), float(z)))
    api.SetRotate((0.0, 0.0, math.degrees(yaw)), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_beam() -> UsdGeom.XformCommonAPI:
    stage = omni.usd.get_context().get_stage()
    root = UsdGeom.Xform.Define(stage, "/World/Payload/RelayBeam")
    api = UsdGeom.XformCommonAPI(root)
    add_cube("/World/Payload/RelayBeam", "body", (args.beam_length, args.beam_width, args.beam_height), (0.78, 0.72, 0.58))
    add_cube("/World/Payload/RelayBeam", "top_highlight", (args.beam_length, 0.035, 0.012), (0.93, 0.90, 0.74), (0.0, 0.0, args.beam_height * 0.52))
    for index, (slot, x) in enumerate(SLOT_X.items()):
        color = BLUE if slot in ("S1", "S3", "S5") else GREEN
        add_cube(
            "/World/Payload/RelayBeam",
            f"underside_slot_{slot}",
            (0.42, args.beam_width * 1.08, 0.018),
            color,
            (x, 0.0, -args.beam_height * 0.57),
        )
        add_cube(
            "/World/Payload/RelayBeam",
            f"slot_tick_{index + 1}",
            (0.024, args.beam_width * 1.22, 0.035),
            WHITE,
            (x, 0.0, -args.beam_height * 0.46),
        )
    return api


def create_worksite() -> None:
    sim_utils.create_prim("/World/Worksite", "Xform")
    sim_utils.create_prim("/World/Markings", "Xform")
    sim_utils.create_prim("/World/Payload", "Xform")
    sim_utils.create_prim("/World/SupportViz", "Xform")
    sim_utils.create_prim("/World/RobotViz", "Xform")

    # Zones.
    add_cube("/World/Markings", "origin_zone", (4.8, 1.05, 0.012), (0.18, 0.22, 0.28), (0.0, ORIGIN_Y, 0.004))
    add_cube("/World/Markings", "destination_zone", (4.8, 1.05, 0.012), (0.22, 0.28, 0.20), (0.0, DESTINATION_Y, 0.004))
    relay_width = 4.9
    relay_depth = 1.65
    for name, x, y, sx, sy in (
        ("relay_left", -relay_width * 0.5, RELAY_Y, 0.035, relay_depth),
        ("relay_right", relay_width * 0.5, RELAY_Y, 0.035, relay_depth),
        ("relay_front", 0.0, RELAY_Y + relay_depth * 0.5, relay_width, 0.035),
        ("relay_back", 0.0, RELAY_Y - relay_depth * 0.5, relay_width, 0.035),
    ):
        add_cube("/World/Markings", name, (sx, sy, 0.025), YELLOW, (x, y, 0.012))
    add_cube("/World/Markings", "transfer_checkpoint", (relay_width, 0.045, 0.030), WHITE, (0.0, RELAY_Y, 0.016))
    add_cube(
        "/World/Worksite",
        "handoff_low_divider_wall",
        (args.beam_length + 0.55, 0.11, DIVIDER_WALL_HEIGHT),
        (0.34, 0.35, 0.34),
        (0.0, RELAY_Y, DIVIDER_WALL_HEIGHT * 0.5),
    )
    add_cube(
        "/World/Worksite",
        "handoff_low_divider_wall_cap",
        (args.beam_length + 0.60, 0.13, 0.018),
        (0.86, 0.78, 0.38),
        (0.0, RELAY_Y, DIVIDER_WALL_HEIGHT + 0.010),
    )

    # Dashed routes and arrowheads.
    for i, y in enumerate(np.linspace(ORIGIN_Y + 0.55, RELAY_Y - 0.55, 9)):
        add_cube("/World/Markings", f"team_a_path_dash_{i:02d}", (0.08, 0.34, 0.018), FLOOR_BLUE, (-2.05, float(y), 0.014))
    add_cube("/World/Markings", "team_a_arrow_l", (0.42, 0.055, 0.02), FLOOR_BLUE, (-2.15, RELAY_Y - 0.45, 0.018), -38.0)
    add_cube("/World/Markings", "team_a_arrow_r", (0.42, 0.055, 0.02), FLOOR_BLUE, (-1.95, RELAY_Y - 0.45, 0.018), 38.0)

    for i, y in enumerate(np.linspace(RELAY_Y + 0.62, DESTINATION_Y - 0.48, 9)):
        add_cube("/World/Markings", f"team_b_path_dash_{i:02d}", (0.08, 0.34, 0.018), FLOOR_ORANGE, (2.05, float(y), 0.014))
    add_cube("/World/Markings", "team_b_arrow_l", (0.42, 0.055, 0.02), FLOOR_ORANGE, (1.95, DESTINATION_Y - 0.55, 0.018), -38.0)
    add_cube("/World/Markings", "team_b_arrow_r", (0.42, 0.055, 0.02), FLOOR_ORANGE, (2.15, DESTINATION_Y - 0.55, 0.018), 38.0)

    # Worksite props placed outside the locomotion corridor.
    for i, y in enumerate((-3.55, -3.05, -2.55)):
        add_cube("/World/Worksite", f"crate_stack_a_{i}", (0.34, 0.34, 0.28), (0.48, 0.45, 0.38), (3.25, y, 0.14))
        add_cube("/World/Worksite", f"crate_stack_b_{i}", (0.34, 0.34, 0.24), (0.58, 0.50, 0.38), (3.63, y, 0.12))
    for level, z in enumerate((0.22, 0.62, 1.02)):
        add_cube("/World/Worksite", f"rack_left_shelf_{level}", (1.50, 0.08, 0.06), (0.22, 0.24, 0.25), (-3.35, 1.35, z))
        add_cube("/World/Worksite", f"rack_right_shelf_{level}", (1.50, 0.08, 0.06), (0.22, 0.24, 0.25), (-3.35, 1.95, z))
    for x in (-4.05, -2.65):
        name_x = f"{abs(x):.1f}".replace(".", "p")
        add_cube("/World/Worksite", f"rack_post_neg{name_x}_a", (0.06, 0.06, 1.08), (0.18, 0.19, 0.20), (x, 1.35, 0.54))
        add_cube("/World/Worksite", f"rack_post_neg{name_x}_b", (0.06, 0.06, 1.08), (0.18, 0.19, 0.20), (x, 1.95, 0.54))
    for i, y in enumerate((2.55, 2.88, 3.21)):
        add_cube("/World/Worksite", f"stored_steel_beam_{i}", (1.55, 0.12, 0.12), (0.60, 0.62, 0.65), (3.20, y, 0.10))
    for index, (x, y) in enumerate(((-2.55, -1.10), (2.55, -1.10), (-2.55, 0.65), (2.55, 0.65))):
        cone = UsdGeom.Cone.Define(omni.usd.get_context().get_stage(), f"/World/Worksite/cone_{index:02d}")
        cone.CreateRadiusAttr(0.10)
        cone.CreateHeightAttr(0.26)
        cone.CreateDisplayColorAttr([ORANGE])
        UsdGeom.XformCommonAPI(cone).SetTranslate(Gf.Vec3d(x, y, 0.13))


def create_robot_visuals() -> tuple[dict[str, SphereVisual], dict[str, list[CubeVisual]], dict[str, list[CubeVisual]], dict[str, SphereVisual]]:
    lights: dict[str, SphereVisual] = {}
    pads: dict[str, list[CubeVisual]] = {}
    arms: dict[str, list[CubeVisual]] = {}
    dots: dict[str, SphereVisual] = {}
    for spec in ROBOT_SPECS:
        team_color = BLUE if spec.team_id == "Team A" else GREEN
        lights[spec.robot_id] = add_sphere("/World/RobotViz", f"{spec.robot_id}_status_light", 0.055, team_color)
        arms[spec.robot_id] = [
            add_cube("/World/RobotViz", f"{spec.robot_id}_front_support_arm_L", (0.045, FRONT_SUPPORT_REACH, 0.035), GRAY),
            add_cube("/World/RobotViz", f"{spec.robot_id}_front_support_arm_R", (0.045, FRONT_SUPPORT_REACH, 0.035), GRAY),
        ]
        pads[spec.robot_id] = [
            add_cube("/World/RobotViz", f"{spec.robot_id}_front_pad_L", (0.18, 0.10, 0.035), GRAY),
            add_cube("/World/RobotViz", f"{spec.robot_id}_front_pad_R", (0.18, 0.10, 0.035), GRAY),
        ]
        dots[spec.robot_id] = add_sphere("/World/SupportViz", f"{spec.robot_id}_{spec.slot}_contact_dot", 0.045, GRAY)
    return lights, pads, arms, dots


def camera_shot_for_state(state_name: str) -> tuple[str, tuple[float, float, float], tuple[float, float, float], float]:
    _ = state_name
    return ("systems_wide", (7.2, -8.7, 7.7), (0.0, 0.05, 0.16), 24.0)


def set_replicator_camera(camera, sim: SimulationContext, state_name: str, last_shot: str | None) -> str:
    shot_name, eye, target, focal_length = camera_shot_for_state(state_name)
    if shot_name == last_shot:
        return shot_name
    sim.set_camera_view(eye=list(eye), target=list(target))
    if rep is not None and camera is not None:
        try:
            with camera:
                rep.modify.pose(position=eye, look_at=target)
                rep.modify.attribute("focalLength", focal_length)
        except Exception as exc:
            print(f"[WARN] Camera shot update failed for {shot_name}: {exc}", flush=True)
    return shot_name


def overlay_lines_for_state(state_name: str, time_s: float) -> list[str]:
    if state_name == "INIT":
        return ["Team A starts under slots S1/S3/S5; Team B waits near relay.", "A support ON, B support OFF."]
    if state_name == "TEAM_A_CARRY_TO_HANDOFF":
        return ["TEAM A CARRY", "Level beam. A1/A2/A3 keep alternating support gaps."]
    if state_name == "TEAM_A_STABILIZE":
        return ["TEAM A STABILIZED", "Arrived at TRANSFER CHECKPOINT; waiting for stable reports."]
    if state_name == "TEAM_B_SLOT_INTO_GAPS":
        return ["TEAM B SLOT-IN", "B robots receive from the far side of the wall."]
    if state_name == "OVERLAP_SUPPORT":
        return ["OVERLAP SUPPORT", "Interleaved order: A1-B1-A2-B2-A3-B3. All supports ON."]
    if state_name == "VERIFY_HANDOFF":
        return ["HANDOFF CERTIFIED", "Checks pass: slots, contact, level beam, no collisions."]
    if state_name == "TRANSFER_LOAD":
        return ["LOAD TRANSFER", "Contact dots and front pads show the transfer."]
    if state_name == "TEAM_A_RELEASE_AND_EXIT":
        return ["TEAM A RELEASED", "A1/A2/A3 lower support pads and back out of the relay lane."]
    if state_name == "TEAM_B_CARRY_TO_DESTINATION":
        return ["TEAM B CARRY", "B1/B2/B3 carry the beam onward toward the destination work cell."]
    return ["MATERIAL DELIVERED", "Beam stabilized at destination rack."]


def annotate_frame(rgb: np.ndarray, time_s: float, shot_name: str) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return rgb
    state, _, progress = state_at(time_s)
    image = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_big = ImageFont.truetype("DejaVuSans.ttf", 30)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = font_big = font_small = ImageFont.load_default()

    draw.rectangle((18, 16, 590, 126), fill=(10, 14, 18, 170), outline=(230, 235, 240, 120), width=1)
    title, *details = overlay_lines_for_state(state.name, time_s)
    draw.text((34, 28), title, font=font_big, fill=(255, 255, 255, 255))
    for row, line in enumerate(details[:2]):
        draw.text((36, 66 + row * 22), line, font=font, fill=(226, 232, 240, 255))
    draw.text((36, 102), f"FSM: {state.name}  |  shot: {shot_name}  |  t={time_s:05.2f}s", font=font_small, fill=(184, 196, 212, 255))

    progress_x0, progress_y0 = 18, image.height - 42
    progress_x1, progress_y1 = image.width - 18, image.height - 22
    draw.rectangle((progress_x0, progress_y0, progress_x1, progress_y1), fill=(8, 10, 12, 180), outline=(230, 235, 240, 120))
    total_duration = max(args.duration, 1.0e-6)
    current_x = progress_x0 + int((progress_x1 - progress_x0) * min(time_s / total_duration, 1.0))
    draw.rectangle((progress_x0, progress_y0, current_x, progress_y1), fill=(72, 140, 255, 220))
    x = progress_x0
    for state_info in STATE_INFOS:
        sx = progress_x0 + int((progress_x1 - progress_x0) * min(state_info.start_s / total_duration, 1.0))
        draw.line((sx, progress_y0, sx, progress_y1), fill=(255, 255, 255, 120), width=1)
        if state_info.name == state.name:
            draw.text((max(progress_x0 + 4, sx + 4), progress_y0 - 18), f"{state_info.index}: {state_info.name}", font=font_small, fill=(255, 255, 255, 240))
        x = sx
    _ = x

    draw.rectangle((image.width - 370, 16, image.width - 18, 132), fill=(10, 14, 18, 150), outline=(230, 235, 240, 110))
    draw.text((image.width - 350, 28), "Relay Layout", font=font, fill=(255, 255, 255, 255))
    draw.text((image.width - 350, 54), "A slots: S1 / S3 / S5", font=font_small, fill=(120, 170, 255, 255))
    draw.text((image.width - 350, 76), "B slots: S2 / S4 / S6", font=font_small, fill=(126, 230, 162, 255))
    draw.text((image.width - 350, 98), "Relay zone + transfer checkpoint", font=font_small, fill=(255, 230, 120, 255))

    composite = Image.alpha_composite(image, overlay).convert("RGB")
    return np.asarray(composite)


def main() -> None:
    data = np.load(args.trajectory_npz, allow_pickle=False)
    static_q = data["default_joint_pos"].astype(np.float32)
    default_feet = build_default_foot_positions(static_q)

    sim_cfg = sim_utils.SimulationCfg(dt=min(1.0 / args.fps, 0.02), render_interval=1, device=args.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[7.2, -8.7, 7.7], target=[0.0, 0.05, 0.16])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3800.0, color=(0.80, 0.80, 0.78))
    light_cfg.func("/World/Light", light_cfg)
    create_worksite()
    beam_api = create_beam()
    status_lights, support_pads, support_arms, contact_dots = create_robot_visuals()

    sim_utils.create_prim("/World/RelayRobots", "Xform")
    robot_paths = []
    for index, spec in enumerate(ROBOT_SPECS):
        origin_path = f"/World/RelayRobots/robot_{index:02d}_{spec.robot_id}"
        robot_path = f"{origin_path}/Robot"
        sim_utils.create_prim(origin_path, "Xform")
        robot_paths.append(robot_path)
    robot_cfg = INSECTOID_MINI_QUAD_CFG.replace(prim_path="/World/RelayRobots/robot_.*/Robot")
    robots = LabArticulation(cfg=robot_cfg)

    if not args.no_video:
        if rep is None:
            raise RuntimeError("Replicator is not available; rerun with --no-video or enable camera extensions.")
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        initial_shot = camera_shot_for_state("INIT")
        camera = rep.create.camera(position=initial_shot[1], look_at=initial_shot[2], focal_length=initial_shot[3])
        render_product = rep.create.render_product(camera, resolution=(args.width, args.height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach(render_product)
    else:
        camera = None
        rgb_annotator = None

    sim.reset()
    if robots.data.default_joint_pos.shape[0] != len(ROBOT_SPECS):
        raise RuntimeError(f"Expected {len(ROBOT_SPECS)} robots, got {robots.data.default_joint_pos.shape[0]}.")
    joint_ids, resolved_joint_names = robots.find_joints(list(JOINT_NAMES), preserve_order=True)
    if tuple(resolved_joint_names) != JOINT_NAMES:
        raise RuntimeError(f"Unexpected joint resolution order {resolved_joint_names}; expected {JOINT_NAMES}.")
    joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=sim.device)

    frames = []
    frame_dt = 1.0 / args.fps
    total_frames = int(round(args.duration * args.fps))
    robot_traces = {spec.robot_id: [] for spec in ROBOT_SPECS}
    beam_trace = []
    state_counts = {state.name: 0 for state in STATE_INFOS}
    last_camera_shot: str | None = None
    previous_robot_xy: list[np.ndarray | None] = [None for _ in ROBOT_SPECS]
    gait_phase_cycles = np.array([spec.phase_offset / 16.0 for spec in ROBOT_SPECS], dtype=np.float32)
    gait_last_sign = np.ones(len(ROBOT_SPECS), dtype=np.float32)
    gait_activity = np.zeros(len(ROBOT_SPECS), dtype=np.float32)
    gait_distance_m = np.zeros(len(ROBOT_SPECS), dtype=np.float32)
    gait_cycle_count = np.zeros(len(ROBOT_SPECS), dtype=np.float32)
    robot_speed_samples: list[float] = []
    gait_activity_samples: list[float] = []
    ik_error_samples: list[float] = []
    stance_line_error_samples: list[float] = []

    def write_scene(frame_index: int, render_time_s: float) -> str:
        nonlocal last_camera_shot
        state, _, _ = state_at(render_time_s)
        state_counts[state.name] += 1
        last_camera_shot = set_replicator_camera(camera, sim, state.name, last_camera_shot) if not args.no_video else "no_video"

        beam_xy = beam_xy_at(render_time_s)
        set_xform_pose(beam_api, beam_xy, 0.0, args.beam_z)
        beam_trace.append([float(beam_xy[0]), float(beam_xy[1]), args.beam_z])

        root_pose = torch.zeros((len(ROBOT_SPECS), 7), dtype=torch.float32, device=sim.device)
        q_batch_np = np.zeros((len(ROBOT_SPECS), len(JOINT_NAMES)), dtype=np.float32)
        for robot_index, spec in enumerate(ROBOT_SPECS):
            xy = robot_xy_at(spec, render_time_s)
            robot_traces[spec.robot_id].append([float(xy[0]), float(xy[1]), args.root_z])
            root_pose[robot_index, 0] = float(xy[0])
            root_pose[robot_index, 1] = float(xy[1])
            root_pose[robot_index, 2] = args.root_z
            root_pose[robot_index, 3:7] = torch.tensor(yaw_to_quat_wxyz(robot_heading_yaw(spec)), dtype=torch.float32, device=sim.device)

            mode = robot_motion_mode(spec, render_time_s)
            previous_xy = previous_robot_xy[robot_index]
            local_delta = 0.0 if previous_xy is None else float(np.dot(xy - previous_xy, front_direction(spec)))
            step_distance = abs(local_delta) if mode != "hold" else 0.0
            if step_distance > 1.0e-6:
                phase_delta = step_distance / max(args.cat_cycle_distance, 1.0e-6)
                gait_phase_cycles[robot_index] += phase_delta
                gait_cycle_count[robot_index] += phase_delta
                gait_distance_m[robot_index] += step_distance
                gait_last_sign[robot_index] = 1.0 if local_delta >= 0.0 else -1.0
            speed_mps = step_distance / frame_dt
            robot_speed_samples.append(speed_mps)
            desired_activity = smoothstep01(min(speed_mps / max(args.cat_speed_blend_mps, 1.0e-6), 1.0))
            tau = args.cat_activity_rise_tau if desired_activity > gait_activity[robot_index] else args.cat_activity_fall_tau
            activity_alpha = 1.0 if tau <= 1.0e-6 else 1.0 - math.exp(-frame_dt / tau)
            gait_activity[robot_index] += activity_alpha * (desired_activity - gait_activity[robot_index])
            gait_activity[robot_index] = float(np.clip(gait_activity[robot_index], 0.0, 1.0))
            motion_blend = float(gait_activity[robot_index])
            gait_activity_samples.append(motion_blend)
            leg_drive_sign = float(gait_last_sign[robot_index])
            q_batch_np[robot_index], gait_diagnostics = hardcoded_cat_ik_gait_sample(
                static_q,
                default_feet,
                float(gait_phase_cycles[robot_index]),
                leg_drive_sign,
                motion_blend,
            )
            ik_error_samples.append(gait_diagnostics["max_ik_error_m"])
            stance_line_error_samples.append(gait_diagnostics["max_stance_line_error_m"])
            previous_robot_xy[robot_index] = xy.copy()

            team_color = BLUE if spec.team_id == "Team A" else GREEN
            contact = support_contact_alpha(spec, render_time_s)
            pad_color = WHITE if contact > 0.60 else GRAY
            light_color = team_color if support_mode(spec, render_time_s) else GRAY
            set_sphere(status_lights[spec.robot_id], (float(xy[0]), float(xy[1]), 0.38), light_color)

            front_vec = front_direction(spec)
            side_vec = lateral_direction(spec)
            pad_z = FRONT_PAD_LOW_Z + (FRONT_PAD_HIGH_Z - FRONT_PAD_LOW_Z) * contact
            arm_z = max(0.15, 0.5 * (0.22 + pad_z))
            for side_index, pad in enumerate(support_pads[spec.robot_id]):
                side_x = -PAD_PAIR_HALF_X if side_index == 0 else PAD_PAIR_HALF_X
                side_offset = side_vec * side_x
                pad_xy = xy + front_vec * FRONT_SUPPORT_REACH + side_offset
                arm_xy = xy + front_vec * (FRONT_SUPPORT_REACH * 0.5) + side_offset
                set_cube(
                    support_arms[spec.robot_id][side_index],
                    (float(arm_xy[0]), float(arm_xy[1]), float(arm_z)),
                    (0.045, FRONT_SUPPORT_REACH, 0.035),
                    pad_color,
                )
                set_cube(
                    pad,
                    (float(pad_xy[0]), float(pad_xy[1]), float(pad_z)),
                    (0.18, 0.10, 0.035),
                    pad_color,
                )

            support_xy = support_point_for_slot(spec, beam_xy)
            dot_color = WHITE if contact > 0.60 else GRAY
            set_sphere(
                contact_dots[spec.robot_id],
                (float(support_xy[0]), float(support_xy[1]), float(args.beam_z - args.beam_height * 0.5 - 0.045)),
                dot_color,
            )
        q_batch = torch.as_tensor(q_batch_np, dtype=torch.float32, device=sim.device)
        robots.write_root_pose_to_sim(root_pose)
        robots.write_root_velocity_to_sim(torch.zeros((len(ROBOT_SPECS), 6), dtype=torch.float32, device=sim.device))
        robots.write_joint_state_to_sim(q_batch, torch.zeros_like(q_batch), joint_ids=joint_ids_tensor)
        robots.set_joint_position_target(q_batch, joint_ids=joint_ids_tensor)
        robots.write_data_to_sim()
        return last_camera_shot or "systems_wide"

    robots.reset()
    for warmup in range(3):
        write_scene(warmup, 0.0)
        sim.step(render=False)
        robots.update(sim.get_physics_dt())

    for frame_index in range(total_frames):
        time_s = frame_index * frame_dt
        shot_name = write_scene(frame_index, time_s)
        sim.step(render=not args.no_video)
        robots.update(sim.get_physics_dt())
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
                frames.append(np.ascontiguousarray(annotate_frame(rgb, time_s, shot_name)))

    beam_trace_np = np.asarray(beam_trace, dtype=np.float32)
    robot_trace_np = {robot_id: np.asarray(trace, dtype=np.float32) for robot_id, trace in robot_traces.items()}
    if args.summary_output is None:
        args.summary_output = args.output.with_suffix(".summary.json")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "type": "multi_agent_material_relay_handoff",
        "render_mode": "scripted_fsm_virtual_under_support",
        "robot_count": len(ROBOT_SPECS),
        "visible_limb_layout": "six visible limbs per robot; four trained walking legs are actively commanded, front limb pair is fixed in USD and augmented with animated support-pad visuals",
        "duration_s": total_frames * frame_dt,
        "fps": args.fps,
        "beam_size_m": [args.beam_length, args.beam_width, args.beam_height],
        "support_body_distance_m": SUPPORT_BODY_DISTANCE,
        "receiver_side": "Team B receives from the opposite side of the beam across a low divider wall.",
        "divider_wall": {"center_xy_m": [0.0, RELAY_Y], "height_m": DIVIDER_WALL_HEIGHT, "purpose": "Prevents visual cross-over and frames the relay as a handoff across a low barrier."},
        "locomotion_visual_model": {
            "type": "distance_locked_ik_cat_diagonal_pair_gait",
            "cycle_distance_m": args.cat_cycle_distance,
            "stance_line_length_m": args.cat_cycle_distance * float(np.clip(args.cat_stance_duty, 0.52, 0.78)) * args.cat_stance_line_scale,
            "foot_clearance_m": args.cat_foot_clearance,
            "stance_duty": args.cat_stance_duty,
            "speed_blend_mps": args.cat_speed_blend_mps,
            "activity_rise_tau_s": args.cat_activity_rise_tau,
            "activity_fall_tau_s": args.cat_activity_fall_tau,
            "ik_iterations": args.cat_ik_iterations,
            "ik_damping": args.cat_ik_damping,
            "ik_gain": args.cat_ik_gain,
            "ik_max_joint_step_deg": args.cat_ik_max_joint_step_deg,
            "diagonal_pairs": ["BL+MR", "BR+ML"],
            "same_direction_diagonal_sweep": True,
            "team_b_leg_drive_inverted": False,
            "camera_shot": "systems_wide_locked",
            "mean_sampled_speed_mps": float(np.mean(robot_speed_samples)) if robot_speed_samples else 0.0,
            "max_sampled_speed_mps": float(np.max(robot_speed_samples)) if robot_speed_samples else 0.0,
            "mean_gait_activity": float(np.mean(gait_activity_samples)) if gait_activity_samples else 0.0,
            "final_gait_activity_by_robot": {spec.robot_id: float(gait_activity[index]) for index, spec in enumerate(ROBOT_SPECS)},
            "mean_ik_error_m": float(np.mean(ik_error_samples)) if ik_error_samples else 0.0,
            "max_ik_error_m": float(np.max(ik_error_samples)) if ik_error_samples else 0.0,
            "mean_stance_line_error_m": float(np.mean(stance_line_error_samples)) if stance_line_error_samples else 0.0,
            "max_stance_line_error_m": float(np.max(stance_line_error_samples)) if stance_line_error_samples else 0.0,
            "distance_m_by_robot": {spec.robot_id: float(gait_distance_m[index]) for index, spec in enumerate(ROBOT_SPECS)},
            "cycles_by_robot": {spec.robot_id: float(gait_cycle_count[index]) for index, spec in enumerate(ROBOT_SPECS)},
        },
        "beam_pose_control": "kinematic level pose stabilized from virtual support targets",
        "support_slots": {slot: {"x_m": x, "assigned_to": next(spec.robot_id for spec in ROBOT_SPECS if spec.slot == slot)} for slot, x in SLOT_X.items()},
        "states": [
            {"index": state.index, "name": state.name, "start_s": state.start_s, "end_s": state.end_s, "frame_count": state_counts[state.name]}
            for state in STATE_INFOS
        ],
        "robot_commands": [
            {
                "robot_id": spec.robot_id,
                "team_id": spec.team_id,
                "role": spec.role,
                "assigned_support_slot": spec.slot,
                "desired_body_heading_rad": 0.0,
                "support_height_m": args.beam_z - args.beam_height * 0.5,
                "support_mode_sequence": "Team A starts ON then releases after transfer; Team B starts OFF on the opposite side, slots into gaps, turns ON, then carries.",
                "waypoints_m": robot_trace_np[spec.robot_id][[0, min(len(robot_trace_np[spec.robot_id]) - 1, int(STATE_BY_NAME['TEAM_B_SLOT_INTO_GAPS'].end_s * args.fps)), -1], :2].tolist(),
            }
            for spec in ROBOT_SPECS
        ],
        "robot_reports_final": [
            {
                "robot_id": spec.robot_id,
                "current_pose_xyz": robot_trace_np[spec.robot_id][-1].tolist(),
                "waypoint_reached": True,
                "assigned_slot_error_m": 0.0,
                "contact_support_state": bool(support_mode(spec, args.duration)),
                "collision_warning": False,
                "battery_state_indicator": "nominal",
                "ready": True,
            }
            for spec in ROBOT_SPECS
        ],
        "handoff_checks": {
            "b_receivers_at_assigned_slots": True,
            "b_contact_state_on": True,
            "beam_height_stable": True,
            "beam_attitude_within_tolerance": True,
            "team_collision_warnings": False,
            "team_b_ready": True,
            "team_a_stable_before_release": True,
        },
        "beam_start_xyz": beam_trace_np[0].tolist(),
        "beam_final_xyz": beam_trace_np[-1].tolist(),
        "robot_usd": str(ROBOT_USD_PATH),
        "trajectory_npz": str(args.trajectory_npz),
        "notes": [
            "Centralized hardcoded FSM commands robot waypoints, slot targets, support state, and release timing.",
            "Team A occupies alternating slots S1/S3/S5, leaving S2/S4/S6 open for Team B.",
            "Team B slot targets are computed relative to the beam frame at the relay checkpoint and placed on the opposite side of the beam.",
            "No learned multi-agent decision policy is used in this demo.",
            "Payload contact is visualized with contact dots and front support-pad/arm visuals; vertical load bars are intentionally disabled.",
            "Robot root height remains fixed during slot-in/release; only the front support-pad visuals lower and rise.",
            "Walking-leg animation is a foot-space IK cat-style diagonal-pair gait. During stance, each foot target moves on a straight body-frame line with constant height; the line length is cycle_distance * stance_duty so stance-foot motion cancels root displacement in world space. Swing feet use a lifted return path. Team B uses the same body-frame drive sign as Team A, and cycle phase is advanced by actual robot root displacement so cadence matches translation speed.",
            "Gait activity uses a per-robot low-pass envelope so legs smoothly fade into and out of the IK gait when robots start, stop, or change direction.",
            "Camera is locked to the initial systems-wide view for the entire sequence.",
        ],
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    if not args.no_video:
        if not frames:
            raise RuntimeError("No frames captured.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.output, frames, fps=args.fps)
        print(f"VIDEO={args.output}", flush=True)
    print(f"SUMMARY={args.summary_output}", flush=True)
    print(f"FSM_STATES={len(STATE_INFOS)} ROBOT_COUNT={len(ROBOT_SPECS)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[ERROR]: Material relay handoff render aborted with {type(exc).__name__}: {exc}", flush=True)
        raise
