#!/usr/bin/env python3
"""Render a large cooperative handling yard with several payload teams."""

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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trajectory-npz", type=Path, default=TRAJECTORY_PATH)
parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/renders/multi_object_payload_yard_v1.mp4")
parser.add_argument("--summary-output", type=Path, default=None)
parser.add_argument("--duration", type=float, default=22.0)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--playback-speed", type=float, default=2.5)
parser.add_argument("--gait-ramp-s", type=float, default=0.45)
parser.add_argument("--turn-coxa-amp", type=float, default=0.22)
parser.add_argument("--turn-femur-lift", type=float, default=0.18)
parser.add_argument("--turn-tibia-lift", type=float, default=0.10)
parser.add_argument("--turn-playback-multiplier", type=float, default=2.0)
parser.add_argument("--root-z", type=float, default=0.15)
parser.add_argument("--payload-z", type=float, default=0.23)
parser.add_argument("--camera-eye", type=float, nargs=3, default=(10.0, -13.0, 9.0))
parser.add_argument("--camera-lookat", type=float, nargs=3, default=(0.0, 0.0, 0.15))
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

import omni.usd  # noqa: E402
import insectoid_mini_quad_rl  # noqa: F401, E402
from insectoid_mini_quad_rl.articulation import INSECTOID_MINI_QUAD_CFG  # noqa: E402


@dataclass(frozen=True)
class MotionSegment:
    name: str
    duration_s: float
    local_velocity_xy: tuple[float, float] = (0.0, 0.0)
    yaw_rate: float = 0.0
    reference: str = "hold"


@dataclass(frozen=True)
class PayloadSpec:
    kind: str
    size: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class TeamSpec:
    name: str
    start_xy: tuple[float, float]
    start_yaw: float
    payload: PayloadSpec
    slot_half_x: float
    slot_side_y: float
    segments: tuple[MotionSegment, ...]


@dataclass(frozen=True)
class CarrierSlot:
    team_index: int
    slot_name: str
    local_offset_xy: tuple[float, float]
    yaw_offset: float
    opposite_side: bool
    phase_offset: int


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def rot2(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(((c, -s), (s, c)), dtype=np.float32)


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
    return min(smoothstep01(t / ramp_s), smoothstep01((segment_s - t) / ramp_s))


def build_turn_primitive(default_q: np.ndarray, direction: float, cycle_len: int = 64) -> np.ndarray:
    sequence = np.tile(default_q[None, :], (cycle_len, 1)).astype(np.float32)
    leg_phase_offsets = np.array((0.0, math.pi, math.pi, 0.0), dtype=np.float32)
    side_signs = np.array((-1.0, 1.0, -1.0, 1.0), dtype=np.float32)
    coxa_limits = (
        np.array((-0.72, -0.62, 0.30, 0.30), dtype=np.float32),
        np.array((-0.08, 0.10, 0.82, 0.92), dtype=np.float32),
    )
    femur_limits = (-0.88, -0.02)
    tibia_limits = (1.32, 2.10)
    for index in range(cycle_len):
        phase = 2.0 * math.pi * index / cycle_len
        q = default_q.copy()
        for leg_index, leg_phase_offset in enumerate(leg_phase_offsets):
            leg_phase = phase + float(leg_phase_offset)
            coxa_sweep = math.sin(leg_phase)
            lift = max(0.0, math.sin(leg_phase))
            q[leg_index] += direction * side_signs[leg_index] * args.turn_coxa_amp * coxa_sweep
            q[4 + leg_index] -= args.turn_femur_lift * lift
            q[8 + leg_index] += args.turn_tibia_lift * lift
        q[:4] = np.clip(q[:4], coxa_limits[0], coxa_limits[1])
        q[4:8] = np.clip(q[4:8], femur_limits[0], femur_limits[1])
        q[8:12] = np.clip(q[8:12], tibia_limits[0], tibia_limits[1])
        sequence[index] = q
    return sequence


def build_joint_primitives(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    time_s = data["time_s"].astype(np.float32)
    joint_pos = data["actual_joint_pos"].astype(np.float32)
    default_q = data["default_joint_pos"].astype(np.float32)

    def window(start_s: float, end_s: float, reverse: bool = False) -> np.ndarray:
        mask = (time_s >= start_s) & (time_s < end_s)
        if not np.any(mask):
            raise ValueError(f"No trajectory samples in [{start_s}, {end_s}).")
        indices = np.flatnonzero(mask)
        if reverse:
            indices = indices[::-1]
        return joint_pos[indices].astype(np.float32)

    forward = window(0.72, 1.30)
    backward = window(1.30, 1.88, reverse=True)
    # Sideways primitives are visual gait stand-ins in this large overview render.
    return {
        "hold": default_q[None, :].astype(np.float32),
        "forward": forward,
        "backward": backward,
        "xneg": forward,
        "xpos": forward,
        "turn_left": build_turn_primitive(default_q, 1.0),
        "turn_right": build_turn_primitive(default_q, -1.0),
    }


def mapped_primitive(reference_name: str, opposite_side: bool) -> str:
    if reference_name == "hold":
        return "hold"
    if not opposite_side:
        return {
            "forward": "forward",
            "backward": "backward",
            "left": "xneg",
            "right": "xpos",
            "turn_left": "turn_left",
            "turn_right": "turn_right",
        }[reference_name]
    return {
        "forward": "backward",
        "backward": "forward",
        "left": "xpos",
        "right": "xneg",
        "turn_left": "turn_left",
        "turn_right": "turn_right",
    }[reference_name]


def add_cube(parent_path: str, name: str, size: tuple[float, float, float], color: tuple[float, float, float], translate=(0.0, 0.0, 0.0)) -> None:
    cube = UsdGeom.Cube.Define(omni.usd.get_context().get_stage(), f"{parent_path}/{name}")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([color])
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetScale(Gf.Vec3f(*size))
    xform.SetTranslate(Gf.Vec3d(*translate))


def create_payload(payload_path: str, spec: PayloadSpec) -> UsdGeom.XformCommonAPI:
    stage = omni.usd.get_context().get_stage()
    root = UsdGeom.Xform.Define(stage, payload_path)
    api = UsdGeom.XformCommonAPI(root)
    if spec.kind == "pipe":
        cyl = UsdGeom.Cylinder.Define(stage, f"{payload_path}/pipe")
        cyl.CreateRadiusAttr(spec.size[1] * 0.5)
        cyl.CreateHeightAttr(spec.size[0])
        cyl.CreateDisplayColorAttr([spec.color])
        child_api = UsdGeom.XformCommonAPI(cyl)
        child_api.SetRotate((0.0, 90.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    elif spec.kind == "l_frame":
        add_cube(payload_path, "long_member", (spec.size[0], 0.16, spec.size[2]), spec.color, (0.0, 0.0, 0.0))
        add_cube(payload_path, "cross_member", (0.16, spec.size[1], spec.size[2]), spec.color, (-0.48 * spec.size[0], 0.25 * spec.size[1], 0.0))
    else:
        add_cube(payload_path, "body", spec.size, spec.color)
    return api


def set_payload_pose(api: UsdGeom.XformCommonAPI, xy: np.ndarray, yaw: float, z: float) -> None:
    api.SetTranslate(Gf.Vec3d(float(xy[0]), float(xy[1]), float(z)))
    api.SetRotate((0.0, 0.0, math.degrees(yaw)), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def make_segments(*segments: MotionSegment) -> tuple[MotionSegment, ...]:
    return segments


def team_specs() -> list[TeamSpec]:
    hold = MotionSegment("stage", 1.0)
    pause = lambda name: MotionSegment(name, 0.7)
    return [
        TeamSpec(
            name="beam",
            start_xy=(-5.0, -4.0),
            start_yaw=0.0,
            payload=PayloadSpec("beam", (2.1, 0.34, 0.16), (0.82, 0.72, 0.48)),
            slot_half_x=0.78,
            slot_side_y=0.52,
            segments=make_segments(
                hold,
                MotionSegment("beam_forward", 4.0, (0.0, 0.51), reference="forward"),
                pause("beam_hold_a"),
                MotionSegment("beam_side_shift", 3.0, (0.38, 0.0), reference="right"),
                pause("beam_hold_b"),
                MotionSegment("beam_reverse", 3.0, (0.0, -0.42), reference="backward"),
            ),
        ),
        TeamSpec(
            name="crate",
            start_xy=(4.2, -4.2),
            start_yaw=math.radians(-18.0),
            payload=PayloadSpec("crate", (0.86, 0.86, 0.42), (0.52, 0.67, 0.80)),
            slot_half_x=0.48,
            slot_side_y=0.52,
            segments=make_segments(
                hold,
                MotionSegment("crate_forward", 3.4, (0.0, 0.45), reference="forward"),
                pause("crate_hold_a"),
                MotionSegment("crate_left", 2.8, (-0.34, 0.0), reference="left"),
                pause("crate_hold_b"),
                MotionSegment("crate_forward_finish", 3.0, (0.0, 0.42), reference="forward"),
            ),
        ),
        TeamSpec(
            name="panel",
            start_xy=(-5.0, 2.6),
            start_yaw=-math.pi / 2.0,
            payload=PayloadSpec("panel", (1.55, 1.05, 0.10), (0.78, 0.78, 0.70)),
            slot_half_x=0.68,
            slot_side_y=0.66,
            segments=make_segments(
                hold,
                MotionSegment("panel_east", 3.2, (0.0, 0.48), reference="forward"),
                pause("panel_hold_a"),
                MotionSegment("panel_rotate", 2.8, (0.0, 0.0), yaw_rate=(math.pi / 2.0) / 2.8, reference="turn_left"),
                pause("panel_hold_b"),
                MotionSegment("panel_north", 3.4, (0.0, 0.44), reference="forward"),
            ),
        ),
        TeamSpec(
            name="pipe",
            start_xy=(4.8, 3.4),
            start_yaw=math.pi,
            payload=PayloadSpec("pipe", (1.8, 0.20, 0.20), (0.62, 0.62, 0.66)),
            slot_half_x=0.72,
            slot_side_y=0.42,
            segments=make_segments(
                hold,
                MotionSegment("pipe_south", 3.5, (0.0, 0.47), reference="forward"),
                pause("pipe_hold_a"),
                MotionSegment("pipe_sidestep", 3.0, (-0.36, 0.0), reference="left"),
                pause("pipe_hold_b"),
                MotionSegment("pipe_return", 2.6, (0.0, -0.38), reference="backward"),
            ),
        ),
        TeamSpec(
            name="l_frame",
            start_xy=(0.0, -5.3),
            start_yaw=0.0,
            payload=PayloadSpec("l_frame", (1.35, 0.95, 0.16), (0.73, 0.58, 0.44)),
            slot_half_x=0.58,
            slot_side_y=0.56,
            segments=make_segments(
                hold,
                MotionSegment("frame_forward", 3.0, (0.0, 0.46), reference="forward"),
                pause("frame_hold_a"),
                MotionSegment("frame_lane_change", 2.8, (0.35, 0.0), reference="right"),
                pause("frame_hold_b"),
                MotionSegment("frame_rotate", 2.6, (0.0, 0.0), yaw_rate=-(math.pi / 3.0) / 2.6, reference="turn_right"),
                pause("frame_hold_c"),
                MotionSegment("frame_final", 2.6, (0.0, 0.42), reference="forward"),
            ),
        ),
    ]


def main() -> None:
    data = np.load(args.trajectory_npz, allow_pickle=False)
    primitives = build_joint_primitives(data)
    static_q = data["default_joint_pos"].astype(np.float32)

    sim_cfg = sim_utils.SimulationCfg(dt=min(1.0 / args.fps, 0.02), render_interval=1, device=args.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(args.camera_eye), target=list(args.camera_lookat))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3600.0, color=(0.78, 0.78, 0.76))
    light_cfg.func("/World/Light", light_cfg)
    sim_utils.create_prim("/World/CarrierRobots", "Xform")
    sim_utils.create_prim("/World/Payloads", "Xform")

    teams = team_specs()
    payload_apis = []
    slots: list[CarrierSlot] = []
    origin_paths = []
    robot_paths = []
    for team_index, team in enumerate(teams):
        payload_path = f"/World/Payloads/payload_{team_index:02d}_{team.name}"
        payload_apis.append(create_payload(payload_path, team.payload))
        local_slots = (
            ("near_left", (-team.slot_half_x, -team.slot_side_y), 0.0, False, 0),
            ("near_right", (team.slot_half_x, -team.slot_side_y), 0.0, False, 4),
            ("far_left", (-team.slot_half_x, team.slot_side_y), math.pi, True, 4),
            ("far_right", (team.slot_half_x, team.slot_side_y), math.pi, True, 0),
        )
        for slot_index, (slot_name, offset, yaw_offset, opposite_side, phase_offset) in enumerate(local_slots):
            origin_path = f"/World/CarrierRobots/team_{team_index:02d}_{team.name}/robot_{slot_index:02d}_{slot_name}"
            robot_path = f"{origin_path}/Robot"
            sim_utils.create_prim(origin_path, "Xform")
            origin_paths.append(origin_path)
            robot_paths.append(robot_path)
            slots.append(CarrierSlot(team_index, slot_name, offset, yaw_offset, opposite_side, phase_offset))

    robot_cfg = INSECTOID_MINI_QUAD_CFG.replace(prim_path="/World/CarrierRobots/team_.*/robot_.*/Robot")
    robots = LabArticulation(cfg=robot_cfg)

    if not args.no_video:
        if rep is None:
            raise RuntimeError("Replicator is not available; rerun without --no-video only when camera extensions are loaded.")
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        camera = rep.create.camera(position=tuple(args.camera_eye), look_at=tuple(args.camera_lookat), focal_length=args.camera_focal_length)
        render_product = rep.create.render_product(camera, resolution=(args.width, args.height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annotator.attach(render_product)
    else:
        rgb_annotator = None

    sim.reset()
    if robots.data.default_joint_pos.shape[0] != len(slots):
        raise RuntimeError(f"Expected {len(slots)} robots, got {robots.data.default_joint_pos.shape[0]}.")
    joint_ids, resolved_joint_names = robots.find_joints(list(JOINT_NAMES), preserve_order=True)
    if tuple(resolved_joint_names) != JOINT_NAMES:
        raise RuntimeError(f"Unexpected joint resolution order {resolved_joint_names}; expected {JOINT_NAMES}.")
    joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=sim.device)

    team_xy = np.array([team.start_xy for team in teams], dtype=np.float32)
    team_yaw = np.array([team.start_yaw for team in teams], dtype=np.float32)
    team_segment_index = np.zeros(len(teams), dtype=np.int32)
    team_segment_frame = np.zeros(len(teams), dtype=np.int32)
    segment_frames = [[max(1, int(round(segment.duration_s * args.fps))) for segment in team.segments] for team in teams]
    frame_dt = 1.0 / args.fps
    total_frames = int(args.duration * args.fps)
    frames = []
    payload_traces = [[] for _ in teams]
    primitive_counts = {name: 0 for name in primitives}

    def current_segment(team_index: int) -> MotionSegment:
        team = teams[team_index]
        index = min(int(team_segment_index[team_index]), len(team.segments) - 1)
        return team.segments[index]

    def current_motion_weight(team_index: int) -> float:
        segment = current_segment(team_index)
        if segment.reference == "hold":
            return 0.0
        frames_in_segment = segment_frames[team_index][min(int(team_segment_index[team_index]), len(segment_frames[team_index]) - 1)]
        return segment_motion_weight(int(team_segment_frame[team_index]), frames_in_segment, frame_dt, args.gait_ramp_s)

    def primitive_sample(name: str, local_frame: int, phase_offset: int) -> np.ndarray:
        primitive = primitives[name]
        if name == "hold":
            return static_q.copy()
        speed = args.playback_speed * (args.turn_playback_multiplier if name.startswith("turn_") else 1.0)
        sequence_index = (int(local_frame * speed) + phase_offset) % primitive.shape[0]
        return primitive[sequence_index].copy()

    def build_robot_state() -> tuple[np.ndarray, np.ndarray]:
        root_pose = torch.zeros((len(slots), 7), dtype=torch.float32, device=sim.device)
        q_batch = np.zeros((len(slots), len(JOINT_NAMES)), dtype=np.float32)
        for robot_index, slot in enumerate(slots):
            team_index = slot.team_index
            yaw = float(team_yaw[team_index])
            world_offset = rot2(yaw) @ np.asarray(slot.local_offset_xy, dtype=np.float32)
            root_pose[robot_index, 0] = float(team_xy[team_index, 0] + world_offset[0])
            root_pose[robot_index, 1] = float(team_xy[team_index, 1] + world_offset[1])
            root_pose[robot_index, 2] = args.root_z
            root_pose[robot_index, 3:7] = torch.tensor(
                yaw_to_quat_wxyz(yaw + slot.yaw_offset),
                dtype=torch.float32,
                device=sim.device,
            )

            segment = current_segment(team_index)
            weight = current_motion_weight(team_index)
            reference = segment.reference if weight > 1.0e-4 else "hold"
            primitive_name = mapped_primitive(reference, slot.opposite_side)
            primitive_counts[primitive_name] += 1
            sample = primitive_sample(primitive_name, int(team_segment_frame[team_index]), slot.phase_offset)
            q_batch[robot_index] = static_q + weight * (sample - static_q)
        return root_pose, q_batch

    def write_scene() -> None:
        root_pose, q_batch_np = build_robot_state()
        q_batch = torch.as_tensor(q_batch_np, dtype=torch.float32, device=sim.device)
        robots.write_root_pose_to_sim(root_pose)
        robots.write_root_velocity_to_sim(torch.zeros((len(slots), 6), dtype=torch.float32, device=sim.device))
        robots.write_joint_state_to_sim(q_batch, torch.zeros_like(q_batch), joint_ids=joint_ids_tensor)
        robots.set_joint_position_target(q_batch, joint_ids=joint_ids_tensor)
        robots.write_data_to_sim()
        for team_index, api in enumerate(payload_apis):
            set_payload_pose(api, team_xy[team_index], float(team_yaw[team_index]), args.payload_z)
            payload_traces[team_index].append([float(team_xy[team_index, 0]), float(team_xy[team_index, 1]), args.payload_z])

    def advance_team_state() -> None:
        for team_index, team in enumerate(teams):
            segment = current_segment(team_index)
            weight = current_motion_weight(team_index)
            local_velocity = np.asarray(segment.local_velocity_xy, dtype=np.float32) * weight
            team_xy[team_index] += (rot2(float(team_yaw[team_index])) @ local_velocity) * frame_dt
            team_yaw[team_index] += float(segment.yaw_rate) * weight * frame_dt
            team_segment_frame[team_index] += 1
            segment_index = int(team_segment_index[team_index])
            if team_segment_frame[team_index] >= segment_frames[team_index][segment_index]:
                if segment_index < len(team.segments) - 1:
                    team_segment_index[team_index] += 1
                    team_segment_frame[team_index] = 0

    robots.reset()
    for _ in range(3):
        write_scene()
        sim.step(render=False)
        robots.update(sim.get_physics_dt())

    for _ in range(total_frames):
        write_scene()
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
                frames.append(np.ascontiguousarray(rgb))
        advance_team_state()

    payload_traces_np = [np.asarray(trace, dtype=np.float32) for trace in payload_traces]
    summary = {
        "type": "multi_object_cooperative_handling_yard",
        "render_mode": "kinematic_multi_team_payload_choreography",
        "robot_count": len(slots),
        "team_count": len(teams),
        "payload_count": len(teams),
        "duration_s": total_frames * frame_dt,
        "fps": args.fps,
        "playback_speed": args.playback_speed,
        "gait_ramp_s": args.gait_ramp_s,
        "turn_coxa_amp": args.turn_coxa_amp,
        "turn_femur_lift": args.turn_femur_lift,
        "turn_tibia_lift": args.turn_tibia_lift,
        "turn_playback_multiplier": args.turn_playback_multiplier,
        "camera_eye": list(args.camera_eye),
        "camera_lookat": list(args.camera_lookat),
        "camera_focal_length": args.camera_focal_length,
        "field_extent_hint_m": [-7.0, 7.0, -6.5, 6.5],
        "payloads": [
            {
                "name": team.name,
                "kind": team.payload.kind,
                "size_m": list(team.payload.size),
                "start_xy_m": list(team.start_xy),
                "final_xy_m": payload_traces_np[index][-1, :2].tolist(),
                "displacement_xy_m": (payload_traces_np[index][-1, :2] - payload_traces_np[index][0, :2]).tolist(),
                "segments": [
                    {
                        "name": segment.name,
                        "duration_s": segment.duration_s,
                        "local_velocity_xy_mps": list(segment.local_velocity_xy),
                        "yaw_rate_radps": segment.yaw_rate,
                        "reference": segment.reference,
                    }
                    for segment in team.segments
                ],
            }
            for index, team in enumerate(teams)
        ],
        "primitive_counts": primitive_counts,
        "robot_usd": str(ROBOT_USD_PATH),
        "trajectory_npz": str(args.trajectory_npz),
        "notes": [
            "Large-field presentation render with five independent four-robot payload teams.",
            "Payloads are kinematic visual objects; this is a cooperative handling choreography, not payload-contact physics.",
            "Each team uses a different object geometry and scripted path to show heterogeneous material handling objectives.",
            "Moving segments use shared gait/root ramps so legs stop during holds and ramp before direction changes.",
            "Rotational segments use a procedural turn primitive with side-opposed coxa swing and diagonal lift timing.",
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
    print(f"TEAM_COUNT={len(teams)} ROBOT_COUNT={len(slots)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[ERROR]: Multi-object yard render aborted with {type(exc).__name__}: {exc}", flush=True)
        raise
