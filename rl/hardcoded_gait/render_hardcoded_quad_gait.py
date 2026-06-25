#!/usr/bin/env python3
"""Render an open-loop quadruped foot-line walk for the insectoid mini quad.

This is intentionally separate from the RL environment/training setup. It uses
the current robot asset and standing pose, then drives the four walking legs with
an analytic one-leg-at-a-time crawl gait in foot space.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0")
parser.add_argument("--output-dir", type=Path, default=Path("/workspace/insectoid_mini_quad_rl/hardcoded_gait/outputs"))
parser.add_argument("--duration", type=float, default=8.0)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--cycle-time", type=float, default=2.4)
parser.add_argument("--swing-fraction", type=float, default=0.18)
parser.add_argument("--step-length", type=float, default=0.10, help="World-Y step line length in meters.")
parser.add_argument("--step-height", type=float, default=0.03, help="Swing foot clearance in meters.")
parser.add_argument("--ik-gain", type=float, default=0.7)
parser.add_argument("--ik-damping", type=float, default=0.04)
parser.add_argument("--max-joint-step-deg", type=float, default=4.0)
parser.add_argument("--settle-time", type=float, default=0.75)
parser.add_argument("--static-friction", type=float, default=1.0)
parser.add_argument("--dynamic-friction", type=float, default=1.0)
parser.add_argument("--summary-output", type=Path, default=None)
parser.add_argument("--eye", type=float, nargs=3, default=(2.6, -2.2, 1.25))
parser.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.0, 0.13))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import insectoid_mini_quad_rl  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


LEGS = ("BL", "BR", "ML", "MR")
# One-leg-at-a-time crawl sequence. The order alternates sides and rear/middle
# support so that the robot is never intentionally left on a diagonal pair only.
WALK_PHASE_OFFSETS = {
    "BR": 0.00,
    "ML": 0.25,
    "BL": 0.50,
    "MR": 0.75,
}


def _smoothstep(x: torch.Tensor | float) -> torch.Tensor | float:
    return x * x * (3.0 - 2.0 * x)


def _reset_to_default(raw) -> None:
    joint_pos = raw._robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(raw._robot.data.default_joint_vel)
    root_state = raw._robot.data.default_root_state.clone()
    root_state[:, :3] += raw._terrain.env_origins
    root_state[:, 2] += 0.02
    raw._robot.write_root_pose_to_sim(root_state[:, :7])
    raw._robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    raw._robot.write_joint_state_to_sim(joint_pos, joint_vel)
    raw._actions.zero_()
    raw._previous_actions.zero_()
    raw._processed_actions[:] = joint_pos


def _build_line_targets(t: float, foot_anchors_w: torch.Tensor, active_swing: dict[str, bool]) -> torch.Tensor:
    foot_targets_w = foot_anchors_w.clone()
    for foot_index, leg in enumerate(LEGS):
        phase = ((t / args.cycle_time) + WALK_PHASE_OFFSETS[leg]) % 1.0
        if phase < args.swing_fraction:
            swing_u = phase / args.swing_fraction
            active_swing[leg] = True
            foot_targets_w[:, foot_index, 1] = foot_anchors_w[:, foot_index, 1] + args.step_length * _smoothstep(swing_u)
            foot_targets_w[:, foot_index, 2] = foot_anchors_w[:, foot_index, 2] + args.step_height * math.sin(math.pi * swing_u)
        else:
            if active_swing[leg]:
                foot_anchors_w[:, foot_index, 1] += args.step_length
                active_swing[leg] = False
            foot_targets_w[:, foot_index, :] = foot_anchors_w[:, foot_index, :]
    return foot_targets_w


def _ik_track_foot_targets(
    raw,
    foot_targets_w: torch.Tensor,
    foot_ids: list[int],
    joint_ids_by_leg: dict[str, torch.Tensor],
    q_command: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    jacobians = raw._robot.root_physx_view.get_jacobians()
    foot_pos_w = raw._robot.data.body_pos_w[:, foot_ids, :]
    next_q = q_command.clone()
    max_joint_step = math.radians(args.max_joint_step_deg)

    for foot_index, leg in enumerate(LEGS):
        joint_ids = joint_ids_by_leg[leg]
        jacobi_idx = foot_ids[foot_index] - 1
        jacobian = jacobians[:, jacobi_idx, :3, :][:, :, joint_ids]
        error = (foot_targets_w[:, foot_index, :] - foot_pos_w[:, foot_index, :]).unsqueeze(-1)
        j_t = torch.transpose(jacobian, 1, 2)
        identity = torch.eye(3, device=raw.device).unsqueeze(0)
        solve_mat = torch.bmm(jacobian, j_t) + (args.ik_damping**2) * identity
        dq = torch.bmm(j_t, torch.linalg.solve(solve_mat, error)).squeeze(-1)
        dq = torch.clamp(args.ik_gain * dq, -max_joint_step, max_joint_step)
        next_q[:, joint_ids] += dq

    next_q = torch.maximum(torch.minimum(next_q, upper_limits), lower_limits)
    return next_q


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.sim.device = args.device
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.viewer.eye = tuple(args.eye)
    env_cfg.viewer.lookat = tuple(args.lookat)
    env_cfg.viewer.origin_type = "world"
    env_cfg.sim.physics_material.static_friction = args.static_friction
    env_cfg.sim.physics_material.dynamic_friction = args.dynamic_friction
    env_cfg.terrain.physics_material.static_friction = args.static_friction
    env_cfg.terrain.physics_material.dynamic_friction = args.dynamic_friction
    if hasattr(env_cfg, "events") and env_cfg.events.physics_material is not None:
        env_cfg.events.physics_material.params["static_friction_range"] = (args.static_friction, args.static_friction)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (args.dynamic_friction, args.dynamic_friction)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    raw = env.unwrapped
    _reset_to_default(raw)

    joint_names = tuple(raw._robot.data.joint_names)
    joint_index_by_name = {name: index for index, name in enumerate(joint_names)}
    foot_names = tuple(f"{leg}_FOOT" for leg in LEGS)
    foot_ids, found_foot_names = raw._robot.find_bodies("|".join(foot_names), preserve_order=True)
    if tuple(found_foot_names) != foot_names:
        raise RuntimeError(f"Unexpected foot order {found_foot_names}; expected {foot_names}.")
    joint_ids_by_leg = {
        leg: torch.tensor(
            [
                joint_index_by_name[f"{leg}_coxa_joint"],
                joint_index_by_name[f"{leg}_femur_joint"],
                joint_index_by_name[f"{leg}_tibia_joint"],
            ],
            dtype=torch.long,
            device=raw.device,
        )
        for leg in LEGS
    }

    action_scale = raw.cfg.action_scale
    default_joint_pos = raw._robot.data.default_joint_pos.clone()
    lower_limits = raw._robot.data.soft_joint_pos_limits[:, :, 0]
    upper_limits = raw._robot.data.soft_joint_pos_limits[:, :, 1]
    action_dim = raw.single_action_space.shape[0]
    frames = []

    settle_steps = max(0, int(args.settle_time / raw.step_dt))
    total_steps = max(1, int(args.duration / raw.step_dt))
    capture_every = max(1, round(1.0 / (args.fps * raw.step_dt)))

    zero_actions = torch.zeros(raw.num_envs, action_dim, device=raw.device)
    for _ in range(settle_steps):
        with torch.inference_mode():
            env.step(zero_actions)

    raw._robot.update(raw.step_dt)
    foot_anchors_w = raw._robot.data.body_pos_w[:, foot_ids, :].clone()
    active_swing = {leg: False for leg in LEGS}
    q_command = raw._robot.data.joint_pos.clone()
    initial_base_pos = raw._robot.data.root_pos_w[0].detach().cpu().numpy().copy()
    root_positions = []
    projected_gravity_xy_norms = []

    for step in range(total_steps):
        t = step * raw.step_dt
        foot_targets_w = _build_line_targets(t, foot_anchors_w, active_swing)
        target = _ik_track_foot_targets(
            raw,
            foot_targets_w,
            foot_ids,
            joint_ids_by_leg,
            q_command,
            lower_limits,
            upper_limits,
        )
        q_command = target.clone()
        actions = torch.clamp((target - default_joint_pos) / action_scale, -1.0, 1.0)
        with torch.inference_mode():
            env.step(actions)
        root_positions.append(raw._robot.data.root_pos_w[0].detach().cpu().numpy().copy())
        projected_gravity_xy_norms.append(torch.norm(raw._robot.data.projected_gravity_b[0, :2]).item())
        if step % capture_every == 0 or step == total_steps - 1:
            frame = env.render()
            if frame is not None and frame.size:
                frames.append(frame)

    if not frames:
        raise RuntimeError("No render frames were captured. Make sure cameras are enabled.")

    video_path = args.output_dir / "hardcoded_foot_line_walk.mp4"
    snapshot_path = args.output_dir / "hardcoded_foot_line_walk_final.png"
    first_visible_path = args.output_dir / "hardcoded_foot_line_walk_initial.png"
    first_visible = next((frame for frame in frames if np.mean(frame) > 2.0), frames[0])
    imageio.imwrite(first_visible_path, first_visible)
    imageio.imwrite(snapshot_path, frames[-1])
    imageio.mimsave(video_path, frames, fps=args.fps)
    root_positions_array = np.asarray(root_positions)
    final_base_pos = raw._robot.data.root_pos_w[0].detach().cpu().numpy().copy()
    displacement = final_base_pos - initial_base_pos
    summary = {
        "video": str(video_path),
        "initial_snapshot": str(first_visible_path),
        "final_snapshot": str(snapshot_path),
        "duration_s": args.duration,
        "fps": args.fps,
        "cycle_time_s": args.cycle_time,
        "swing_fraction": args.swing_fraction,
        "step_length_m": args.step_length,
        "step_height_m": args.step_height,
        "ik_gain": args.ik_gain,
        "ik_damping": args.ik_damping,
        "max_joint_step_deg": args.max_joint_step_deg,
        "static_friction": args.static_friction,
        "dynamic_friction": args.dynamic_friction,
        "initial_base_pos_w": initial_base_pos.tolist(),
        "final_base_pos_w": final_base_pos.tolist(),
        "base_displacement_w": displacement.tolist(),
        "mean_world_vel_y_mps": float(displacement[1] / max(args.duration, 1.0e-6)),
        "min_base_z_w": float(np.min(root_positions_array[:, 2])),
        "max_projected_gravity_xy_norm": float(np.max(projected_gravity_xy_norms)),
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"VIDEO={video_path}", flush=True)
    print(f"INITIAL_SNAPSHOT={first_visible_path}", flush=True)
    print(f"FINAL_SNAPSHOT={snapshot_path}", flush=True)
    print(f"FRAMES={len(frames)} FPS={args.fps} DURATION_S={args.duration}", flush=True)
    print(f"JOINT_NAMES={joint_names}", flush=True)
    print(f"FOOT_NAMES={found_foot_names}", flush=True)
    print(f"STEP_LENGTH_M={args.step_length}", flush=True)
    print(f"STEP_HEIGHT_M={args.step_height}", flush=True)
    print(f"BASE_POS_W={final_base_pos.tolist()}", flush=True)
    print(f"BASE_DISPLACEMENT_W={displacement.tolist()}", flush=True)
    print(f"MEAN_WORLD_VEL_Y={summary['mean_world_vel_y_mps']:.4f}", flush=True)
    print(f"MIN_BASE_Z_W={summary['min_base_z_w']:.4f}", flush=True)
    print(f"PROJECTED_GRAVITY_XY_NORM={torch.norm(raw._robot.data.projected_gravity_b[0, :2]).item():.6f}", flush=True)
    print(f"MAX_PROJECTED_GRAVITY_XY_NORM={summary['max_projected_gravity_xy_norm']:.6f}", flush=True)
    if args.summary_output is not None:
        print(f"SUMMARY={args.summary_output}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
