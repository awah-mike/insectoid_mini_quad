#!/usr/bin/env python3
"""Render a headless policy rollout video for the insectoid mini quad."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_vec3(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated floats, for example: 2.4,-2.4,2.6")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated floats") from exc


parser = argparse.ArgumentParser(description="Render an insectoid mini quad policy rollout to MP4.")
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=REPO_ROOT / "trained_models/forward_strict_best/model_1399.pt",
)
parser.add_argument(
    "--output",
    type=Path,
    default=REPO_ROOT / "outputs/training_rollouts/forward_strict_best/model_1399_rollout.mp4",
)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--forward-vel", type=float, default=0.30)
parser.add_argument("--video-length", type=int, default=300, help="Number of simulation steps to record.")
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--camera-eye", type=parse_vec3, default=(2.4, -2.4, 2.6))
parser.add_argument("--camera-lookat", type=parse_vec3, default=(0.0, 0.0, 0.12))
parser.add_argument(
    "--disable-observation-noise",
    action="store_true",
    help="Disable task observation noise for deterministic policy visualization.",
)
parser.add_argument(
    "--base-lin-vel-observation",
    choices=("task", "privileged", "kinematic", "zero"),
    default="task",
    help="Override the policy obs[0:3] source for deployment debugging.",
)
parser.add_argument(
    "--post-bl-tibia-max-rad",
    type=float,
    default=None,
    help="Optional rollout-only cap for BL_tibia_joint target position.",
)
parser.add_argument(
    "--post-bl-tibia-br-offset-rad",
    type=float,
    default=None,
    help="Optional rollout-only cap: BL_tibia_joint <= BR_tibia_joint + offset.",
)
parser.add_argument(
    "--post-bl-tibia-blend",
    type=float,
    default=1.0,
    help="Blend factor for optional BL tibia target correction.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import torch
from rsl_rl.runners import OnPolicyRunner

import insectoid_mini_quad_rl  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.lin_vel_x_range = (0.0, 0.0)
    env_cfg.lin_vel_y_range = (args.forward_vel, args.forward_vel)
    env_cfg.ang_vel_z_range = (0.0, 0.0)
    env_cfg.rel_standing_envs = 0.0
    if args.disable_observation_noise and hasattr(env_cfg, "observation_noise_enabled"):
        env_cfg.observation_noise_enabled = False
    if args.base_lin_vel_observation != "task":
        env_cfg.use_privileged_base_lin_vel_obs = args.base_lin_vel_observation == "privileged"
        env_cfg.use_foot_kinematic_base_lin_vel_obs = args.base_lin_vel_observation == "kinematic"
    if hasattr(env_cfg, "events"):
        env_cfg.events.physics_material = None
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = args.camera_eye
        env_cfg.viewer.lookat = args.camera_lookat
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.resolution = (args.width, args.height)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device

    raw_env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[VIDEO] loading checkpoint: {args.checkpoint}", flush=True)
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    raw = raw_env.unwrapped
    joint_names = tuple(raw._robot.data.joint_names)
    bl_tibia_index = joint_names.index("BL_tibia_joint")
    br_tibia_index = joint_names.index("BR_tibia_joint")
    action_scale = float(raw.cfg.action_scale)
    default_joint_pos = raw._robot.data.default_joint_pos
    post_bl_tibia_blend = max(0.0, min(1.0, args.post_bl_tibia_blend))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps)
    try:
        for step in range(args.video_length):
            with torch.inference_mode():
                raw._commands[:, 0] = 0.0
                raw._commands[:, 1] = args.forward_vel
                raw._commands[:, 2] = 0.0
                actions = policy(obs)
                if args.post_bl_tibia_max_rad is not None or args.post_bl_tibia_br_offset_rad is not None:
                    target = default_joint_pos + action_scale * actions
                    bl_limit = torch.full_like(target[:, bl_tibia_index], float("inf"))
                    if args.post_bl_tibia_max_rad is not None:
                        bl_limit = torch.minimum(
                            bl_limit,
                            torch.full_like(bl_limit, args.post_bl_tibia_max_rad),
                        )
                    if args.post_bl_tibia_br_offset_rad is not None:
                        bl_limit = torch.minimum(
                            bl_limit,
                            target[:, br_tibia_index] + args.post_bl_tibia_br_offset_rad,
                        )
                    corrected_bl = torch.minimum(target[:, bl_tibia_index], bl_limit)
                    target[:, bl_tibia_index] = (
                        (1.0 - post_bl_tibia_blend) * target[:, bl_tibia_index]
                        + post_bl_tibia_blend * corrected_bl
                    )
                    actions = torch.clamp((target - default_joint_pos) / action_scale, -1.0, 1.0)
                obs, _, _, _ = env.step(actions)

            frame = raw_env.render()
            writer.append_data(frame)

            if (step + 1) % 100 == 0:
                vel_y = raw_env.unwrapped._robot.data.root_lin_vel_b[:, 1].mean().item()
                print(f"[VIDEO] step={step + 1} mean_vel_y={vel_y:.3f}", flush=True)
    finally:
        writer.close()
        env.close()

    print(f"VIDEO={args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
