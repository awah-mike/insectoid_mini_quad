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
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=args.fps)
    try:
        for step in range(args.video_length):
            with torch.inference_mode():
                raw = raw_env.unwrapped
                raw._commands[:, 0] = 0.0
                raw._commands[:, 1] = args.forward_vel
                raw._commands[:, 2] = 0.0
                actions = policy(obs)
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
