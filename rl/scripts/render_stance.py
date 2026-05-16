#!/usr/bin/env python3
"""Capture a stance snapshot and video for the insectoid mini quad."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0")
parser.add_argument("--duration", type=float, default=5.0, help="Seconds of zero-action stance simulation to record.")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--snapshot-step", type=int, default=-1, help="Frame index to save as PNG; -1 saves the final frame.")
parser.add_argument("--output-dir", type=Path, default=Path("/workspace/insectoid_mini_quad_rl/outputs/stance"))
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--eye", type=float, nargs=3, default=(0.75, 1.25, 0.55))
parser.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.03, 0.12))
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


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.sim.device = args.device
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.viewer.eye = tuple(args.eye)
    env_cfg.viewer.lookat = tuple(args.lookat)
    env_cfg.viewer.origin_type = "world"

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    obs, _ = env.reset()
    del obs

    unwrapped = env.unwrapped
    joint_pos = unwrapped._robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(unwrapped._robot.data.default_joint_vel)
    root_state = unwrapped._robot.data.default_root_state.clone()
    root_state[:, :3] += unwrapped._terrain.env_origins
    unwrapped._robot.write_root_pose_to_sim(root_state[:, :7])
    unwrapped._robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    unwrapped._robot.write_joint_state_to_sim(joint_pos, joint_vel)
    unwrapped._actions.zero_()
    unwrapped._previous_actions.zero_()
    unwrapped._processed_actions[:] = joint_pos

    action_dim = unwrapped.single_action_space.shape[0]
    actions = torch.zeros(unwrapped.num_envs, action_dim, device=unwrapped.device)
    total_steps = max(1, int(args.duration / unwrapped.step_dt))
    capture_every = max(1, round(1.0 / (args.fps * unwrapped.step_dt)))
    frames = []

    for step in range(total_steps):
        with torch.inference_mode():
            env.step(actions)
        if step % capture_every == 0 or step == total_steps - 1:
            frame = env.render()
            if frame is not None and frame.size:
                frames.append(frame)

    if not frames:
        raise RuntimeError("No render frames were captured. Make sure cameras are enabled.")

    snapshot_index = args.snapshot_step if args.snapshot_step >= 0 else len(frames) - 1
    snapshot_index = min(snapshot_index, len(frames) - 1)
    initial_snapshot_path = args.output_dir / "insectoid_mini_quad_stance_initial.png"
    final_snapshot_path = args.output_dir / "insectoid_mini_quad_stance_final.png"
    video_path = args.output_dir / "insectoid_mini_quad_stance.mp4"
    first_visible = next((frame for frame in frames if np.mean(frame) > 2.0), frames[0])
    imageio.imwrite(initial_snapshot_path, first_visible)
    imageio.imwrite(final_snapshot_path, frames[snapshot_index])
    imageio.mimsave(video_path, frames, fps=args.fps)

    base_pos = unwrapped._robot.data.root_pos_w[0].detach().cpu().tolist()
    gravity_xy_norm = torch.norm(unwrapped._robot.data.projected_gravity_b[0, :2]).item()
    foot_contacts = unwrapped._get_foot_contacts()[0].detach().cpu().tolist()
    front_contact_forces = unwrapped._contact_sensor.data.net_forces_w_history[
        0, :, unwrapped._undesired_contact_body_ids
    ]
    max_front_or_body_contact_n = torch.amax(torch.norm(front_contact_forces, dim=-1)).item()

    print(f"INITIAL_SNAPSHOT={initial_snapshot_path}", flush=True)
    print(f"FINAL_SNAPSHOT={final_snapshot_path}", flush=True)
    print(f"VIDEO={video_path}", flush=True)
    print(f"FRAMES={len(frames)} FPS={args.fps} DURATION_S={args.duration}", flush=True)
    print(f"BASE_POS_W={base_pos}", flush=True)
    print(f"PROJECTED_GRAVITY_XY_NORM={gravity_xy_norm:.6f}", flush=True)
    print(f"FOOT_CONTACTS_BL_BR_ML_MR={foot_contacts}", flush=True)
    print(f"MAX_NON_FOOT_CONTACT_N={max_front_or_body_contact_n:.6f}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
