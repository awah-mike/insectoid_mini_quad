#!/usr/bin/env python3
"""Sweep static femur support offsets and report settled front-up attitude."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV2-Direct-Play-v0")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--skip-steps", type=int, default=60)
parser.add_argument("--front-up-reset-deg", type=float, default=10.0)
parser.add_argument("--root-z-offset", type=float, default=0.08)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import insectoid_mini_quad_rl  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


VARIANTS: dict[str, dict[str, float]] = {
    "none": {},
    "v1_rear_neg_mid_pos": {
        "BL_femur_joint": -0.18,
        "BR_femur_joint": -0.18,
        "ML_femur_joint": 0.28,
        "MR_femur_joint": 0.28,
    },
    "v2_rear_neg_mid_pos": {
        "BL_femur_joint": -0.32,
        "BR_femur_joint": -0.32,
        "ML_femur_joint": 0.48,
        "MR_femur_joint": 0.48,
    },
    "inverse_rear_pos_mid_neg": {
        "BL_femur_joint": 0.32,
        "BR_femur_joint": 0.32,
        "ML_femur_joint": -0.48,
        "MR_femur_joint": -0.48,
    },
    "rear_neg_only": {
        "BL_femur_joint": -0.50,
        "BR_femur_joint": -0.50,
    },
    "rear_neg_only_020": {
        "BL_femur_joint": -0.20,
        "BR_femur_joint": -0.20,
    },
    "rear_neg_only_030": {
        "BL_femur_joint": -0.30,
        "BR_femur_joint": -0.30,
    },
    "rear_neg_only_035": {
        "BL_femur_joint": -0.35,
        "BR_femur_joint": -0.35,
    },
    "rear_neg_only_040": {
        "BL_femur_joint": -0.40,
        "BR_femur_joint": -0.40,
    },
    "rear_pos_only": {
        "BL_femur_joint": 0.50,
        "BR_femur_joint": 0.50,
    },
    "mid_pos_only": {
        "ML_femur_joint": 0.60,
        "MR_femur_joint": 0.60,
    },
    "mid_neg_only": {
        "ML_femur_joint": -0.60,
        "MR_femur_joint": -0.60,
    },
    "strong_rear_neg_mid_pos": {
        "BL_femur_joint": -0.55,
        "BR_femur_joint": -0.55,
        "ML_femur_joint": 0.75,
        "MR_femur_joint": 0.75,
    },
    "strong_rear_pos_mid_neg": {
        "BL_femur_joint": 0.55,
        "BR_femur_joint": 0.55,
        "ML_femur_joint": -0.75,
        "MR_femur_joint": -0.75,
    },
}


def set_reset_offset(raw_env, offsets: Mapping[str, float]) -> torch.Tensor:
    raw_env._reset_joint_pos_offset.zero_()
    action = torch.zeros(raw_env.num_envs, raw_env.cfg.action_space, device=raw_env.device)
    for joint_name, offset in offsets.items():
        joint_ids, _ = raw_env._robot.find_joints([joint_name], preserve_order=True)
        if len(joint_ids) != 1:
            raise RuntimeError(f"Expected one joint for {joint_name}, got {joint_ids}.")
        joint_id = joint_ids[0]
        raw_env._reset_joint_pos_offset[joint_id] = float(offset)
        action[:, joint_id] = float(offset) / raw_env.cfg.action_scale
    return action


def main() -> None:
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.front_up_reset_deg = args.front_up_reset_deg
    env_cfg.front_up_reset_root_z_offset = args.root_z_offset
    env_cfg.action_center_offsets = {}
    env_cfg.reset_joint_pos_offsets = {}
    env_cfg.front_up_posture_joint_offsets = {}
    env_cfg.lin_vel_x_range = (0.0, 0.0)
    env_cfg.lin_vel_y_range = (0.0, 0.0)
    env_cfg.ang_vel_z_range = (0.0, 0.0)
    if hasattr(env_cfg, "events"):
        env_cfg.events.physics_material = None

    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped

    print("variant,mean_front_up_deg,std_front_up_deg,min_front_up_deg,max_front_up_deg,mean_base_z,resets")
    for name, offsets in VARIANTS.items():
        action = set_reset_offset(raw, offsets)
        env.reset()
        raw.episode_length_buf.zero_()
        front_up_deg = []
        base_z = []
        resets = 0
        for step in range(args.steps):
            raw._commands[:, 0] = 0.0
            raw._commands[:, 1] = 0.0
            raw._commands[:, 2] = 0.0
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            resets += int(torch.count_nonzero(done).item())
            if step >= args.skip_steps:
                projected_gravity_y = raw._robot.data.projected_gravity_b[:, 1]
                front_up = torch.rad2deg(torch.asin(torch.clamp(-projected_gravity_y, -1.0, 1.0)))
                front_up_deg.append(front_up.detach().clone())
                base_z.append(raw._robot.data.root_pos_w[:, 2].detach().clone())
        front_up_all = torch.cat(front_up_deg)
        base_z_all = torch.cat(base_z)
        print(
            f"{name},"
            f"{front_up_all.mean().item():.3f},"
            f"{front_up_all.std().item():.3f},"
            f"{front_up_all.min().item():.3f},"
            f"{front_up_all.max().item():.3f},"
            f"{base_z_all.mean().item():.3f},"
            f"{resets}"
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
