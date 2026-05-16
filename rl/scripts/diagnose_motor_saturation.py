#!/usr/bin/env python3
"""Diagnose motor/action saturation for an insectoid mini quad RSL-RL checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--forward-vel", type=float, default=0.30)
parser.add_argument("--output", type=Path, default=Path("/workspace/insectoid_mini_quad_rl/outputs/eval/motor_saturation.txt"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import insectoid_mini_quad_rl  # noqa: F401, E402
from insectoid_mini_quad_rl.articulation import AK45_36_CONTINUOUS_TORQUE_NM, AK45_36_PEAK_TORQUE_NM, AK45_36_SIM_VELOCITY_LIMIT_RAD_PER_S  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.sim.device = args.device
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = raw_env.unwrapped
    joint_names = tuple(raw._robot.data.joint_names)
    num_joints = len(joint_names)
    soft_limits = raw._robot.data.soft_joint_pos_limits
    lower_limits = soft_limits[:, :, 0]
    upper_limits = soft_limits[:, :, 1]
    limit_span = torch.clamp(upper_limits - lower_limits, min=1.0e-6)

    obs = env.get_observations()
    resets = 0
    samples = 0
    abs_torque_sum = torch.zeros(num_joints, device=raw.device)
    abs_torque_max = torch.zeros(num_joints, device=raw.device)
    over_cont_count = torch.zeros(num_joints, device=raw.device)
    near_peak_count = torch.zeros(num_joints, device=raw.device)
    abs_vel_sum = torch.zeros(num_joints, device=raw.device)
    abs_vel_max = torch.zeros(num_joints, device=raw.device)
    near_vel_count = torch.zeros(num_joints, device=raw.device)
    action_abs_sum = torch.zeros(num_joints, device=raw.device)
    action_abs_max = torch.zeros(num_joints, device=raw.device)
    action_sat_count = torch.zeros(num_joints, device=raw.device)
    near_joint_limit_count = torch.zeros(num_joints, device=raw.device)

    for _ in range(args.steps):
        with torch.inference_mode():
            raw._commands[:, 0] = 0.0
            raw._commands[:, 1] = args.forward_vel
            raw._commands[:, 2] = 0.0
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            resets += int(torch.count_nonzero(dones).item())

            abs_actions = torch.abs(actions[:, :num_joints])
            abs_torque = torch.abs(raw._robot.data.applied_torque)
            abs_joint_vel = torch.abs(raw._robot.data.joint_vel)
            joint_pos = raw._robot.data.joint_pos
            normalized_limit_margin = torch.minimum(joint_pos - lower_limits, upper_limits - joint_pos) / limit_span

            abs_torque_sum += torch.sum(abs_torque, dim=0)
            abs_torque_max = torch.maximum(abs_torque_max, torch.amax(abs_torque, dim=0))
            over_cont_count += torch.sum(abs_torque > AK45_36_CONTINUOUS_TORQUE_NM, dim=0)
            near_peak_count += torch.sum(abs_torque > 0.90 * AK45_36_PEAK_TORQUE_NM, dim=0)
            abs_vel_sum += torch.sum(abs_joint_vel, dim=0)
            abs_vel_max = torch.maximum(abs_vel_max, torch.amax(abs_joint_vel, dim=0))
            near_vel_count += torch.sum(abs_joint_vel > 0.90 * AK45_36_SIM_VELOCITY_LIMIT_RAD_PER_S, dim=0)
            action_abs_sum += torch.sum(abs_actions, dim=0)
            action_abs_max = torch.maximum(action_abs_max, torch.amax(abs_actions, dim=0))
            action_sat_count += torch.sum(abs_actions > 0.98, dim=0)
            near_joint_limit_count += torch.sum(normalized_limit_margin < 0.05, dim=0)
            samples += actions.shape[0]

    torque_mean = abs_torque_sum / samples
    vel_mean = abs_vel_sum / samples
    action_mean = action_abs_sum / samples
    over_cont_frac = over_cont_count / samples
    near_peak_frac = near_peak_count / samples
    near_vel_frac = near_vel_count / samples
    action_sat_frac = action_sat_count / samples
    near_joint_limit_frac = near_joint_limit_count / samples

    lines = [
        f"CHECKPOINT={args.checkpoint}",
        f"STEPS={args.steps} NUM_ENVS={args.num_envs} COMMAND_FORWARD_Y={args.forward_vel:.3f}",
        f"TORQUE_CONTINUOUS_NM={AK45_36_CONTINUOUS_TORQUE_NM:.3f}",
        f"TORQUE_PEAK_LIMIT_NM={AK45_36_PEAK_TORQUE_NM:.3f}",
        f"VELOCITY_LIMIT_RADPS={AK45_36_SIM_VELOCITY_LIMIT_RAD_PER_S:.3f}",
        f"RESETS={resets}",
        f"OVER_CONTINUOUS_ANY_FRACTION={torch.mean((over_cont_frac > 0.0).float()).item():.4f}",
        f"NEAR_PEAK_ANY_FRACTION={torch.mean((near_peak_frac > 0.0).float()).item():.4f}",
        f"ACTION_SAT_ANY_FRACTION={torch.mean((action_sat_frac > 0.0).float()).item():.4f}",
        f"MEAN_ABS_TORQUE_NM_ALL={torch.mean(torque_mean).item():.4f}",
        f"MAX_ABS_TORQUE_NM_ALL={torch.amax(abs_torque_max).item():.4f}",
        f"FRACTION_SAMPLES_OVER_8NM_ALL={torch.mean(over_cont_frac).item():.4f}",
        f"FRACTION_SAMPLES_OVER_21P6NM_ALL={torch.mean(near_peak_frac).item():.4f}",
        f"MEAN_ABS_JOINT_VEL_RADPS_ALL={torch.mean(vel_mean).item():.4f}",
        f"MAX_ABS_JOINT_VEL_RADPS_ALL={torch.amax(abs_vel_max).item():.4f}",
        f"FRACTION_SAMPLES_OVER_2P7RADPS_ALL={torch.mean(near_vel_frac).item():.4f}",
        f"MEAN_ABS_ACTION_ALL={torch.mean(action_mean).item():.4f}",
        f"MAX_ABS_ACTION_ALL={torch.amax(action_abs_max).item():.4f}",
        f"FRACTION_ACTION_ABS_GT_0P98_ALL={torch.mean(action_sat_frac).item():.4f}",
        f"FRACTION_NEAR_JOINT_LIMIT_ALL={torch.mean(near_joint_limit_frac).item():.4f}",
    ]
    for index, joint_name in enumerate(joint_names):
        lines.extend(
            [
                f"{joint_name}_MEAN_ABS_TORQUE_NM={torque_mean[index].item():.4f}",
                f"{joint_name}_MAX_ABS_TORQUE_NM={abs_torque_max[index].item():.4f}",
                f"{joint_name}_FRAC_OVER_8NM={over_cont_frac[index].item():.4f}",
                f"{joint_name}_FRAC_OVER_21P6NM={near_peak_frac[index].item():.4f}",
                f"{joint_name}_MEAN_ABS_VEL_RADPS={vel_mean[index].item():.4f}",
                f"{joint_name}_MAX_ABS_VEL_RADPS={abs_vel_max[index].item():.4f}",
                f"{joint_name}_FRAC_OVER_2P7RADPS={near_vel_frac[index].item():.4f}",
                f"{joint_name}_MEAN_ABS_ACTION={action_mean[index].item():.4f}",
                f"{joint_name}_MAX_ABS_ACTION={action_abs_max[index].item():.4f}",
                f"{joint_name}_FRAC_ACTION_ABS_GT_0P98={action_sat_frac[index].item():.4f}",
                f"{joint_name}_FRAC_NEAR_JOINT_LIMIT={near_joint_limit_frac[index].item():.4f}",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line, flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
