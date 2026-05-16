#!/usr/bin/env python3
"""Evaluate an insectoid mini quad RSL-RL checkpoint with simple locomotion metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--forward-vel", type=float, default=0.28)
parser.add_argument("--output", type=Path, default=Path("/workspace/insectoid_mini_quad_rl/outputs/eval/latest.txt"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import insectoid_mini_quad_rl  # noqa: F401, E402
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

    obs = env.get_observations()
    forward_vel = []
    lateral_vel = []
    yaw_rate = []
    tilt = []
    projected_gravity_x = []
    projected_gravity_y = []
    base_height = []
    stride_ema = []
    touchdown_stride = []
    stance_foot_speed = []
    swing_low_speed = []
    low_swing_fraction = []
    foot_contact_count = []
    per_foot_contact_count = []
    per_foot_stance_speed = []
    per_foot_low_swing_speed = []
    per_foot_stride_ema = []
    per_foot_latest_touchdown_stride = []
    per_foot_touchdown_rate = []
    per_foot_completed_stance_duration = []
    per_foot_completed_swing_duration = []
    touchdown_rate = []
    undesired_contact_count = []
    resets = 0

    for _ in range(args.steps):
        with torch.inference_mode():
            raw = raw_env.unwrapped
            raw._commands[:, 0] = 0.0
            raw._commands[:, 1] = args.forward_vel
            raw._commands[:, 2] = 0.0
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            resets += int(torch.count_nonzero(dones).item())
            forward_vel.append(raw._robot.data.root_lin_vel_b[:, 1].detach().mean())
            lateral_vel.append(torch.abs(raw._robot.data.root_lin_vel_b[:, 0]).detach().mean())
            yaw_rate.append(torch.abs(raw._robot.data.root_ang_vel_b[:, 2]).detach().mean())
            tilt.append(torch.norm(raw._robot.data.projected_gravity_b[:, :2], dim=1).detach().mean())
            projected_gravity_x.append(raw._robot.data.projected_gravity_b[:, 0].detach().mean())
            projected_gravity_y.append(raw._robot.data.projected_gravity_b[:, 1].detach().mean())
            base_height.append(raw._robot.data.root_pos_w[:, 2].detach().mean())
            foot_contacts = raw._get_foot_contacts()
            forces = raw._contact_sensor.data.net_forces_w_history
            undesired_contact = torch.amax(
                torch.norm(forces[:, :, raw._undesired_contact_body_ids], dim=-1), dim=1
            ) > 1.0
            foot_pos_w = raw._robot.data.body_pos_w[:, raw._feet_body_ids, :]
            foot_vel_w = raw._robot.data.body_lin_vel_w[:, raw._feet_body_ids, :]
            contact_f = foot_contacts.float()
            swing_f = 1.0 - contact_f
            foot_xy_speed = torch.norm(foot_vel_w[:, :, :2], dim=2)
            foot_height = torch.clamp(foot_pos_w[:, :, 2] - raw._terrain.env_origins[:, 2].unsqueeze(1), min=0.0)
            stance_foot_speed.append(
                (torch.sum(foot_xy_speed * contact_f, dim=1) / (torch.sum(contact_f, dim=1) + 1.0e-6)).detach().mean()
            )
            low_clearance_m = 0.025
            swing_low_speed.append(
                torch.sum(foot_xy_speed * swing_f * (foot_height < low_clearance_m).float(), dim=1).detach().mean()
            )
            low_swing = swing_f * (foot_height < low_clearance_m).float()
            low_swing_fraction.append(low_swing.detach().mean())
            foot_contact_count.append(torch.sum(contact_f, dim=1).detach().mean())
            per_foot_contact_count.append(contact_f.detach().mean(dim=0))
            per_foot_stance_speed.append(
                (torch.sum(foot_xy_speed * contact_f, dim=0) / (torch.sum(contact_f, dim=0) + 1.0e-6)).detach()
            )
            per_foot_low_swing_speed.append(
                (torch.sum(foot_xy_speed * low_swing, dim=0) / (torch.sum(low_swing, dim=0) + 1.0e-6)).detach()
            )
            per_foot_stride_ema.append(raw._stride_length_ema.detach().mean(dim=0))
            per_foot_latest_touchdown_stride.append(raw._latest_touchdown_stride.detach().mean(dim=0))
            per_foot_touchdown_rate.append(
                (raw._step_counts / torch.clamp(raw._total_time.unsqueeze(1), min=raw.step_dt)).detach().mean(dim=0)
            )
            per_foot_completed_stance_duration.append(
                (
                    raw._completed_stance_time_accum
                    / torch.clamp(raw._completed_stance_count, min=1.0)
                ).detach().mean(dim=0)
            )
            per_foot_completed_swing_duration.append(
                (
                    raw._completed_swing_time_accum
                    / torch.clamp(raw._completed_swing_count, min=1.0)
                ).detach().mean(dim=0)
            )
            touchdown_rate.append(
                (torch.sum(raw._step_counts, dim=1) / torch.clamp(raw._total_time, min=raw.step_dt)).detach().mean()
            )
            undesired_contact_count.append(torch.sum(undesired_contact.float(), dim=1).detach().mean())
            stride_ema.append(raw._stride_length_ema.detach().mean())
            touchdown_stride.append(raw._latest_touchdown_stride.detach().mean())

    def mean(values: list[torch.Tensor]) -> float:
        return torch.stack(values).mean().item()

    def vector_mean(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean(dim=0)

    contact_by_foot = vector_mean(per_foot_contact_count)
    stance_speed_by_foot = vector_mean(per_foot_stance_speed)
    low_swing_speed_by_foot = vector_mean(per_foot_low_swing_speed)
    stride_ema_by_foot = vector_mean(per_foot_stride_ema)
    latest_stride_by_foot = vector_mean(per_foot_latest_touchdown_stride)
    touchdown_rate_by_foot = vector_mean(per_foot_touchdown_rate)
    stance_duration_by_foot = vector_mean(per_foot_completed_stance_duration)
    swing_duration_by_foot = vector_mean(per_foot_completed_swing_duration)
    foot_names = ("BL_FOOT", "BR_FOOT", "ML_FOOT", "MR_FOOT")
    contact_imbalance = torch.mean(torch.abs(contact_by_foot - torch.mean(contact_by_foot))).item()
    touchdown_rate_imbalance = torch.mean(torch.abs(touchdown_rate_by_foot - torch.mean(touchdown_rate_by_foot))).item()

    mean_forward = mean(forward_vel)
    mean_command = args.forward_vel
    mean_lateral = mean(lateral_vel)
    mean_yaw = mean(yaw_rate)
    mean_tilt = mean(tilt)
    mean_height = mean(base_height)
    mean_contact = mean(foot_contact_count)
    mean_step_rate = mean(touchdown_rate)
    mean_bad_contact = mean(undesired_contact_count)
    mean_stance_slip = mean(stance_foot_speed)
    mean_low_swing_drag = mean(swing_low_speed)
    mean_stride = torch.mean(stride_ema_by_foot).item()
    mean_stance_duration = torch.mean(stance_duration_by_foot).item()

    velocity_score = torch.exp(torch.tensor(-((mean_command - mean_forward) ** 2) / 0.04)).item()
    lateral_score = torch.exp(torch.tensor(-(mean_lateral**2) / 0.03)).item()
    yaw_score = torch.exp(torch.tensor(-(mean_yaw**2) / 0.12)).item()
    height_score = max(0.0, min(1.0, (mean_height - 0.10) / 0.07))
    tilt_score = torch.exp(torch.tensor(-(mean_tilt**2) / 0.20)).item()
    contact_score = max(0.0, min(1.0, (mean_contact - 1.2) / 1.2))
    stride_score = max(0.0, min(1.0, (mean_stride - 0.04) / 0.06))
    stance_duration_score = max(0.0, min(1.0, (mean_stance_duration - 0.06) / 0.12))
    touchdown_activity_score = max(0.0, min(1.0, mean_step_rate / 2.0))
    rapid_tap_score = torch.exp(torch.tensor(-(max(0.0, mean_step_rate - 12.0) ** 2) / 36.0)).item()
    contact_balance_score = torch.exp(torch.tensor(-(contact_imbalance**2) / 0.02)).item()
    touchdown_balance_score = torch.exp(torch.tensor(-(touchdown_rate_imbalance**2) / 4.0)).item()
    bad_contact_score = torch.exp(torch.tensor(-(mean_bad_contact**2) / 0.25)).item()
    stance_slip_score = torch.exp(torch.tensor(-(mean_stance_slip**2) / 0.04)).item()
    low_swing_drag_score = torch.exp(torch.tensor(-(mean_low_swing_drag**2) / 0.04)).item()
    tracking_score = velocity_score * lateral_score * yaw_score
    posture_score = height_score * tilt_score
    gait_score = (
        contact_score
        + stride_score
        + stance_duration_score
        + touchdown_activity_score
        + rapid_tap_score
        + contact_balance_score
        + touchdown_balance_score
        + bad_contact_score
        + stance_slip_score
        + low_swing_drag_score
    ) / 10.0
    command_direction = 1.0 if mean_command >= 0.0 else -1.0
    commanded_progress = command_direction * mean_forward
    command_speed = abs(mean_command)
    forward_progress_gate = max(0.0, min(1.0, (commanded_progress - 0.08) / max(command_speed - 0.08, 0.08)))
    task_score = forward_progress_gate * (0.45 * tracking_score + 0.25 * posture_score + 0.30 * gait_score)

    lines = [
        f"CHECKPOINT={args.checkpoint}",
        f"STEPS={args.steps} NUM_ENVS={args.num_envs} COMMAND_FORWARD_Y={args.forward_vel:.3f}",
        f"TASK_SCORE={task_score:.4f}",
        f"MEAN_FORWARD_VEL_Y_MPS={mean_forward:.4f}",
        f"MEAN_ABS_LATERAL_VEL_X_MPS={mean_lateral:.4f}",
        f"MEAN_ABS_YAW_RATE_RADPS={mean_yaw:.4f}",
        f"MEAN_PROJECTED_GRAVITY_XY_NORM={mean_tilt:.4f}",
        f"MEAN_PROJECTED_GRAVITY_X={mean(projected_gravity_x):.4f}",
        f"MEAN_PROJECTED_GRAVITY_Y={mean(projected_gravity_y):.4f}",
        f"MEAN_BASE_HEIGHT_M={mean_height:.4f}",
        f"MEAN_STRIDE_EMA_M={mean_stride:.4f}",
        f"MEAN_LATEST_TOUCHDOWN_STRIDE_M={mean(touchdown_stride):.4f}",
        f"MEAN_COMPLETED_STANCE_DURATION_S={mean_stance_duration:.4f}",
        f"MEAN_STANCE_FOOT_XY_SPEED_MPS={mean_stance_slip:.4f}",
        f"MEAN_LOW_SWING_FOOT_XY_SPEED_MPS={mean_low_swing_drag:.4f}",
        f"MEAN_LOW_SWING_FRACTION={mean(low_swing_fraction):.4f}",
        f"MEAN_FOOT_CONTACT_COUNT={mean_contact:.4f}",
        f"MEAN_TOUCHDOWN_RATE_HZ={mean_step_rate:.4f}",
        f"MEAN_UNDESIRED_CONTACT_COUNT={mean_bad_contact:.4f}",
        f"CONTACT_FRACTION_IMBALANCE={contact_imbalance:.4f}",
        f"TOUCHDOWN_RATE_IMBALANCE_HZ={touchdown_rate_imbalance:.4f}",
        f"SCORE_VELOCITY={velocity_score:.4f}",
        f"SCORE_TRACKING={tracking_score:.4f}",
        f"SCORE_POSTURE={posture_score:.4f}",
        f"SCORE_GAIT={gait_score:.4f}",
        f"SCORE_FORWARD_PROGRESS_GATE={forward_progress_gate:.4f}",
        f"SCORE_STRIDE={stride_score:.4f}",
        f"SCORE_STANCE_DURATION={stance_duration_score:.4f}",
        f"SCORE_RAPID_TAP={rapid_tap_score:.4f}",
        f"SCORE_CONTACT_BALANCE={contact_balance_score:.4f}",
        f"SCORE_TOUCHDOWN_BALANCE={touchdown_balance_score:.4f}",
        f"SCORE_TILT={tilt_score:.4f}",
        f"SCORE_STANCE_SLIP={stance_slip_score:.4f}",
        f"SCORE_LOW_SWING_DRAG={low_swing_drag_score:.4f}",
        f"RESETS={resets}",
    ]
    for index, foot_name in enumerate(foot_names):
        lines.extend(
            [
                f"{foot_name}_CONTACT_FRACTION={contact_by_foot[index].item():.4f}",
                f"{foot_name}_TOUCHDOWN_RATE_HZ={touchdown_rate_by_foot[index].item():.4f}",
                f"{foot_name}_STRIDE_EMA_M={stride_ema_by_foot[index].item():.4f}",
                f"{foot_name}_LATEST_TOUCHDOWN_STRIDE_M={latest_stride_by_foot[index].item():.4f}",
                f"{foot_name}_COMPLETED_STANCE_DURATION_S={stance_duration_by_foot[index].item():.4f}",
                f"{foot_name}_COMPLETED_SWING_DURATION_S={swing_duration_by_foot[index].item():.4f}",
                f"{foot_name}_STANCE_XY_SPEED_MPS={stance_speed_by_foot[index].item():.4f}",
                f"{foot_name}_LOW_SWING_XY_SPEED_MPS={low_swing_speed_by_foot[index].item():.4f}",
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
