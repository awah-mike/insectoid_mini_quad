#!/usr/bin/env python3
"""Measure torso roll, pitch, and yaw drift for a trained policy rollout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=REPO_ROOT / "trained_models/forward_strict_best/model_1399.pt",
)
parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/analysis/attitude/model_1399_forward_y_0p30")
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--skip-steps", type=int, default=250)
parser.add_argument("--forward-vel", type=float, default=0.30)
parser.add_argument("--lateral-vel", type=float, default=0.0)
parser.add_argument("--yaw-rate", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import insectoid_mini_quad_rl  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def quat_wxyz_to_rpy(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=1)


def summarize_deg(name: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{name}_mean_deg": float(np.mean(values)),
        f"{name}_std_deg": float(np.std(values)),
        f"{name}_min_deg": float(np.min(values)),
        f"{name}_max_deg": float(np.max(values)),
        f"{name}_peak_to_peak_deg": float(np.ptp(values)),
        f"{name}_max_abs_deg": float(np.max(np.abs(values))),
    }


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1
    env_cfg.lin_vel_x_range = (args.lateral_vel, args.lateral_vel)
    env_cfg.lin_vel_y_range = (args.forward_vel, args.forward_vel)
    env_cfg.ang_vel_z_range = (args.yaw_rate, args.yaw_rate)
    env_cfg.rel_standing_envs = 0.0
    if hasattr(env_cfg, "events"):
        env_cfg.events.physics_material = None

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = raw_env.unwrapped
    obs = env.get_observations()
    raw.episode_length_buf.zero_()

    times = []
    quats = []
    root_lin_vel_b = []
    root_ang_vel_b = []
    projected_gravity_b = []
    dones = []

    for step in range(args.steps):
        with torch.inference_mode():
            raw._commands[:, 0] = args.lateral_vel
            raw._commands[:, 1] = args.forward_vel
            raw._commands[:, 2] = args.yaw_rate
            actions = policy(obs)
            obs, _, done_tensor, _ = env.step(actions)

        times.append(step * raw.step_dt)
        quats.append(raw._robot.data.root_link_quat_w[0].detach().cpu().numpy().copy())
        root_lin_vel_b.append(raw._robot.data.root_lin_vel_b[0].detach().cpu().numpy().copy())
        root_ang_vel_b.append(raw._robot.data.root_ang_vel_b[0].detach().cpu().numpy().copy())
        projected_gravity_b.append(raw._robot.data.projected_gravity_b[0].detach().cpu().numpy().copy())
        done = bool(done_tensor[0].item())
        dones.append(done)
        if done:
            print(f"[WARN] env reset at step {step}; zeroing timer", flush=True)
            raw.episode_length_buf.zero_()

    env.close()

    times = np.asarray(times)
    quats = np.asarray(quats)
    root_lin_vel_b = np.asarray(root_lin_vel_b)
    root_ang_vel_b = np.asarray(root_ang_vel_b)
    projected_gravity_b = np.asarray(projected_gravity_b)
    dones = np.asarray(dones, dtype=bool)

    rpy = quat_wxyz_to_rpy(quats)
    rpy[:, 2] = np.unwrap(rpy[:, 2])
    rpy_deg = np.rad2deg(rpy)
    rpy_deg[:, 2] -= rpy_deg[args.skip_steps, 2]

    eval_slice = slice(args.skip_steps, None)
    summary = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "steps": args.steps,
        "skip_steps": args.skip_steps,
        "step_dt": raw.step_dt,
        "duration_s": args.steps * raw.step_dt,
        "analysis_duration_s": (args.steps - args.skip_steps) * raw.step_dt,
        "command": {
            "lateral_vel_x": args.lateral_vel,
            "forward_vel_y": args.forward_vel,
            "yaw_rate_z": args.yaw_rate,
        },
        "done_steps": np.flatnonzero(dones).astype(int).tolist(),
        "mean_root_lin_vel_b_after_skip": root_lin_vel_b[eval_slice].mean(axis=0).tolist(),
        "mean_root_ang_vel_b_after_skip_radps": root_ang_vel_b[eval_slice].mean(axis=0).tolist(),
        "rms_root_ang_vel_b_after_skip_radps": np.sqrt(np.mean(np.square(root_ang_vel_b[eval_slice]), axis=0)).tolist(),
        "mean_projected_gravity_b_after_skip": projected_gravity_b[eval_slice].mean(axis=0).tolist(),
        "rms_projected_gravity_xy_after_skip": float(
            np.sqrt(np.mean(np.sum(np.square(projected_gravity_b[eval_slice, :2]), axis=1)))
        ),
    }
    summary.update(summarize_deg("roll", rpy_deg[eval_slice, 0]))
    summary.update(summarize_deg("pitch", rpy_deg[eval_slice, 1]))
    summary.update(summarize_deg("yaw_drift", rpy_deg[eval_slice, 2]))

    csv_path = args.output_dir / "model_1399_policy_attitude.csv"
    json_path = args.output_dir / "model_1399_policy_attitude_summary.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "time_s",
                "roll_deg",
                "pitch_deg",
                "yaw_drift_deg",
                "root_vx_b",
                "root_vy_b",
                "root_vz_b",
                "root_wx_b",
                "root_wy_b",
                "root_wz_b",
                "projected_gravity_x",
                "projected_gravity_y",
                "projected_gravity_z",
                "done",
            ]
        )
        for step in range(args.steps):
            writer.writerow(
                [step, times[step], *rpy_deg[step].tolist(), *root_lin_vel_b[step].tolist(), *root_ang_vel_b[step].tolist(), *projected_gravity_b[step].tolist(), int(dones[step])]
            )
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"CSV={csv_path}", flush=True)
    print(f"SUMMARY={json_path}", flush=True)
    print(f"ROLL_P2P_DEG={summary['roll_peak_to_peak_deg']:.3f}", flush=True)
    print(f"PITCH_P2P_DEG={summary['pitch_peak_to_peak_deg']:.3f}", flush=True)
    print(f"YAW_DRIFT_P2P_DEG={summary['yaw_drift_peak_to_peak_deg']:.3f}", flush=True)
    print(f"MEAN_VEL_Y={summary['mean_root_lin_vel_b_after_skip'][1]:.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
