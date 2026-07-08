#!/usr/bin/env python3
"""Replay an extracted phase-average gait with optional smoothing and yaw stabilization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0")
parser.add_argument(
    "--phase-npz",
    type=Path,
    default=REPO_ROOT / "data/extracted_gaits/model_1399_forward_y_0p30_clean/model_1399_phase_average.npz",
)
parser.add_argument(
    "--summary-json",
    type=Path,
    default=REPO_ROOT / "data/extracted_gaits/model_1399_forward_y_0p30_clean/model_1399_trajectory_summary.json",
)
parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/renders/model_1399_phase_smoothed_stabilized.mp4")
parser.add_argument("--attitude-output", type=Path, default=None)
parser.add_argument("--duration", type=float, default=6.0)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--cycle-time", type=float, default=None)
parser.add_argument("--cycle-time-scale", type=float, default=1.10)
parser.add_argument("--source", choices=("action", "target", "actual"), default="action")
parser.add_argument("--action-filter-alpha", type=float, default=None)
parser.add_argument("--smooth-window", type=int, default=7)
parser.add_argument("--smooth-passes", type=int, default=2)
parser.add_argument("--amplitude-scale", type=float, default=0.90)
parser.add_argument("--side-coxa-bias", type=float, default=0.0)
parser.add_argument("--yaw-feedback-kp", type=float, default=0.18)
parser.add_argument("--yaw-rate-feedback-kd", type=float, default=0.35)
parser.add_argument("--yaw-feedback-sign", type=float, default=1.0)
parser.add_argument("--max-yaw-feedback", type=float, default=0.20)
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--eye", type=float, nargs=3, default=(3.6, -3.6, 3.9))
parser.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.0, 0.12))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = not args.no_video

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import insectoid_mini_quad_rl  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


COXA_LEFT = (0, 2)  # BL, ML in joint order
COXA_RIGHT = (1, 3)  # BR, MR in joint order


def quat_wxyz_to_yaw(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_wxyz_to_rpy_np(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=1)


def circular_smooth(values: np.ndarray, window: int, passes: int) -> np.ndarray:
    if window <= 1 or passes <= 0:
        return values
    if window % 2 == 0:
        window += 1
    radius = window // 2
    out = values.copy()
    for _ in range(passes):
        acc = np.zeros_like(out)
        for offset in range(-radius, radius + 1):
            acc += np.roll(out, shift=offset, axis=0)
        out = acc / float(window)
    return out


def interpolate_cycle(cycle: np.ndarray, phase: float) -> np.ndarray:
    sample_count = cycle.shape[0]
    x = phase * sample_count
    i0 = int(math.floor(x)) % sample_count
    i1 = (i0 + 1) % sample_count
    u = x - math.floor(x)
    return (1.0 - u) * cycle[i0] + u * cycle[i1]


def summarize_attitude(times: np.ndarray, quats: np.ndarray, lin_vel: np.ndarray, ang_vel: np.ndarray, dones: np.ndarray):
    rpy = quat_wxyz_to_rpy_np(quats)
    rpy[:, 2] = np.unwrap(rpy[:, 2])
    rpy_deg = np.rad2deg(rpy)
    rpy_deg[:, 2] -= rpy_deg[0, 2]

    def stats(name: str, values: np.ndarray) -> dict[str, float]:
        return {
            f"{name}_mean_deg": float(np.mean(values)),
            f"{name}_std_deg": float(np.std(values)),
            f"{name}_min_deg": float(np.min(values)),
            f"{name}_max_deg": float(np.max(values)),
            f"{name}_peak_to_peak_deg": float(np.ptp(values)),
            f"{name}_max_abs_deg": float(np.max(np.abs(values))),
        }

    summary = {
        "duration_s": float(times[-1] - times[0]) if len(times) else 0.0,
        "done_steps": np.flatnonzero(dones).astype(int).tolist(),
        "mean_root_lin_vel_b": lin_vel.mean(axis=0).tolist(),
        "mean_root_ang_vel_b_radps": ang_vel.mean(axis=0).tolist(),
        "rms_root_ang_vel_b_radps": np.sqrt(np.mean(np.square(ang_vel), axis=0)).tolist(),
    }
    summary.update(stats("roll", rpy_deg[:, 0]))
    summary.update(stats("pitch", rpy_deg[:, 1]))
    summary.update(stats("yaw_drift", rpy_deg[:, 2]))
    return summary


def main() -> None:
    phase_data = np.load(args.phase_npz)
    summary = json.loads(args.summary_json.read_text()) if args.summary_json.is_file() else {}
    if args.source == "action":
        actions = phase_data["actions_mean"].astype(np.float32)
    else:
        key = "target_joint_pos_mean" if args.source == "target" else "joint_pos_mean"
        default_joint_pos = np.asarray(summary["default_joint_pos"], dtype=np.float32)
        action_scale = float(summary["action_scale"])
        actions = (phase_data[key].astype(np.float32) - default_joint_pos[None, :]) / action_scale
    actions = circular_smooth(actions, args.smooth_window, args.smooth_passes)
    center = actions.mean(axis=0, keepdims=True)
    actions = center + args.amplitude_scale * (actions - center)

    if args.cycle_time is None:
        lengths = summary.get("cycle_summary", {}).get("cycle_lengths_steps", [])
        step_dt = float(summary.get("step_dt", 0.02))
        cycle_time = float(np.mean(lengths) * step_dt) if lengths else 0.36
    else:
        cycle_time = args.cycle_time
    cycle_time *= args.cycle_time_scale

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.viewer.eye = tuple(args.eye)
    env_cfg.viewer.lookat = tuple(args.lookat)
    env_cfg.viewer.origin_type = "world"
    if args.action_filter_alpha is not None and hasattr(env_cfg, "action_target_filter_alpha"):
        env_cfg.action_target_filter_alpha = args.action_filter_alpha
    if hasattr(env_cfg, "events"):
        env_cfg.events.physics_material = None

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device

    raw_env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if not args.no_video else None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    env.get_observations()
    raw = raw_env.unwrapped
    raw.episode_length_buf.zero_()

    initial_yaw = quat_wxyz_to_yaw(raw._robot.data.root_link_quat_w)[0].detach().clone()
    writer = None
    captured_frames = 0
    if not args.no_video:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.output, fps=args.fps)
    times = []
    quats = []
    lin_vel = []
    ang_vel = []
    dones = []

    total_steps = max(1, int(args.duration / raw.step_dt))
    capture_every = max(1, round(1.0 / (args.fps * raw.step_dt)))
    action_tensor = torch.zeros(raw.num_envs, actions.shape[1], device=raw.device)

    try:
        for step in range(total_steps):
            phase = ((step * raw.step_dt) / cycle_time) % 1.0
            action = interpolate_cycle(actions, phase)
            action[list(COXA_LEFT)] += args.side_coxa_bias
            action[list(COXA_RIGHT)] -= args.side_coxa_bias

            yaw = quat_wxyz_to_yaw(raw._robot.data.root_link_quat_w)[0]
            yaw_error = torch.atan2(torch.sin(yaw - initial_yaw), torch.cos(yaw - initial_yaw))
            yaw_rate = raw._robot.data.root_ang_vel_b[0, 2]
            feedback = args.yaw_feedback_sign * (
                -args.yaw_feedback_kp * yaw_error - args.yaw_rate_feedback_kd * yaw_rate
            )
            feedback = torch.clamp(feedback, -args.max_yaw_feedback, args.max_yaw_feedback).item()
            action[list(COXA_LEFT)] += feedback
            action[list(COXA_RIGHT)] -= feedback

            action = np.clip(action, -1.0, 1.0)
            action_tensor[0] = torch.as_tensor(action, dtype=torch.float32, device=raw.device)
            with torch.inference_mode():
                _, _, done_tensor, _ = env.step(action_tensor)

            times.append(step * raw.step_dt)
            quats.append(raw._robot.data.root_link_quat_w[0].detach().cpu().numpy().copy())
            lin_vel.append(raw._robot.data.root_lin_vel_b[0].detach().cpu().numpy().copy())
            ang_vel.append(raw._robot.data.root_ang_vel_b[0].detach().cpu().numpy().copy())
            done = bool(done_tensor[0].item())
            dones.append(done)
            if done:
                raw.episode_length_buf.zero_()

            if writer is not None and (step % capture_every == 0 or step == total_steps - 1):
                frame = raw_env.render()
                if frame is not None and frame.size:
                    writer.append_data(frame)
                    captured_frames += 1

            if (step + 1) % 100 == 0 or step == total_steps - 1:
                print(
                    f"[REPLAY] step={step + 1}/{total_steps} captured_frames={captured_frames}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    attitude = summarize_attitude(
        np.asarray(times),
        np.asarray(quats),
        np.asarray(lin_vel),
        np.asarray(ang_vel),
        np.asarray(dones, dtype=bool),
    )
    attitude.update(
        {
            "phase_npz": str(args.phase_npz),
            "source": args.source,
            "cycle_time_s": cycle_time,
            "action_filter_alpha": args.action_filter_alpha,
            "smooth_window": args.smooth_window,
            "smooth_passes": args.smooth_passes,
            "amplitude_scale": args.amplitude_scale,
            "side_coxa_bias": args.side_coxa_bias,
            "yaw_feedback_kp": args.yaw_feedback_kp,
            "yaw_rate_feedback_kd": args.yaw_rate_feedback_kd,
            "yaw_feedback_sign": args.yaw_feedback_sign,
            "max_yaw_feedback": args.max_yaw_feedback,
        }
    )

    if args.attitude_output is None:
        args.attitude_output = args.output.with_suffix(".attitude.json")
    args.attitude_output.parent.mkdir(parents=True, exist_ok=True)
    args.attitude_output.write_text(json.dumps(attitude, indent=2) + "\n")

    if not args.no_video:
        if not captured_frames:
            raise RuntimeError("No frames captured.")
        print(f"VIDEO={args.output}", flush=True)
    print(f"ATTITUDE={args.attitude_output}", flush=True)
    print(f"ROLL_P2P_DEG={attitude['roll_peak_to_peak_deg']:.3f}", flush=True)
    print(f"PITCH_P2P_DEG={attitude['pitch_peak_to_peak_deg']:.3f}", flush=True)
    print(f"YAW_DRIFT_P2P_DEG={attitude['yaw_drift_peak_to_peak_deg']:.3f}", flush=True)
    print(f"MEAN_VEL_Y={attitude['mean_root_lin_vel_b'][1]:.3f}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
