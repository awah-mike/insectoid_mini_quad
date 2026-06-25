#!/usr/bin/env python3
"""Extract joint/action/contact trajectories from a trained RSL-RL policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
FOOT_NAMES = ("BL_FOOT", "BR_FOOT", "ML_FOOT", "MR_FOOT")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=REPO_ROOT / "trained_models/forward_strict_best/model_1399.pt",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=REPO_ROOT / "data/extracted_gaits/model_1399_forward_y_0p30",
)
parser.add_argument("--steps", type=int, default=1600)
parser.add_argument("--skip-steps", type=int, default=250)
parser.add_argument("--forward-vel", type=float, default=0.30)
parser.add_argument("--lateral-vel", type=float, default=0.0)
parser.add_argument("--yaw-rate", type=float, default=0.0)
parser.add_argument("--phase-samples", type=int, default=128)
parser.add_argument("--cycle-foot", choices=("auto", *FOOT_NAMES), default="auto")
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


def _interp_rows(segment: np.ndarray, sample_count: int) -> np.ndarray:
    if len(segment) < 2:
        return np.repeat(segment[:1], sample_count, axis=0)
    src_x = np.linspace(0.0, 1.0, len(segment))
    dst_x = np.linspace(0.0, 1.0, sample_count)
    out = np.empty((sample_count, segment.shape[1]), dtype=np.float32)
    for col in range(segment.shape[1]):
        out[:, col] = np.interp(dst_x, src_x, segment[:, col])
    return out


def _write_matrix_csv(path: Path, header: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(matrix.tolist())


def _write_raw_csv(
    path: Path,
    time_s: np.ndarray,
    command: np.ndarray,
    root_lin_vel_b: np.ndarray,
    root_ang_vel_b: np.ndarray,
    contacts: np.ndarray,
    joint_names: tuple[str, ...],
    actions: np.ndarray,
    policy_actions_raw: np.ndarray,
    target_joint_pos: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
) -> None:
    header = ["step", "time_s", "cmd_x", "cmd_y", "cmd_yaw", "root_vx_b", "root_vy_b", "root_vz_b"]
    header += ["root_wx_b", "root_wy_b", "root_wz_b"]
    header += [f"contact_{name}" for name in FOOT_NAMES]
    header += [f"policy_action_raw_{name}" for name in joint_names]
    header += [f"action_{name}" for name in joint_names]
    header += [f"target_pos_rad_{name}" for name in joint_names]
    header += [f"actual_pos_rad_{name}" for name in joint_names]
    header += [f"actual_vel_radps_{name}" for name in joint_names]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for step in range(len(time_s)):
            row = [step, time_s[step]]
            row += command[step].tolist()
            row += root_lin_vel_b[step].tolist()
            row += root_ang_vel_b[step].tolist()
            row += contacts[step].astype(int).tolist()
            row += policy_actions_raw[step].tolist()
            row += actions[step].tolist()
            row += target_joint_pos[step].tolist()
            row += joint_pos[step].tolist()
            row += joint_vel[step].tolist()
            writer.writerow(row)


def _extract_cycles(
    contacts: np.ndarray,
    dones: np.ndarray,
    arrays_by_name: dict[str, np.ndarray],
    skip_steps: int,
    phase_samples: int,
    cycle_foot: str,
) -> tuple[str, dict[str, np.ndarray], dict[str, object]]:
    post_contacts = contacts[skip_steps:]
    touchdowns = (~post_contacts[:-1]) & post_contacts[1:]
    touchdown_counts = touchdowns.sum(axis=0)
    if cycle_foot == "auto":
        cycle_foot_index = int(np.argmax(touchdown_counts))
        cycle_foot = FOOT_NAMES[cycle_foot_index]
    else:
        cycle_foot_index = FOOT_NAMES.index(cycle_foot)

    touchdown_indices = np.flatnonzero(touchdowns[:, cycle_foot_index]) + skip_steps + 1
    min_cycle_steps = 8
    cycle_bounds = [
        (int(a), int(b))
        for a, b in zip(touchdown_indices[:-1], touchdown_indices[1:])
        if int(b - a) >= min_cycle_steps and not bool(np.any(dones[int(a) : int(b)]))
    ]

    cycle_arrays: dict[str, list[np.ndarray]] = {name: [] for name in arrays_by_name}
    for start, end in cycle_bounds:
        for name, arr in arrays_by_name.items():
            cycle_arrays[name].append(_interp_rows(arr[start:end], phase_samples))

    averaged: dict[str, np.ndarray] = {}
    for name, cycles in cycle_arrays.items():
        if cycles:
            stacked = np.stack(cycles, axis=0)
            averaged[f"{name}_mean"] = stacked.mean(axis=0)
            averaged[f"{name}_std"] = stacked.std(axis=0)

    foot_phase_values: dict[str, list[float]] = {name: [] for name in FOOT_NAMES}
    for start, end in cycle_bounds:
        span = max(1, end - start)
        local_touchdowns = touchdowns[start - skip_steps : end - skip_steps]
        for foot_index, foot_name in enumerate(FOOT_NAMES):
            hits = np.flatnonzero(local_touchdowns[:, foot_index])
            if len(hits):
                foot_phase_values[foot_name].append(float(hits[0] / span))

    foot_phase_summary = {
        foot_name: {
            "mean_phase": float(np.mean(values)) if values else None,
            "std_phase": float(np.std(values)) if values else None,
            "count": len(values),
        }
        for foot_name, values in foot_phase_values.items()
    }

    summary = {
        "cycle_foot": cycle_foot,
        "cycle_foot_index": cycle_foot_index,
        "touchdown_counts_after_skip": {
            foot_name: int(touchdown_counts[index]) for index, foot_name in enumerate(FOOT_NAMES)
        },
        "cycle_count": len(cycle_bounds),
        "excluded_done_steps": np.flatnonzero(dones).astype(int).tolist(),
        "cycle_bounds_step_indices": cycle_bounds,
        "cycle_lengths_steps": [end - start for start, end in cycle_bounds],
        "foot_phase_summary": foot_phase_summary,
    }
    return cycle_foot, averaged, summary


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
    joint_names = tuple(raw._robot.data.joint_names)
    default_joint_pos = raw._robot.data.default_joint_pos[0].detach().cpu().numpy()
    action_scale = float(raw.cfg.action_scale)

    records = {
        "time_s": [],
        "command": [],
        "root_lin_vel_b": [],
        "root_ang_vel_b": [],
        "contacts": [],
        "policy_actions_raw": [],
        "actions": [],
        "target_joint_pos": [],
        "joint_pos": [],
        "joint_vel": [],
        "dones": [],
    }

    for step in range(args.steps):
        with torch.inference_mode():
            raw._commands[:, 0] = args.lateral_vel
            raw._commands[:, 1] = args.forward_vel
            raw._commands[:, 2] = args.yaw_rate
            policy_action_tensor = policy(obs)
            obs, _, dones, _ = env.step(policy_action_tensor)

        if bool(dones[0].item()):
            print(f"[WARN] env reset at step {step}; keeping continuous trace after reset", flush=True)

        records["time_s"].append(step * raw.step_dt)
        records["command"].append(raw._commands[0].detach().cpu().numpy().copy())
        records["root_lin_vel_b"].append(raw._robot.data.root_lin_vel_b[0].detach().cpu().numpy().copy())
        records["root_ang_vel_b"].append(raw._robot.data.root_ang_vel_b[0].detach().cpu().numpy().copy())
        records["contacts"].append(raw._get_foot_contacts()[0].detach().cpu().numpy().copy())
        records["policy_actions_raw"].append(policy_action_tensor[0].detach().cpu().numpy().copy())
        records["actions"].append(raw._actions[0].detach().cpu().numpy().copy())
        records["target_joint_pos"].append(raw._processed_actions[0].detach().cpu().numpy().copy())
        records["joint_pos"].append(raw._robot.data.joint_pos[0].detach().cpu().numpy().copy())
        records["joint_vel"].append(raw._robot.data.joint_vel[0].detach().cpu().numpy().copy())
        records["dones"].append(bool(dones[0].item()))
        if bool(dones[0].item()):
            raw.episode_length_buf.zero_()

    arrays = {name: np.asarray(values) for name, values in records.items()}
    env.close()

    raw_npz_path = args.output_dir / "model_1399_raw_trajectory.npz"
    raw_csv_path = args.output_dir / "model_1399_raw_trajectory.csv"
    phase_npz_path = args.output_dir / "model_1399_phase_average.npz"
    phase_csv_path = args.output_dir / "model_1399_phase_average.csv"
    summary_path = args.output_dir / "model_1399_trajectory_summary.json"

    np.savez_compressed(
        raw_npz_path,
        **arrays,
        joint_names=np.asarray(joint_names),
        foot_names=np.asarray(FOOT_NAMES),
        default_joint_pos=default_joint_pos,
        action_scale=np.asarray(action_scale),
        step_dt=np.asarray(raw.step_dt),
    )
    _write_raw_csv(
        raw_csv_path,
        arrays["time_s"],
        arrays["command"],
        arrays["root_lin_vel_b"],
        arrays["root_ang_vel_b"],
        arrays["contacts"],
        joint_names,
        arrays["actions"],
        arrays["policy_actions_raw"],
        arrays["target_joint_pos"],
        arrays["joint_pos"],
        arrays["joint_vel"],
    )

    arrays_for_cycles = {
        "actions": arrays["actions"],
        "target_joint_pos": arrays["target_joint_pos"],
        "joint_pos": arrays["joint_pos"],
        "joint_vel": arrays["joint_vel"],
    }
    cycle_foot, averaged, cycle_summary = _extract_cycles(
        arrays["contacts"], arrays["dones"], arrays_for_cycles, args.skip_steps, args.phase_samples, args.cycle_foot
    )

    if averaged:
        phase = np.linspace(0.0, 1.0, args.phase_samples, endpoint=True, dtype=np.float32).reshape(-1, 1)
        phase_matrix = np.concatenate(
            [
                phase,
                averaged["actions_mean"],
                averaged["target_joint_pos_mean"],
                averaged["joint_pos_mean"],
                averaged["joint_vel_mean"],
            ],
            axis=1,
        )
        phase_header = ["phase"]
        phase_header += [f"action_mean_{name}" for name in joint_names]
        phase_header += [f"target_pos_mean_rad_{name}" for name in joint_names]
        phase_header += [f"actual_pos_mean_rad_{name}" for name in joint_names]
        phase_header += [f"actual_vel_mean_radps_{name}" for name in joint_names]
        _write_matrix_csv(phase_csv_path, phase_header, phase_matrix)
        np.savez_compressed(
            phase_npz_path,
            phase=phase[:, 0],
            joint_names=np.asarray(joint_names),
            foot_names=np.asarray(FOOT_NAMES),
            **averaged,
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "output_dir": str(args.output_dir),
        "steps": args.steps,
        "skip_steps": args.skip_steps,
        "step_dt": raw.step_dt,
        "duration_s": args.steps * raw.step_dt,
        "command": {
            "lateral_vel_x": args.lateral_vel,
            "forward_vel_y": args.forward_vel,
            "yaw_rate_z": args.yaw_rate,
        },
        "joint_names": list(joint_names),
        "foot_names": list(FOOT_NAMES),
        "default_joint_pos": default_joint_pos.tolist(),
        "action_scale": action_scale,
        "mean_root_lin_vel_b_after_skip": arrays["root_lin_vel_b"][args.skip_steps :].mean(axis=0).tolist(),
        "mean_root_ang_vel_b_after_skip": arrays["root_ang_vel_b"][args.skip_steps :].mean(axis=0).tolist(),
        "mean_contacts_after_skip": arrays["contacts"][args.skip_steps :].mean(axis=0).tolist(),
        "done_steps": np.flatnonzero(arrays["dones"]).astype(int).tolist(),
        "cycle_summary": cycle_summary,
        "files": {
            "raw_npz": str(raw_npz_path),
            "raw_csv": str(raw_csv_path),
            "phase_npz": str(phase_npz_path) if averaged else None,
            "phase_csv": str(phase_csv_path) if averaged else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"RAW_NPZ={raw_npz_path}", flush=True)
    print(f"RAW_CSV={raw_csv_path}", flush=True)
    if averaged:
        print(f"PHASE_NPZ={phase_npz_path}", flush=True)
        print(f"PHASE_CSV={phase_csv_path}", flush=True)
    print(f"SUMMARY={summary_path}", flush=True)
    print(f"CYCLE_FOOT={cycle_foot}", flush=True)
    print(f"CYCLE_COUNT={cycle_summary['cycle_count']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
