#!/usr/bin/env python3
"""Analyze foot contact event order for an RSL-RL insectoid mini quad checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--skip-steps", type=int, default=100)
parser.add_argument("--forward-vel", type=float, default=0.30)
parser.add_argument("--output", type=Path, required=True)
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


FOOT_NAMES = ("BL_FOOT", "BR_FOOT", "ML_FOOT", "MR_FOOT")
HUMAN_NAMES = {
    "BL_FOOT": "back left",
    "BR_FOOT": "back right",
    "ML_FOOT": "front left",
    "MR_FOOT": "front right",
}


def main() -> None:
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.sim.device = args.device
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = raw_env.unwrapped
    obs = env.get_observations()
    previous_contacts = raw._get_foot_contacts()[0].clone()
    events: list[tuple[float, str, str, tuple[bool, ...]]] = []
    contact_samples: list[tuple[bool, ...]] = []

    for step in range(args.steps):
        with torch.inference_mode():
            raw._commands[:, 0] = 0.0
            raw._commands[:, 1] = args.forward_vel
            raw._commands[:, 2] = 0.0
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if bool(dones[0].item()):
                previous_contacts = raw._get_foot_contacts()[0].clone()
                continue

            contacts = raw._get_foot_contacts()[0].clone()
            if step >= args.skip_steps:
                state = tuple(bool(x) for x in contacts.tolist())
                contact_samples.append(state)
                touchdown = (~previous_contacts) & contacts
                liftoff = previous_contacts & (~contacts)
                for foot_index, foot_name in enumerate(FOOT_NAMES):
                    if bool(touchdown[foot_index].item()):
                        events.append((step * raw.step_dt, "touchdown", foot_name, state))
                    if bool(liftoff[foot_index].item()):
                        events.append((step * raw.step_dt, "liftoff", foot_name, state))
            previous_contacts = contacts

    touchdown_sequence = [foot for _, kind, foot, _ in events if kind == "touchdown"]
    liftoff_sequence = [foot for _, kind, foot, _ in events if kind == "liftoff"]
    touchdown_counts = Counter(touchdown_sequence)
    state_counts = Counter(contact_samples)
    cycle_counts = Counter(tuple(touchdown_sequence[i : i + 4]) for i in range(max(0, len(touchdown_sequence) - 3)))

    lines = [
        f"CHECKPOINT={args.checkpoint}",
        f"STEPS={args.steps}",
        f"SKIP_STEPS={args.skip_steps}",
        f"STEP_DT={raw.step_dt:.6f}",
        "FOOT_ORDER=BL_FOOT,BR_FOOT,ML_FOOT,MR_FOOT",
        "HUMAN_MAPPING=ML_FOOT:front_left,MR_FOOT:front_right,BL_FOOT:back_left,BR_FOOT:back_right",
        "",
        "TOUCHDOWN_SEQUENCE_FOOT_NAMES=" + ",".join(touchdown_sequence[:80]),
        "TOUCHDOWN_SEQUENCE_HUMAN=" + ",".join(HUMAN_NAMES[foot] for foot in touchdown_sequence[:80]),
        "LIFTOFF_SEQUENCE_HUMAN=" + ",".join(HUMAN_NAMES[foot] for foot in liftoff_sequence[:80]),
        "",
        "TOUCHDOWN_COUNTS=" + ",".join(f"{foot}:{touchdown_counts[foot]}" for foot in FOOT_NAMES),
        "MOST_COMMON_4_TOUCHDOWN_WINDOWS="
        + ";".join(
            f"{' > '.join(HUMAN_NAMES[foot] for foot in cycle)}:{count}"
            for cycle, count in cycle_counts.most_common(8)
        ),
        "MOST_COMMON_CONTACT_STATES="
        + ";".join(
            f"{''.join('1' if value else '0' for value in state)}:{count}" for state, count in state_counts.most_common(10)
        ),
        "",
        "EVENTS_FIRST_120=",
    ]
    for time_s, kind, foot_name, state in events[:120]:
        state_text = "".join("1" if value else "0" for value in state)
        lines.append(f"{time_s:.3f}s {kind:9s} {foot_name:7s} {HUMAN_NAMES[foot_name]:11s} contacts_BL_BR_ML_MR={state_text}")

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
