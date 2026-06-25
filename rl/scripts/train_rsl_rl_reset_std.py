#!/usr/bin/env python3
"""Train RSL-RL while optionally resetting loaded policy exploration std."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_isaaclab_script(script_name: str) -> Path:
    candidates = []
    if os.environ.get("ISAACLAB_PATH"):
        candidates.append(Path(os.environ["ISAACLAB_PATH"]).expanduser())
    candidates.extend(
        [
            Path("/workspace/isaaclab"),
            Path("/workspace/IsaacLab"),
            Path("/home/ubuntu/IsaacLab"),
            Path("/opt/IsaacLab"),
        ]
    )
    for root in candidates:
        script = root / "scripts/reinforcement_learning/rsl_rl" / script_name
        if script.is_file():
            return script
    searched = "\n".join(str(root / "scripts/reinforcement_learning/rsl_rl" / script_name) for root in candidates)
    raise FileNotFoundError(f"Could not find Isaac Lab RSL-RL script {script_name}. Searched:\n{searched}")


if __name__ == "__main__":
    script = resolve_isaaclab_script("train.py")
    sys.argv[0] = str(script)
    sys.path.insert(0, str(script.parent))
    source = script.read_text()
    source = source.replace(
        'parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")',
        'parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")\n'
        'parser.add_argument("--reset_action_std", type=float, default=None, help="Reset loaded policy action std.")',
    )
    source = source.replace(
        "import isaaclab_tasks  # noqa: F401",
        "import isaaclab_tasks  # noqa: F401\nimport insectoid_mini_quad_rl  # noqa: F401",
    )
    source = source.replace(
        "resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)",
        "resume_path = (\n"
        "            os.path.abspath(agent_cfg.load_checkpoint)\n"
        "            if agent_cfg.load_checkpoint and os.path.isfile(agent_cfg.load_checkpoint)\n"
        "            else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)\n"
        "        )",
    )
    source = source.replace(
        "runner.load(resume_path)",
        "runner.load(resume_path)\n"
        "        if args_cli.reset_action_std is not None:\n"
        "            policy = runner.alg.policy\n"
        "            if hasattr(policy, 'std'):\n"
        "                policy.std.data.fill_(args_cli.reset_action_std)\n"
        "            elif hasattr(policy, 'log_std'):\n"
        "                policy.log_std.data.fill_(torch.log(torch.tensor(args_cli.reset_action_std, device=policy.log_std.device)))\n"
        "            else:\n"
        "                raise AttributeError('Loaded policy has no std/log_std parameter to reset.')\n"
        "            print(f'[INFO]: Reset loaded policy action std to {args_cli.reset_action_std}')",
    )
    exec(compile(source, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})
