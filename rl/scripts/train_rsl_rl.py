#!/usr/bin/env python3
"""Train the insectoid mini quad DirectRL task with RSL-RL."""

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
    source = source.replace("import isaaclab_tasks  # noqa: F401", "import isaaclab_tasks  # noqa: F401\nimport insectoid_mini_quad_rl  # noqa: F401")
    source = source.replace(
        "resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)",
        "resume_path = (\n"
        "            os.path.abspath(agent_cfg.load_checkpoint)\n"
        "            if agent_cfg.load_checkpoint and os.path.isfile(agent_cfg.load_checkpoint)\n"
        "            else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)\n"
        "        )",
    )
    exec(compile(source, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})
