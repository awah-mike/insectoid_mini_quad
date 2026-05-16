#!/usr/bin/env python3
"""Play an insectoid mini quad checkpoint with a fixed sampled command range."""

from __future__ import annotations

import sys
from pathlib import Path


if __name__ == "__main__":
    script = Path("/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py")
    sys.argv[0] = str(script)
    sys.path.insert(0, str(script.parent))
    source = script.read_text()
    source = source.replace(
        'parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")',
        'parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")\n'
        'parser.add_argument("--fixed-lin-vel-y", type=float, default=None, help="Force sampled command Y velocity.")\n'
        'parser.add_argument("--video-dir-name", type=str, default="play", help="Video subdirectory under the run log dir.")',
    )
    source = source.replace(
        "import isaaclab_tasks  # noqa: F401",
        "import isaaclab_tasks  # noqa: F401\nimport insectoid_mini_quad_rl  # noqa: F401",
    )
    source = source.replace(
        "env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs",
        "env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs\n"
        "    if args_cli.fixed_lin_vel_y is not None:\n"
        "        env_cfg.lin_vel_y_range = (args_cli.fixed_lin_vel_y, args_cli.fixed_lin_vel_y)\n"
        "        env_cfg.rel_standing_envs = 0.0",
    )
    source = source.replace(
        'os.path.join(log_dir, "videos", "play")',
        'os.path.join(log_dir, "videos", args_cli.video_dir_name)',
    )
    exec(compile(source, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})
