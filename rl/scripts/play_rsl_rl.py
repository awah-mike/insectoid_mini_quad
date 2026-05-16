#!/usr/bin/env python3
"""Play an insectoid mini quad RSL-RL checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path


if __name__ == "__main__":
    script = Path("/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py")
    sys.argv[0] = str(script)
    sys.path.insert(0, str(script.parent))
    source = script.read_text()
    source = source.replace(
        "import isaaclab_tasks  # noqa: F401",
        "import isaaclab_tasks  # noqa: F401\nimport insectoid_mini_quad_rl  # noqa: F401",
    )
    exec(compile(source, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})
