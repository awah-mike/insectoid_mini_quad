"""Workspace import shim for the editable insectoid_mini_quad_rl package."""

from __future__ import annotations

from pathlib import Path

_INNER_PACKAGE = Path(__file__).resolve().parent / "insectoid_mini_quad_rl"
if str(_INNER_PACKAGE) not in __path__:
    __path__.append(str(_INNER_PACKAGE))

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-InsectoidMiniQuad-Flat-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-Flat-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadFlatPPORunnerCfg",
    },
)
