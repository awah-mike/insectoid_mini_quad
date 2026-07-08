"""Gym registrations for the insectoid mini quad Isaac Lab task."""

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

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefine-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadGaitRefinePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefine-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefinePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadGaitRefinePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadence-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadGaitRefineSlowCadencePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadence-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadencePlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadGaitRefineSlowCadencePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceStrideLock-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceStrideLockEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceStrideLockPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceStrideLock-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceStrideLockPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceStrideLockPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUp-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUp-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpAcquire-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquireEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquirePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpAcquire-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquirePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquirePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpStrong-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpStrongEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpStrongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpStrong-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpStrongPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpStrongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupported-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupported-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV2-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV2-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV2PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV3-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV3EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV3PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV3-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV3PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV3PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV4-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV4EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV4PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV4-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV4PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV4PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV5-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV5-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV6-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV6EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV6PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV6-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV6PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV6PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV7-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV7EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV7PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV7-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV7PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV7PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV8-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV8EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV8PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV8-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV8PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV8PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV9-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV9EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV9PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV9-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV9PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV9PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV10-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV10EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV10PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV10-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV10PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV10PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV11-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV11EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV11PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV11-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV11PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV11PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV12-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV12EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV12PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV12-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV12PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV12PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV13-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV13EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV13PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV13-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV13PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV13PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV14-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV14-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV15-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV15EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV15PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV15-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV15PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV15PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV16-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV16EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV16PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV16-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV16PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV16PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV17-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV17EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV17PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV17-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV17PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV17PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV18-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV18EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV18PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV18-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV18PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV18PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV19-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV19EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV19PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV19-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV19PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV19PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV20-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV20EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV20PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV20-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV20PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV20PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV21-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV21-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV22-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV22EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV22PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV22-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV22PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV22PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23NoVelDeploy-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23NoVelDeployEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23NoVelDeployPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23NoVelDeploy-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23NoVelDeployPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23NoVelDeployPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23RpyDeploy-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeployEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeployPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23RpyDeploy-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeployPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeployPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContact-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContact-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactV2-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactV2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactV2-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactV2PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSway-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSway-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobot-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobot-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23"
        "RpyDeploySmoothContactAntiSwayRealRobotPhase-Direct-v0"
    ),
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPhaseEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPhasePPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV23"
        "RpyDeploySmoothContactAntiSwayRealRobotPhase-Direct-Play-v0"
    ),
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPhasePlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPhasePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV24-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV24EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV24PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpSupportedV24-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV24PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV24PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpRearMirror-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpRearMirrorEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpRearMirrorPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpRearMirror-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpRearMirrorPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpRearMirrorPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpPhaseMirror-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPhaseMirrorEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPhaseMirrorPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineSlowCadenceFrontUpPhaseMirror-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPhaseMirrorPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPhaseMirrorPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineDeploy-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineDeployEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadGaitRefineDeployPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineDeploy-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineDeployPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadGaitRefineDeployPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineNoVelDeploy-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineNoVelDeployEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadGaitRefineNoVelDeployPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-GaitRefineNoVelDeploy-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadGaitRefineNoVelDeployPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadGaitRefineNoVelDeployPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-Backward-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-Backward-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardStabilize-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardStabilizeEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardStabilizePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardStabilize-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardStabilizePlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardStabilizePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardStabilizeV2-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardStabilizeV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardStabilizeV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardStabilizeV2-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardStabilizeV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardStabilizeV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardSymmetry-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardSymmetryEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardSymmetryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardSymmetry-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardSymmetryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardSymmetryPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardAntiInward-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardAntiInwardEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardAntiInwardPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardAntiInward-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardAntiInwardPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardAntiInwardPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLNoCurl-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLNoCurlEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLNoCurlPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLNoCurl-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLNoCurlPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLNoCurlPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLNoCurlBalanced-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLNoCurlBalancedEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLNoCurlBalancedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLNoCurlBalanced-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLNoCurlBalancedPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLNoCurlBalancedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLink-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLink-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalance-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalancePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalance-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalancePlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalancePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV2-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV2-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV2PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV3-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV3-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardDeploy-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardDeployEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardDeployPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardDeploy-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardDeployPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardDeployPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardNoVelDeploy-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardNoVelDeployEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardNoVelDeployPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardNoVelDeploy-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardNoVelDeployPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardNoVelDeployPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV4-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV4EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV4PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV4-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV4PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV4PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV5-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV5EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV5PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV5-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV5PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV5PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV6-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV6EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV6PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV6-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV6PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV6PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV7-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV7EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV7PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV7-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV7PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV7PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV8-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV8EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV8PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV8-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV8PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV8PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV9-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV9EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV9PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV9-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV9PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV9PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV10-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV10EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV10PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV10-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV10PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV10PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV11-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV11EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV11PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV11-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV11PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV11PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV12-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV12EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV12PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV12-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV12PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV12PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV13-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV13EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV13PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV13-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV13PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV13PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV14-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV14EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV14PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV14-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV14PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV14PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV15-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV15-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV16-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV16EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV16PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV16-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV16PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV16PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV17-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV17EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV17PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV17-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV17PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV17PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV18-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV18EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV18PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV18-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV18PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV18PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV19-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV19EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV19PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV19-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV19PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV19PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV20-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV20EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV20PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV20-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV20PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV20PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV21-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV21EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV21PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV21-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV21PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV21PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV22-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV22EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV22PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV22-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV22PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV22PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV23-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV23EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV23PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV23-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV23PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV23PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV24-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV24EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV24PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV24-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV24PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV24PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV25-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV25EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV25PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV25-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV25PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV25PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV26-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV26EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV26PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV26-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV26PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV26PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV27-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV27EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV27PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV27-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV27PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV27PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV28-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV28EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV28PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV28-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV28PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV28PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV29-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV29EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV29PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV29-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV29PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV29PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV30-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV30EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV30PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV30-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV30PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV30PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV31-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV31EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV31PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV31-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV31PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV31PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV32-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV32EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV32PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV32-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV32PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV32PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV33-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV33EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV33PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV33-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV33PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV33PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV34-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV34EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV34PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV34-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV34PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV34PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV35-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV35EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV35PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV35-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV35PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV35PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV36-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV36EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV36PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV36-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV36PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV36PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV37-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV37EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV37PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV37-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV37PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV37PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV38-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV38EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV38PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BackwardBLTibiaLinkBalanceV38-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV38PlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBackwardBLTibiaLinkBalanceV38PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-RearSymmetry-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadRearSymmetryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadRearSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-RearSymmetry-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadRearSymmetryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadRearSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BLTibiaPosture-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBLTibiaPostureEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBLTibiaPosturePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-BLTibiaPosture-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadBLTibiaPosturePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadBLTibiaPosturePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-RearTibiaSymmetry-Direct-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadRearTibiaSymmetryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadRearTibiaSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-InsectoidMiniQuad-RearTibiaSymmetry-Direct-Play-v0",
    entry_point=f"{__name__}.direct_env:InsectoidMiniQuadEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_env:InsectoidMiniQuadRearTibiaSymmetryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InsectoidMiniQuadRearTibiaSymmetryPPORunnerCfg",
    },
)
