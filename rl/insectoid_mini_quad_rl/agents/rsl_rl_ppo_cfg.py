from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class InsectoidMiniQuadFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "insectoid_mini_quad_flat"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadGaitRefinePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 600
    save_interval = 25
    experiment_name = "insectoid_mini_quad_gait_refine"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadencePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 180
    save_interval = 10
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.18,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.8,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceStrideLockPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadencePPORunnerCfg
):
    max_iterations = 80
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_stride_lock"


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceStrideLockPPORunnerCfg
):
    max_iterations = 70
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up"


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquirePPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPPORunnerCfg
):
    max_iterations = 80
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_acquire"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.30,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0015,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=4.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.9,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpStrongPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquirePPORunnerCfg
):
    max_iterations = 70
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_strong"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0035,
        max_grad_norm=0.9,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquirePPORunnerCfg
):
    max_iterations = 120
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.32,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0018,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=4.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.9,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV2PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedPPORunnerCfg
):
    max_iterations = 160
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v2"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0014,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0035,
        max_grad_norm=0.85,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV3PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedPPORunnerCfg
):
    max_iterations = 220
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v3"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.26,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.8,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV4PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV3PPORunnerCfg
):
    max_iterations = 220
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v4"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.22,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0009,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0025,
        max_grad_norm=0.75,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV4PPORunnerCfg
):
    max_iterations = 240
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v5"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.24,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0011,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.8,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV6PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PPORunnerCfg
):
    max_iterations = 260
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v6"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.22,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0009,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0025,
        max_grad_norm=0.75,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV7PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PPORunnerCfg
):
    max_iterations = 240
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v7"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.24,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0010,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.2e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.8,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV8PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PPORunnerCfg
):
    max_iterations = 260
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v8"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.20,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.8e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0025,
        max_grad_norm=0.75,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV9PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV8PPORunnerCfg
):
    max_iterations = 260
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v9"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.18,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0007,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.6e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.7,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV10PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV9PPORunnerCfg
):
    max_iterations = 280
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v10"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.18,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00065,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.7,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV11PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PPORunnerCfg
):
    max_iterations = 220
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v11"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.18,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00065,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.4e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.7,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV12PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV11PPORunnerCfg
):
    max_iterations = 180
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v12"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.2e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.7,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV13PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV12PPORunnerCfg
):
    max_iterations = 180
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v13"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0010,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.6e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0025,
        max_grad_norm=0.7,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PPORunnerCfg
):
    max_iterations = 260
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v14"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.20,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00075,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0022,
        max_grad_norm=0.7,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV15PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14PPORunnerCfg
):
    max_iterations = 300
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v15"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.18,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00065,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.2e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0018,
        max_grad_norm=0.65,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV16PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV14PPORunnerCfg
):
    max_iterations = 280
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v16"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.20,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00080,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0022,
        max_grad_norm=0.70,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV17PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV5PPORunnerCfg
):
    max_iterations = 260
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v17"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.22,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0010,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0028,
        max_grad_norm=0.75,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV18PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV17PPORunnerCfg
):
    max_iterations = 140
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v18"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.16,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00055,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=9.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0016,
        max_grad_norm=0.65,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV19PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV18PPORunnerCfg
):
    max_iterations = 120
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v19"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.13,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00045,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=7.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0014,
        max_grad_norm=0.60,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV20PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV19PPORunnerCfg
):
    max_iterations = 120
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v20"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.12,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00040,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=6.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0012,
        max_grad_norm=0.60,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV20PPORunnerCfg
):
    max_iterations = 140
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v21"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.11,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00035,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0011,
        max_grad_norm=0.55,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV22PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21PPORunnerCfg
):
    max_iterations = 90
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v22"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.10,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00030,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=4.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0010,
        max_grad_norm=0.50,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21PPORunnerCfg
):
    max_iterations = 75
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v23"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.10,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00028,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.5e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0009,
        max_grad_norm=0.48,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23NoVelDeployPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23PPORunnerCfg
):
    max_iterations = 80
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v23_no_vel_deploy"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.08,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00020,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.5e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0007,
        max_grad_norm=0.60,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeployPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23NoVelDeployPPORunnerCfg
):
    max_iterations = 100
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v23_rpy_deploy"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.06,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00016,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00060,
        max_grad_norm=0.50,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeployPPORunnerCfg
):
    max_iterations = 120
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v23_rpy_deploy_smooth_contact"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.045,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.2e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00045,
        max_grad_norm=0.45,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactV2PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactPPORunnerCfg
):
    max_iterations = 80
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v23_rpy_deploy_smooth_contact_v2"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.035,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=8.0e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00035,
        max_grad_norm=0.40,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactPPORunnerCfg
):
    max_iterations = 80
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v23_rpy_deploy_smooth_contact_anti_sway"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.030,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.18,
        entropy_coef=0.00006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=7.0e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00030,
        max_grad_norm=0.38,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayPPORunnerCfg
):
    max_iterations = 100
    save_interval = 5
    experiment_name = (
        "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v23_"
        "rpy_deploy_smooth_contact_anti_sway_real_robot"
    )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.025,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.16,
        entropy_coef=0.00005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00025,
        max_grad_norm=0.35,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPhasePPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV23RpyDeploySmoothContactAntiSwayRealRobotPPORunnerCfg
):
    max_iterations = 80
    save_interval = 5
    experiment_name = (
        "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v23_"
        "rpy_deploy_smooth_contact_anti_sway_real_robot_phase"
    )
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.022,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.16,
        entropy_coef=0.000045,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00025,
        max_grad_norm=0.35,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV24PPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpSupportedV21PPORunnerCfg
):
    max_iterations = 80
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_supported_v24"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.10,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0008,
        max_grad_norm=0.45,
    )


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpRearMirrorPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquirePPORunnerCfg
):
    max_iterations = 60
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_rear_mirror"


@configclass
class InsectoidMiniQuadGaitRefineSlowCadenceFrontUpPhaseMirrorPPORunnerCfg(
    InsectoidMiniQuadGaitRefineSlowCadenceFrontUpAcquirePPORunnerCfg
):
    max_iterations = 50
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_slow_cadence_front_up_phase_mirror"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0010,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0035,
        max_grad_norm=0.8,
    )


@configclass
class InsectoidMiniQuadBackwardPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 25
    experiment_name = "insectoid_mini_quad_backward"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardStabilizePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 220
    save_interval = 25
    experiment_name = "insectoid_mini_quad_backward_stabilize"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0018,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardStabilizeV2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 160
    save_interval = 25
    experiment_name = "insectoid_mini_quad_backward_stabilize_v2"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardSymmetryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 140
    save_interval = 25
    experiment_name = "insectoid_mini_quad_backward_symmetry"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0010,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0035,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardAntiInwardPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 140
    save_interval = 5
    experiment_name = "insectoid_mini_quad_backward_anti_inward"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLNoCurlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 160
    save_interval = 5
    experiment_name = "insectoid_mini_quad_backward_bl_no_curl"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.35,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLNoCurlBalancedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 90
    save_interval = 5
    experiment_name = "insectoid_mini_quad_backward_bl_no_curl_balanced"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.20,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 80
    save_interval = 5
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.16,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=8.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0018,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 70
    save_interval = 5
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.12,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=7.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0016,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 70
    save_interval = 5
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v2"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.10,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00035,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0014,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV3PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 60
    save_interval = 5
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v3"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.08,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=4.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0012,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV4PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 45
    save_interval = 5
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v4"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.05,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0010,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV5PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 24
    save_interval = 2
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v5"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.025,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.2e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0008,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV6PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 30
    save_interval = 2
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v6"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.035,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.8e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0009,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV7PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 16
    save_interval = 2
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v7"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.018,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00015,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=7.5e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00055,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV8PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 12
    save_interval = 2
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v8"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.012,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00010,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=4.5e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00040,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV9PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 18
    save_interval = 2
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v9"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.018,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=7.0e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00045,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV10PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 8
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v10"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.010,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=4.0e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00030,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV11PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v11"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.006,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00020,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV12PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 3
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v12"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.002,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=5.0e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00008,
        max_grad_norm=0.5,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV13PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 4
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v13"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.004,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.2e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00012,
        max_grad_norm=0.75,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV14PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v14"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.003,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00001,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=8.0e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00008,
        max_grad_norm=0.6,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV15PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v15"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.003,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00001,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.0e-7,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00009,
        max_grad_norm=0.65,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV16PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v16"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.003,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00001,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=7.0e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00008,
        max_grad_norm=0.6,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV17PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v17"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.003,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00001,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=6.0e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00007,
        max_grad_norm=0.55,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV18PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v18"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.003,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00001,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=8.0e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00008,
        max_grad_norm=0.60,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV19PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v19"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0025,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.5e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00006,
        max_grad_norm=0.55,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV20PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v20"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.002,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.0e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.000045,
        max_grad_norm=0.50,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV21PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v21"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.002,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.5e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00005,
        max_grad_norm=0.50,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV22PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 8
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v22"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.002,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.5e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00006,
        max_grad_norm=0.52,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV23PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v23"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.002,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.2e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00005,
        max_grad_norm=0.50,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV24PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v24"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0015,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.5e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.000045,
        max_grad_norm=0.48,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV25PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v25"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0012,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000004,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.8e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00004,
        max_grad_norm=0.45,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV26PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v26"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.002,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00006,
        max_grad_norm=0.55,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV27PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v27"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0025,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=6.0e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00007,
        max_grad_norm=0.58,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV28PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v28"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0018,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.5e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00006,
        max_grad_norm=0.55,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV29PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v29"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0018,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000006,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.5e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.000055,
        max_grad_norm=0.52,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV30PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 4
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v30"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0012,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000004,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.8e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.000045,
        max_grad_norm=0.48,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV31PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v31"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0010,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000003,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.2e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.000040,
        max_grad_norm=0.46,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV32PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v32"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0011,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0000035,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.6e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.000045,
        max_grad_norm=0.48,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV33PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v33"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0012,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000004,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=4.2e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.000052,
        max_grad_norm=0.50,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV34PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 4
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v34"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0009,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0000025,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.6e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.000035,
        max_grad_norm=0.45,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV35PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 4
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v35"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0007,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.2e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.000025,
        max_grad_norm=0.40,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV36PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v36"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0008,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.8e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00003,
        max_grad_norm=0.42,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV37PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 8
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v37"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0008,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.2e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.000035,
        max_grad_norm=0.45,
    )


@configclass
class InsectoidMiniQuadBackwardBLTibiaLinkBalanceV38PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 6
    save_interval = 1
    experiment_name = "insectoid_mini_quad_backward_bl_tibia_link_balance_v38"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.0008,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.6e-8,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.00004,
        max_grad_norm=0.45,
    )


@configclass
class InsectoidMiniQuadRearSymmetryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 250
    save_interval = 25
    experiment_name = "insectoid_mini_quad_rear_symmetry"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBLTibiaPosturePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 150
    save_interval = 25
    experiment_name = "insectoid_mini_quad_bl_tibia_posture"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadRearTibiaSymmetryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 25
    experiment_name = "insectoid_mini_quad_rear_tibia_symmetry"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0015,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadGaitRefineDeployPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 60
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_deploy_obs"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.03,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0015,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadBackwardDeployPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 80
    save_interval = 5
    experiment_name = "insectoid_mini_quad_backward_deploy_obs"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.025,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=1.0,
    )


@configclass
class InsectoidMiniQuadGaitRefineNoVelDeployPPORunnerCfg(InsectoidMiniQuadGaitRefineDeployPPORunnerCfg):
    max_iterations = 40
    save_interval = 5
    experiment_name = "insectoid_mini_quad_gait_refine_no_vel_deploy_obs"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.02,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00015,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0008,
        max_grad_norm=0.8,
    )


@configclass
class InsectoidMiniQuadBackwardNoVelDeployPPORunnerCfg(InsectoidMiniQuadBackwardDeployPPORunnerCfg):
    max_iterations = 40
    save_interval = 5
    experiment_name = "insectoid_mini_quad_backward_no_vel_deploy_obs"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.015,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00015,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-6,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0007,
        max_grad_norm=0.8,
    )
