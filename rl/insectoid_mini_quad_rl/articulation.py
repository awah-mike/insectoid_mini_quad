"""Articulation config for the insectoid mini quad robot."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


USD_PATH = "/workspace/insectoid_mini_quad/URDF_description/usd/insectoid_mini_quad.usd"

AK45_36_PEAK_TORQUE_NM = 24.0
AK45_36_RATED_TORQUE_NM = 8.0
AK45_36_CONTINUOUS_TORQUE_NM = AK45_36_RATED_TORQUE_NM
AK45_36_RATED_SPEED_RAD_PER_S = 40.0 * 2.0 * math.pi / 60.0
AK45_36_NO_LOAD_SPEED_RAD_PER_S = 52.0 * 2.0 * math.pi / 60.0
AK45_36_SIM_VELOCITY_LIMIT_RAD_PER_S = 3.0
AK45_36_GEAR_RATIO = 36.0
AK45_36_ROTOR_INERTIA_KG_M2 = 181.90e-7
AK45_36_REFLECTED_ROTOR_INERTIA_KG_M2 = AK45_36_ROTOR_INERTIA_KG_M2 * AK45_36_GEAR_RATIO**2

INSECTOID_MINI_QUAD_STANDING_POSE = {
    "ML_coxa_joint": math.radians(30.0),
    "MR_coxa_joint": math.radians(30.0),
    "BL_coxa_joint": math.radians(-20.0),
    "BR_coxa_joint": math.radians(-20.0),
    "ML_femur_joint": math.radians(-25.0),
    "MR_femur_joint": math.radians(-25.0),
    "BL_femur_joint": math.radians(-25.0),
    "BR_femur_joint": math.radians(-25.0),
    "ML_tibia_joint": math.radians(110.0),
    "MR_tibia_joint": math.radians(110.0),
    "BL_tibia_joint": math.radians(110.0),
    "BR_tibia_joint": math.radians(110.0),
}

INSECTOID_MINI_QUAD_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=INSECTOID_MINI_QUAD_STANDING_POSE,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "walking_legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "ML_.*_joint",
                "MR_.*_joint",
                "BL_.*_joint",
                "BR_.*_joint",
            ],
            effort_limit_sim=AK45_36_PEAK_TORQUE_NM,
            velocity_limit_sim=AK45_36_SIM_VELOCITY_LIMIT_RAD_PER_S,
            armature=AK45_36_REFLECTED_ROTOR_INERTIA_KG_M2,
            stiffness=40.0,
            damping=1.0,
        ),
    },
)
