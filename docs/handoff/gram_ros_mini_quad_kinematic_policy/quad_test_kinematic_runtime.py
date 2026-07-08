"""ROS-side helpers for quad_test_kinematic deployment.

This module is written to be copied into
`gram-ros_mini/src/locomotion/policy`. It intentionally avoids ROS imports so
it can be unit-tested as plain Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


FULL_JOINT_ORDER = (
    "BL_coxa",
    "BL_femur",
    "BL_tibia",
    "BR_coxa",
    "BR_femur",
    "BR_tibia",
    "FL_coxa",
    "FL_femur",
    "FL_tibia",
    "FR_coxa",
    "FR_femur",
    "FR_tibia",
    "ML_coxa",
    "ML_femur",
    "ML_tibia",
    "MR_coxa",
    "MR_femur",
    "MR_tibia",
)

ACTIVE_JOINT_ORDER = (
    "BL_coxa",
    "BR_coxa",
    "ML_coxa",
    "MR_coxa",
    "BL_femur",
    "BR_femur",
    "ML_femur",
    "MR_femur",
    "BL_tibia",
    "BR_tibia",
    "ML_tibia",
    "MR_tibia",
)

ACTIVE_TO_FULL_INDEX = tuple(FULL_JOINT_ORDER.index(name) for name in ACTIVE_JOINT_ORDER)

DEFAULT_ACTIVE_POSITION = np.array(
    [
        -0.3490658503988659,
        -0.3490658503988659,
        0.5235987755982988,
        0.5235987755982988,
        -0.4363323129985824,
        -0.4363323129985824,
        -0.4363323129985824,
        -0.4363323129985824,
        1.9198621771937625,
        1.9198621771937625,
        1.9198621771937625,
        1.9198621771937625,
    ],
    dtype=np.float32,
)

FIXED_FRONT_JOINT_POSITION = {
    "FL_coxa": 1.047198,
    "FL_femur": 0.0,
    "FL_tibia": 0.0,
    "FR_coxa": 1.047198,
    "FR_femur": 0.0,
    "FR_tibia": 0.0,
}


def projected_gravity_from_ros_quat_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    """Return projected gravity using the existing deployment convention."""
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float32).reshape(4)
    return np.array(
        [
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(w * w - x * x - y * y + z * z),
        ],
        dtype=np.float32,
    )


def ros_cmd_to_policy_cmd(linear_x: float, linear_y: float, angular_z: float) -> np.ndarray:
    """Map `/cmd_vel` to the policy command frame."""
    return np.array([-linear_y, linear_x, angular_z], dtype=np.float32)


@dataclass
class QuadKinematicBaseVelocityEstimator:
    """Estimate body-frame base velocity from stance foot FK and IMU omega."""

    smoothing: float = 0.82
    max_abs: float = 1.0
    no_contact_decay: float = 0.98
    zero_z: bool = True
    _initialized: bool = False
    _prev_foot_positions_b: np.ndarray = field(default_factory=lambda: np.zeros((4, 3), dtype=np.float32))
    _estimate_b: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def reset(self) -> None:
        self._initialized = False
        self._prev_foot_positions_b.fill(0.0)
        self._estimate_b.fill(0.0)

    def update(
        self,
        foot_positions_b: np.ndarray,
        stance_contact: np.ndarray,
        angular_velocity_b: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        foot_positions_b = np.asarray(foot_positions_b, dtype=np.float32).reshape(4, 3)
        stance_contact = np.asarray(stance_contact, dtype=bool).reshape(4)
        angular_velocity_b = np.asarray(angular_velocity_b, dtype=np.float32).reshape(3)
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        if not self._initialized:
            self._prev_foot_positions_b[:] = foot_positions_b
            self._estimate_b[:] = 0.0
            self._initialized = True
            return self._estimate_b.copy()

        foot_velocity_b = (foot_positions_b - self._prev_foot_positions_b) / dt
        angular_foot_velocity_b = np.cross(
            np.broadcast_to(angular_velocity_b, foot_positions_b.shape),
            foot_positions_b,
        )
        per_foot_base_velocity_b = -(foot_velocity_b + angular_foot_velocity_b)

        if np.any(stance_contact):
            measured_b = per_foot_base_velocity_b[stance_contact].mean(axis=0)
            if self.zero_z:
                measured_b[2] = 0.0
            measured_b = np.clip(measured_b, -self.max_abs, self.max_abs)
            self._estimate_b = self.smoothing * self._estimate_b + (1.0 - self.smoothing) * measured_b
        else:
            self._estimate_b = self.no_contact_decay * self._estimate_b

        if self.zero_z:
            self._estimate_b[2] = 0.0
        self._prev_foot_positions_b[:] = foot_positions_b
        return self._estimate_b.copy()


def build_quad_test_observation(
    base_linear_velocity_b: np.ndarray,
    angular_velocity_b: np.ndarray,
    projected_gravity_b: np.ndarray,
    policy_command: np.ndarray,
    active_joint_position: np.ndarray,
    active_joint_velocity: np.ndarray,
    previous_action: np.ndarray,
    contact_flags: np.ndarray,
    stance_timers: np.ndarray,
    swing_timers: np.ndarray,
) -> np.ndarray:
    """Build the 60D observation expected by the deploy checkpoints."""
    obs = np.concatenate(
        [
            np.asarray(base_linear_velocity_b, dtype=np.float32).reshape(3),
            np.asarray(angular_velocity_b, dtype=np.float32).reshape(3),
            np.asarray(projected_gravity_b, dtype=np.float32).reshape(3),
            np.asarray(policy_command, dtype=np.float32).reshape(3),
            np.asarray(active_joint_position, dtype=np.float32).reshape(12) - DEFAULT_ACTIVE_POSITION,
            np.asarray(active_joint_velocity, dtype=np.float32).reshape(12),
            np.asarray(previous_action, dtype=np.float32).reshape(12),
            np.asarray(contact_flags, dtype=np.float32).reshape(4),
            np.asarray(stance_timers, dtype=np.float32).reshape(4),
            np.asarray(swing_timers, dtype=np.float32).reshape(4),
        ]
    ).astype(np.float32)
    if obs.shape != (60,):
        raise RuntimeError(f"expected 60D quad observation, got {obs.shape}")
    return obs


def action_to_active_joint_targets(raw_action: np.ndarray, action_clip: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Clip 12D raw policy action and convert it to active joint targets."""
    clipped_action = np.clip(np.asarray(raw_action, dtype=np.float32).reshape(12), -action_clip, action_clip)
    target = DEFAULT_ACTIVE_POSITION + 0.5 * clipped_action
    return target.astype(np.float32), clipped_action.astype(np.float32)


def active_targets_to_full_joint_targets(active_targets: np.ndarray) -> dict[str, float]:
    """Return full 18-joint target map in robot joint coordinates."""
    active_targets = np.asarray(active_targets, dtype=np.float32).reshape(12)
    target_by_name = {name: float(value) for name, value in zip(ACTIVE_JOINT_ORDER, active_targets)}
    target_by_name.update(FIXED_FRONT_JOINT_POSITION)
    return {name: target_by_name[name] for name in FULL_JOINT_ORDER}
