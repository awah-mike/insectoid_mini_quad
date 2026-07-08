"""Reference kinematic base-velocity estimator for the quad_test adapter.

This is not a ROS node. It is the small estimator that the hardware adapter
should port into `gram-ros_mini/src/locomotion/policy` before deploying the
kinematic deploy checkpoints.

Inputs expected at every 50 Hz policy tick:

- `foot_positions_b`: shape `(4, 3)`, body-frame FK foot positions for
  `[BL, BR, ML, MR]`.
- `stance_contact`: shape `(4,)`, the same contact/stance flags that will be
  placed in observation `[48:52]`.
- `angular_velocity_b`: body-frame IMU angular velocity.

The estimate is written into observation `[0:3]`. It intentionally does not use
simulator/root/world base linear velocity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QuadKinematicBaseVelocityEstimator:
    """Estimate body-frame base linear velocity from stance foot motion."""

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


def clip_quad_test_action(raw_action: np.ndarray) -> np.ndarray:
    """Required action guard before `target = default + 0.5 * action`."""
    return np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
