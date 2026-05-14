"""Verify the generated insectoid_mini_quad USD structure."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--usd-path",
    type=Path,
    default=Path("/workspace/insectoid_mini_quad/URDF_description/usd/insectoid_mini_quad.usd"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402


def main() -> None:
    stage = Usd.Stage.Open(str(args.usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {args.usd_path}")

    bodies = []
    revolute_joints = []
    fixed_joints = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(prim.GetName())
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joints.append(prim.GetName())
        if prim.IsA(UsdPhysics.FixedJoint):
            fixed_joints.append(prim.GetName())

    print(f"USD={args.usd_path}", flush=True)
    print(f"BODIES {len(bodies)} {sorted(bodies)}", flush=True)
    print(f"REVOLUTE_JOINTS {len(revolute_joints)} {sorted(revolute_joints)}", flush=True)
    print(f"FIXED_JOINTS {len(fixed_joints)} {sorted(fixed_joints)}", flush=True)

    expected_revolute = {
        "BL_coxa_joint",
        "BL_femur_joint",
        "BL_tibia_joint",
        "BR_coxa_joint",
        "BR_femur_joint",
        "BR_tibia_joint",
        "ML_coxa_joint",
        "ML_femur_joint",
        "ML_tibia_joint",
        "MR_coxa_joint",
        "MR_femur_joint",
        "MR_tibia_joint",
    }
    expected_fixed = {
        "FL_coxa_joint",
        "FL_femur_joint",
        "FL_tibia_joint",
        "FR_coxa_joint",
        "FR_femur_joint",
        "FR_tibia_joint",
    }
    expected_front_bodies = {
        "FL_coxa_1",
        "FL_femur_1",
        "FL_tibia_1",
        "FR_coxa_1",
        "FR_femur_1",
        "FR_tibia_1",
    }
    missing_revolute = sorted(expected_revolute.difference(revolute_joints))
    missing_fixed = sorted(expected_fixed.difference(fixed_joints))
    missing_front_bodies = sorted(expected_front_bodies.difference(bodies))
    if missing_revolute or missing_fixed or missing_front_bodies or len(bodies) != 19:
        raise RuntimeError(
            "Quad USD verification failed: "
            f"missing_revolute={missing_revolute}, missing_fixed={missing_fixed}, "
            f"missing_front_bodies={missing_front_bodies}, body_count={len(bodies)}"
        )
    print("QUAD_USD_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
