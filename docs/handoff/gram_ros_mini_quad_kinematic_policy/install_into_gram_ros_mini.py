#!/usr/bin/env python3
"""Install quad kinematic deploy artifacts into a gram-ros_mini policy folder.

This script is intentionally non-destructive:

- default mode is dry-run
- existing destination files are not overwritten unless `--force` is set
- `--apply` is required before any filesystem writes happen

Expected target:

`gram-ros_mini/src/locomotion/policy`
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
MODEL_DIR = REPO_ROOT / "trained_models/deploy_quad_kinematic"

FILES_TO_INSTALL = (
    (
        MODEL_DIR / "model_1798_forward_deploy_kinematic.pt",
        Path("models/model_1798_forward_deploy_kinematic.pt"),
    ),
    (
        MODEL_DIR / "model_2615_backward_deploy_kinematic.pt",
        Path("models/model_2615_backward_deploy_kinematic.pt"),
    ),
    (
        MODEL_DIR / "manifest.json",
        Path("models/quad_kinematic_deploy_manifest.json"),
    ),
    (
        THIS_DIR / "policy_config_quad_forward_kinematic.yaml",
        Path("policy_config_quad_forward_kinematic.yaml"),
    ),
    (
        THIS_DIR / "policy_config_quad_backward_kinematic.yaml",
        Path("policy_config_quad_backward_kinematic.yaml"),
    ),
    (
        THIS_DIR / "quad_test_kinematic_runtime.py",
        Path("quad_test_kinematic_runtime.py"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy_dir",
        type=Path,
        help="Path to gram-ros_mini/src/locomotion/policy",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy files. Without this flag the script only prints planned actions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing destination files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_dir = args.policy_dir.expanduser().resolve()

    missing_sources = [str(src) for src, _ in FILES_TO_INSTALL if not src.is_file()]
    if missing_sources:
        raise FileNotFoundError("missing source files:\n" + "\n".join(missing_sources))

    print(f"TARGET_POLICY_DIR={policy_dir}")
    print(f"MODE={'apply' if args.apply else 'dry-run'}")
    print(f"FORCE={args.force}")

    if not policy_dir.exists():
        if not args.apply:
            print(f"[DRY-RUN] would create policy directory: {policy_dir}")
        else:
            policy_dir.mkdir(parents=True)

    planned = []
    for src, rel_dst in FILES_TO_INSTALL:
        dst = policy_dir / rel_dst
        planned.append((src, dst))
        exists = dst.exists()
        if exists and not args.force:
            print(f"[SKIP] exists: {dst}")
            continue
        action = "overwrite" if exists else "copy"
        if not args.apply:
            print(f"[DRY-RUN] would {action}: {src} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[OK] {action}: {dst}")

    print("\nManual integration still required:")
    print("- observation_builder.py: use QuadKinematicBaseVelocityEstimator for quad obs[0:3]")
    print("- action_processor.py: clip raw 12D action to [-1, 1] before 0.5 scaling")
    print("- policy_node.py/config launch: select policy_config_quad_forward_kinematic.yaml or backward")
    print("- joint_mapping.py: preserve the documented 12D active order and 18D publish order")
    print("\nInstalled/planned files:")
    for _, dst in planned:
        print(f"- {dst}")


if __name__ == "__main__":
    main()
