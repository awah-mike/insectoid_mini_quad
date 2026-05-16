"""Create the four-walking-leg insectoid_mini_quad URDF.

The source robot is the normalized six-leg insectoid_mini URDF. This variant
keeps the complete front legs but makes each front leg a rigid, non-actuated
appendage. The front coxa joints are baked to a forward 45 degree sweep, and
the front femur/tibia joints are fixed at their URDF zero angles.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


FRONT_FIXED_JOINTS = {
    "FL_coxa_joint",
    "FL_femur_joint",
    "FL_tibia_joint",
    "FR_coxa_joint",
    "FR_femur_joint",
    "FR_tibia_joint",
}
FRONT_COXA_JOINTS = {"FL_coxa_joint", "FR_coxa_joint"}
WALKING_TIBIA_FOOT_OFFSETS = {
    "BL_tibia_1": (-0.200, 0.0, 0.069),
    "BR_tibia_1": (0.200, 0.0, -0.069),
    "ML_tibia_1": (-0.200, 0.0, 0.069),
    "MR_tibia_1": (0.200, 0.0, -0.069),
}
FOOT_RADIUS_M = 0.018
FOOT_MASS_KG = 0.02


def _parse_vector(text: str) -> np.ndarray:
    return np.array([float(value) for value in text.split()], dtype=float)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    sy = -matrix[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1.0e-8:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-matrix[0, 1], matrix[1, 1])
    return np.array([roll, pitch, yaw], dtype=float)


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _format_vector(values: np.ndarray) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _indent(elem: ET.Element, level: int = 0) -> None:
    indent_text = "\n" + level * "  "
    child_indent_text = "\n" + (level + 1) * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = child_indent_text
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent_text
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent_text


def _bake_joint_angle(joint: ET.Element, angle_rad: float) -> None:
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    rpy = _parse_vector(origin.get("rpy", "0 0 0"))
    axis_elem = joint.find("axis")
    axis = _parse_vector(axis_elem.get("xyz", "0 0 1") if axis_elem is not None else "0 0 1")
    baked_rotation = _rpy_to_matrix(rpy) @ _axis_angle_to_matrix(axis, angle_rad)
    origin.set("rpy", _format_vector(_matrix_to_rpy(baked_rotation)))


def _make_joint_fixed(joint: ET.Element) -> None:
    joint.set("type", "fixed")
    for child in list(joint):
        if child.tag in {"axis", "limit", "dynamics", "safety_controller", "calibration"}:
            joint.remove(child)


def _add_spherical_foot(root: ET.Element, tibia_link_name: str, xyz: tuple[float, float, float]) -> None:
    side = tibia_link_name.split("_", 1)[0]
    foot_link_name = f"{side}_FOOT"
    foot_joint_name = f"{side}_foot_fixed_joint"

    # Avoid duplicating foot bodies when regenerating from an already-augmented source.
    if root.find(f"./link[@name='{foot_link_name}']") is not None:
        return

    link = ET.SubElement(root, "link", {"name": foot_link_name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{FOOT_MASS_KG:.6g}"})
    inertia = 0.4 * FOOT_MASS_KG * FOOT_RADIUS_M**2
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{inertia:.10g}",
            "iyy": f"{inertia:.10g}",
            "izz": f"{inertia:.10g}",
            "ixy": "0",
            "iyz": "0",
            "ixz": "0",
        },
    )
    for tag in ("visual", "collision"):
        elem = ET.SubElement(link, tag)
        ET.SubElement(elem, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(elem, "geometry")
        ET.SubElement(geometry, "sphere", {"radius": f"{FOOT_RADIUS_M:.6g}"})
        if tag == "visual":
            ET.SubElement(elem, "material", {"name": "foot_black"})

    joint = ET.SubElement(root, "joint", {"name": foot_joint_name, "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": _format_vector(np.array(xyz, dtype=float)), "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": tibia_link_name})
    ET.SubElement(joint, "child", {"link": foot_link_name})


def make_quad_urdf(source: Path, output: Path, front_coxa_angle_deg: float) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", "insectoid_mini_quad")

    # In this robot +Y is forward. The front-left coxa previously used the
    # opposite sign; visual validation showed that it needed a +90 degree
    # correction, so both front coxas now bake the same positive local sweep.
    forward_sweep = {
        "FR_coxa_joint": math.radians(front_coxa_angle_deg),
        "FL_coxa_joint": math.radians(front_coxa_angle_deg),
    }

    for joint in root.findall("joint"):
        joint_name = joint.get("name", "")
        if joint_name in FRONT_COXA_JOINTS:
            _bake_joint_angle(joint, forward_sweep[joint_name])
        if joint_name in FRONT_FIXED_JOINTS:
            _make_joint_fixed(joint)

    if root.find("./material[@name='foot_black']") is None:
        material = ET.SubElement(root, "material", {"name": "foot_black"})
        ET.SubElement(material, "color", {"rgba": "0.02 0.02 0.02 1"})

    for tibia_link_name, foot_offset in WALKING_TIBIA_FOOT_OFFSETS.items():
        _add_spherical_foot(root, tibia_link_name, foot_offset)

    output.parent.mkdir(parents=True, exist_ok=True)
    _indent(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/workspace/insectoid_mini_quad/URDF_description/urdf/URDF_full_source.urdf"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/insectoid_mini_quad/URDF_description/urdf/URDF_quad.urdf"),
    )
    parser.add_argument("--front-coxa-angle-deg", type=float, default=60.0)
    args = parser.parse_args()
    make_quad_urdf(args.source, args.output, args.front_coxa_angle_deg)
    print(
        f"WROTE_QUAD_URDF {args.output} front_coxa_forward_sweep_deg={args.front_coxa_angle_deg}",
        flush=True,
    )


if __name__ == "__main__":
    main()
