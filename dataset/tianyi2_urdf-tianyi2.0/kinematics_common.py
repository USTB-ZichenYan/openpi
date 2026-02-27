from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_URDF = (
    "/home/SENSETIME/yanzichen/data/file/Geometry-Grounded-Gaussian-Splatting/robotics_world_model/tianyi2_urdf-tianyi2.0"
    "/urdf/tianyi2.0_urdf_with_hands.urdf"
)
LEFT_EE_LINK = "left_tcp_link"
RIGHT_EE_LINK = "right_tcp_link"
HEAD_LINK = "camera_head_link"

LEFT_ARM_JOINTS = (
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
)
RIGHT_ARM_JOINTS = (
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
)


@dataclass(frozen=True)
class JointInfo:
    name: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis_xyz: np.ndarray
    joint_type: str


def rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis.astype(np.float32)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )


def parse_urdf_joints(urdf_path: Path) -> tuple[str, dict[str, list[JointInfo]]]:
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    links = {link.attrib["name"] for link in root.findall("link")}
    children = set()
    joints_by_parent: dict[str, list[JointInfo]] = {}

    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        joint_type = joint.attrib.get("type", "fixed")
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children.add(child)

        origin = joint.find("origin")
        if origin is not None:
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        else:
            xyz = np.zeros(3, dtype=np.float32)
            rpy = np.zeros(3, dtype=np.float32)

        axis_node = joint.find("axis")
        axis = (
            np.fromstring(axis_node.attrib.get("xyz", "1 0 0"), sep=" ")
            if axis_node is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )

        info = JointInfo(
            name=name,
            parent=parent,
            child=child,
            origin_xyz=xyz.astype(np.float32),
            origin_rpy=rpy.astype(np.float32),
            axis_xyz=axis.astype(np.float32),
            joint_type=joint_type,
        )
        joints_by_parent.setdefault(parent, []).append(info)

    roots = sorted(links - children)
    if not roots:
        raise ValueError("No root link found in URDF.")
    return roots[0], joints_by_parent


def compute_link_transforms(
    urdf_path: Path,
    q_map: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    root, joints_by_parent = parse_urdf_joints(urdf_path)
    q_map = q_map or {}

    T_base = np.eye(4, dtype=np.float32)

    # Root link (typically "base") is the canonical world/base frame for this pipeline.
    transforms: dict[str, np.ndarray] = {root: T_base}

    stack = [root]
    while stack:
        parent = stack.pop()
        T_parent = transforms[parent]
        for joint in joints_by_parent.get(parent, []):
            angle = 0.0
            if joint.joint_type in ("revolute", "continuous"):
                angle = float(q_map.get(joint.name, 0.0))

            R_origin = rpy_to_rot(joint.origin_rpy)
            R_joint = axis_angle_to_rot(joint.axis_xyz, angle)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R_origin @ R_joint
            T[:3, 3] = joint.origin_xyz

            T_child = T_parent @ T
            transforms[joint.child] = T_child
            stack.append(joint.child)

    return transforms


def load_q_input(arg: str) -> np.ndarray:
    if any(sep in arg for sep in ("/", "\\")) or arg.endswith((".npy", ".json", ".txt")):
        p = Path(arg)
        if p.exists():
            if p.suffix == ".npy":
                return np.load(p)
            if p.suffix == ".json":
                return np.asarray(json.loads(p.read_text()), dtype=np.float32)
            if p.suffix == ".txt":
                return np.asarray([float(x) for x in p.read_text().replace(",", " ").split()], dtype=np.float32)
    return np.asarray([float(x) for x in arg.replace(",", " ").split()], dtype=np.float32)


def build_q_map_from_q32(q: np.ndarray | None) -> dict[str, float]:
    if q is None:
        return {}
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    q_map: dict[str, float] = {}

    if q.size > 0:
        q_map["head_pitch_joint"] = float(q[0])
    if q.size > 1:
        q_map["head_yaw_joint"] = float(q[1])

    for name, idx in zip(LEFT_ARM_JOINTS, range(2, 9)):
        if q.size > idx:
            q_map[name] = float(q[idx])
    for name, idx in zip(RIGHT_ARM_JOINTS, range(15, 22)):
        if q.size > idx:
            q_map[name] = float(q[idx])

    if q.size > 28:
        q_map["body_yaw_joint"] = float(q[28])
    if q.size > 29:
        q_map["waist_pitch_joint"] = float(q[29])

    if q.size > 14:
        left_finger_vals = [q[9], q[10], q[11], q[12], q[13], q[14]]
        left_finger_names = [
            ("left_little_1_joint", "left_little_2_joint"),
            ("left_ring_1_joint", "left_ring_2_joint"),
            ("left_middle_1_joint", "left_middle_2_joint"),
            ("left_index_1_joint", "left_index_2_joint"),
            ("left_thumb_1_joint", "left_thumb_2_joint", "left_thumb_3_joint", "left_thumb_4_joint"),
            (),
        ]
        for val, names in zip(left_finger_vals, left_finger_names):
            for n in names:
                q_map[n] = float(val)

    if q.size > 27:
        right_finger_vals = [q[22], q[23], q[24], q[25], q[26], q[27]]
        right_finger_names = [
            ("right_little_1_joint", "right_little_2_joint"),
            ("right_ring_1_joint", "right_ring_2_joint"),
            ("right_middle_1_joint", "right_middle_2_joint"),
            ("right_index_1_joint", "right_index_2_joint"),
            ("right_thumb_1_joint", "right_thumb_2_joint", "right_thumb_3_joint", "right_thumb_4_joint"),
            (),
        ]
        for val, names in zip(right_finger_vals, right_finger_names):
            for n in names:
                q_map[n] = float(val)

    return q_map
