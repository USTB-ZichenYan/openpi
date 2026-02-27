#!/usr/bin/env python3
"""
FT cleaning pipeline for LeRobot episode parquet files (sensor-frame core mode).

Implemented steps:
1) Contact detection (hysteresis, bilateral)
   - use raw sensor-frame |ΔF_xyz| = ||F_t - F_{t-1}|| (no bias removal / low-pass)
2) Stage labeling (free/contact/lift/carry, FK-based)
   - contact -> lift: mean EE z rises after contact
   - lift -> carry: mean EE speed stays low
3) EE pose symmetry filtering (optional, episode-level)
4) Write cleaned parquet: episode_xxx_clean.parquet
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


LEFT_FT_KEY = "observation.left_arm_ft"
RIGHT_FT_KEY = "observation.right_arm_ft"
DUAL_FT_KEY = "observation.dual_arm_ft"
STATE_Q32_KEY = "observation.state"
ACTION_Q32_KEY = "action"

STAGE_LABELS = ("free", "contact", "lift", "carry")
DEFAULT_KIN_COMMON_PATH = (
    "/home/SENSETIME/yanzichen/data/file/openpi/dataset/tianyi2_urdf-tianyi2.0/kinematics_common.py"
)
DEFAULT_URDF_PATH = (
    "/home/SENSETIME/yanzichen/data/file/openpi/dataset/tianyi2_urdf-tianyi2.0/"
    "urdf/tianyi2.0_urdf_with_hands.urdf"
)


def resolve_chunk_dir(path: Path) -> Path:
    """Accept dataset root or chunk dir, return dir containing episode_*.parquet."""
    if any(path.glob("episode_*.parquet")):
        return path
    candidate = path / "data" / "chunk-000"
    if any(candidate.glob("episode_*.parquet")):
        return candidate
    candidate2 = path / "chunk-000"
    if any(candidate2.glob("episode_*.parquet")):
        return candidate2
    raise FileNotFoundError(f"No episode_*.parquet found under: {path}")


def default_out_dir(in_path: Path, chunk_dir: Path) -> Path:
    """Generate default output chunk dir."""
    if (in_path / "data" / "chunk-000") == chunk_dir:
        return in_path.parent / f"{in_path.name}_clean" / "data" / "chunk-000"
    return chunk_dir.parent / f"{chunk_dir.name}_clean"


def _to_array_2d(values: list, dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"Expected shape [T, {dim}], got {arr.shape}")
    return arr


def load_dual_arm_ft(table: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    """
    Load per-frame left/right 6D FT arrays.
    Priority:
      1) observation.left_arm_ft + observation.right_arm_ft
      2) observation.dual_arm_ft split into left/right
    """
    names = set(table.column_names)
    if LEFT_FT_KEY in names and RIGHT_FT_KEY in names:
        left = _to_array_2d(table.column(LEFT_FT_KEY).to_pylist(), dim=6)
        right = _to_array_2d(table.column(RIGHT_FT_KEY).to_pylist(), dim=6)
        return left, right

    if DUAL_FT_KEY in names:
        dual = _to_array_2d(table.column(DUAL_FT_KEY).to_pylist(), dim=12)
        return dual[:, :6], dual[:, 6:]

    raise KeyError(
        f"Missing FT columns. Need ({LEFT_FT_KEY},{RIGHT_FT_KEY}) or ({DUAL_FT_KEY})."
    )


def load_q32_series(table: pa.Table) -> np.ndarray:
    cols = set(table.column_names)
    if STATE_Q32_KEY in cols:
        return _to_array_2d(table.column(STATE_Q32_KEY).to_pylist(), dim=32)
    if ACTION_Q32_KEY in cols:
        return _to_array_2d(table.column(ACTION_Q32_KEY).to_pylist(), dim=32)
    raise KeyError(f"Missing {STATE_Q32_KEY!r} and {ACTION_Q32_KEY!r}; cannot compute FK.")


def load_kinematics_module(py_path: Path):
    if not py_path.exists():
        raise FileNotFoundError(f"kinematics_common.py not found: {py_path}")
    module_name = "tianyi2_kinematics_common_runtime"
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclass/type introspection expects module to exist in sys.modules during import.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class FKContext:
    mod: object
    root: str
    joints_by_parent: dict
    left_ee_link: str
    right_ee_link: str


def build_fk_context(
    *,
    kin_common_path: Path,
    urdf_path: Path,
    left_ee_link: str,
    right_ee_link: str,
) -> FKContext:
    mod = load_kinematics_module(kin_common_path)
    root, joints_by_parent = mod.parse_urdf_joints(urdf_path)
    return FKContext(
        mod=mod,
        root=root,
        joints_by_parent=joints_by_parent,
        left_ee_link=left_ee_link,
        right_ee_link=right_ee_link,
    )


def compute_link_transforms_cached(ctx: FKContext, q_map: dict[str, float]) -> dict[str, np.ndarray]:
    """Forward kinematics using pre-parsed URDF joints."""
    transforms: dict[str, np.ndarray] = {ctx.root: np.eye(4, dtype=np.float32)}
    stack = [ctx.root]

    while stack:
        parent = stack.pop()
        t_parent = transforms[parent]
        for joint in ctx.joints_by_parent.get(parent, []):
            angle = 0.0
            if joint.joint_type in ("revolute", "continuous"):
                angle = float(q_map.get(joint.name, 0.0))

            r_origin = ctx.mod.rpy_to_rot(joint.origin_rpy)
            r_joint = ctx.mod.axis_angle_to_rot(joint.axis_xyz, angle)
            t_joint = np.eye(4, dtype=np.float32)
            t_joint[:3, :3] = r_origin @ r_joint
            t_joint[:3, 3] = joint.origin_xyz

            t_child = t_parent @ t_joint
            transforms[joint.child] = t_child
            stack.append(joint.child)
    return transforms


def compute_ee_kinematics(
    q32: np.ndarray, ctx: FKContext, fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (left_ee_xyz[T,3], right_ee_xyz[T,3], mean_ee_xyz[T,3], mean_ee_speed[T])."""
    t = q32.shape[0]
    if t == 0:
        z3 = np.zeros((0, 3), dtype=np.float32)
        z1 = np.zeros((0,), dtype=np.float32)
        return z3, z3, z3, z1

    left_pos = np.zeros((t, 3), dtype=np.float32)
    right_pos = np.zeros((t, 3), dtype=np.float32)
    for i in range(t):
        q_map = ctx.mod.build_q_map_from_q32(q32[i])
        tf = compute_link_transforms_cached(ctx, q_map)
        if ctx.left_ee_link not in tf or ctx.right_ee_link not in tf:
            raise KeyError(
                f"Missing ee transform at frame {i}: "
                f"left={ctx.left_ee_link in tf}, right={ctx.right_ee_link in tf}"
            )
        left_pos[i] = tf[ctx.left_ee_link][:3, 3]
        right_pos[i] = tf[ctx.right_ee_link][:3, 3]

    mean_pos = 0.5 * (left_pos + right_pos)
    vel = np.zeros((t,), dtype=np.float32)
    if t > 1:
        dt = 1.0 / max(float(fps), 1e-8)
        dp = np.diff(mean_pos, axis=0)
        vel[1:] = (np.linalg.norm(dp, axis=1) / dt).astype(np.float32)
    return left_pos, right_pos, mean_pos, vel


def detect_contact_hysteresis(
    signal: np.ndarray,
    *,
    t_enter: float,
    t_exit: float,
    enter_frames: int,
    exit_frames: int,
) -> np.ndarray:
    """FREE <-> CONTACT hysteresis state machine."""
    state = 0  # 0=FREE, 1=CONTACT
    above_cnt = 0
    below_cnt = 0
    flags = np.zeros((signal.shape[0],), dtype=np.int8)

    for i, v in enumerate(signal):
        if state == 0:
            if float(v) > t_enter:
                above_cnt += 1
            else:
                above_cnt = 0
            if above_cnt >= enter_frames:
                state = 1
                below_cnt = 0
        else:
            if float(v) < t_exit:
                below_cnt += 1
            else:
                below_cnt = 0
            if below_cnt >= exit_frames:
                state = 0
                above_cnt = 0

        flags[i] = state
    return flags


def frame_diff_norm3(xyz: np.ndarray) -> np.ndarray:
    """||x[t]-x[t-1]||_2 for 3D signal, same length as input, first frame is 0."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be [T,3], got {xyz.shape}")
    if xyz.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    d = np.diff(xyz.astype(np.float32), axis=0, prepend=xyz[[0]].astype(np.float32))
    return np.linalg.norm(d, axis=1).astype(np.float32)


def first_run_start(mask: np.ndarray, min_len: int) -> int | None:
    run = 0
    for i, ok in enumerate(mask.tolist()):
        if ok:
            run += 1
            if run >= min_len:
                return i - min_len + 1
        else:
            run = 0
    return None


def keep_true_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only True runs with length >= min_len."""
    mask = np.asarray(mask, dtype=bool)
    if min_len <= 1 or mask.size == 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    start = None
    for i, v in enumerate(mask.tolist()):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            if i - start >= min_len:
                out[start:i] = True
            start = None
    if start is not None and (mask.size - start) >= min_len:
        out[start:] = True
    return out


def latch_after_first_true(mask: np.ndarray) -> np.ndarray:
    """Return a non-decreasing boolean mask: once True, always True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return mask.copy()
    idx = np.where(mask)[0]
    out = np.zeros_like(mask, dtype=bool)
    if idx.size > 0:
        out[int(idx[0]) :] = True
    return out


def upsert_column(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    if name in table.column_names:
        idx = table.column_names.index(name)
        return table.set_column(idx, name, array)
    return table.append_column(name, array)


def upsert_stage_columns(table: pa.Table, stage_id: np.ndarray, stage_label: list[str]) -> pa.Table:
    out = upsert_column(table, "stage_id", pa.array(stage_id.tolist(), type=pa.int8()))
    out = upsert_column(out, "stage_label", pa.array(stage_label, type=pa.string()))
    return out


def upsert_ee_columns(
    table: pa.Table,
    left_ee_xyz: np.ndarray,
    right_ee_xyz: np.ndarray,
    mean_ee_xyz: np.ndarray,
    mean_ee_speed: np.ndarray,
) -> pa.Table:
    out = table
    out = upsert_column(
        out,
        "observation.left_ee_xyz",
        pa.array(left_ee_xyz.tolist(), type=pa.list_(pa.float32(), 3)),
    )
    out = upsert_column(
        out,
        "observation.right_ee_xyz",
        pa.array(right_ee_xyz.tolist(), type=pa.list_(pa.float32(), 3)),
    )
    out = upsert_column(
        out,
        "observation.ee_mean_xyz",
        pa.array(mean_ee_xyz.tolist(), type=pa.list_(pa.float32(), 3)),
    )
    out = upsert_column(
        out,
        "observation.ee_mean_speed",
        pa.array(mean_ee_speed.tolist(), type=pa.float32()),
    )
    return out


def rolling_var_1d(x: np.ndarray, window: int) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {x.shape}")
    n = x.shape[0]
    if n == 0:
        return x.copy()
    w = max(1, int(window))
    out = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        s = max(0, i - w + 1)
        out[i] = float(np.var(x[s : i + 1]))
    return out


def evaluate_ee_pose_symmetry(
    left_ee_xyz: np.ndarray,
    right_ee_xyz: np.ndarray,
    stage_id: np.ndarray,
    *,
    x_max: float,
    z_max: float,
    y_abs_max: float,
    min_frames: int,
    quantile: float,
) -> dict[str, float | bool | int]:
    # Evaluate symmetry only after entering lift/carry.
    mask = np.asarray(stage_id, dtype=np.int8) >= 2
    valid_frames = int(np.sum(mask))
    if valid_frames <= 0:
        return {
            "ee_sym_valid_frames": 0,
            "ee_sym_x_mean": float("inf"),
            "ee_sym_z_mean": float("inf"),
            "ee_sym_yabs_mean": float("inf"),
            "ee_sym_x_q": float("inf"),
            "ee_sym_z_q": float("inf"),
            "ee_sym_yabs_q": float("inf"),
            "ee_symmetry_pass": False,
        }

    err_x = np.abs(left_ee_xyz[:, 0] - right_ee_xyz[:, 0]).astype(np.float32)
    err_z = np.abs(left_ee_xyz[:, 2] - right_ee_xyz[:, 2]).astype(np.float32)
    err_yabs = np.abs(np.abs(left_ee_xyz[:, 1]) - np.abs(right_ee_xyz[:, 1])).astype(np.float32)

    ex = err_x[mask]
    ez = err_z[mask]
    ey = err_yabs[mask]
    q = float(np.clip(quantile, 0.0, 1.0))
    ex_q = float(np.quantile(ex, q))
    ez_q = float(np.quantile(ez, q))
    ey_q = float(np.quantile(ey, q))

    symmetry_pass = bool(
        (valid_frames >= int(min_frames))
        and (ex_q <= float(x_max))
        and (ez_q <= float(z_max))
        and (ey_q <= float(y_abs_max))
    )
    return {
        "ee_sym_valid_frames": valid_frames,
        "ee_sym_x_mean": float(np.mean(ex)),
        "ee_sym_z_mean": float(np.mean(ez)),
        "ee_sym_yabs_mean": float(np.mean(ey)),
        "ee_sym_x_q": ex_q,
        "ee_sym_z_q": ez_q,
        "ee_sym_yabs_q": ey_q,
        "ee_symmetry_pass": symmetry_pass,
    }


def evaluate_stage_completeness(stage_id: np.ndarray) -> dict[str, bool | list[int]]:
    required = {0, 1, 2, 3}
    present = set(int(v) for v in np.asarray(stage_id, dtype=np.int8).tolist())
    missing = sorted(required - present)
    return {
        "stage_has_all_4": len(missing) == 0,
        "stage_missing_ids": missing,
    }


def compute_stage_labels(
    contact_flag: np.ndarray,
    ee_mean_z: np.ndarray,
    ee_mean_speed: np.ndarray,
    *,
    lift_dz_min: float,
    lift_min_frames: int,
    carry_speed_max: float,
    carry_min_frames: int,
) -> tuple[np.ndarray, list[str]]:
    n = contact_flag.shape[0]
    stage_id = np.zeros((n,), dtype=np.int8)
    if n == 0:
        return stage_id, []

    # Monotonic stage machine:
    # free -> contact -> lift -> carry (no backward transitions).
    contact_idx = np.where(contact_flag.astype(bool))[0]
    if contact_idx.size == 0:
        stage_label = [STAGE_LABELS[int(v)] for v in stage_id.tolist()]
        return stage_id, stage_label

    first_contact = int(contact_idx[0])
    stage_id[first_contact:] = 1

    # Lift entry: after first contact, mean EE z rises above contact baseline.
    z0 = float(ee_mean_z[first_contact])
    dz = ee_mean_z - z0
    lift_rel = first_run_start(
        dz[first_contact:] >= float(lift_dz_min),
        min_len=max(1, int(lift_min_frames)),
    )
    if lift_rel is not None:
        lift_start = first_contact + int(lift_rel)
        stage_id[lift_start:] = 2

        # Carry entry: after lift starts, mean EE speed stays low.
        carry_rel = first_run_start(
            ee_mean_speed[lift_start:] <= float(carry_speed_max),
            min_len=max(1, int(carry_min_frames)),
        )
        if carry_rel is not None:
            carry_start = lift_start + int(carry_rel)
            stage_id[carry_start:] = 3

    stage_label = [STAGE_LABELS[int(v)] for v in stage_id.tolist()]
    return stage_id, stage_label


def process_one_episode(
    table: pa.Table,
    *,
    fk_context: FKContext,
    fps: float,
    t_enter: float,
    t_exit: float,
    enter_frames: int,
    exit_frames: int,
    both_contact_min_frames: int,
    lift_dz_min: float,
    lift_min_frames: int,
    carry_speed_max: float,
    carry_min_frames: int,
    ee_sym_x_max: float,
    ee_sym_z_max: float,
    ee_sym_yabs_max: float,
    ee_sym_min_frames: int,
    ee_sym_quantile: float,
) -> tuple[pa.Table, dict]:
    left_raw, right_raw = load_dual_arm_ft(table)
    n = left_raw.shape[0]
    if right_raw.shape[0] != n:
        raise ValueError(f"left/right length mismatch: {left_raw.shape} vs {right_raw.shape}")

    # Contact uses raw sensor-frame |ΔF_xyz| = ||F_t - F_{t-1}||.
    left_contact_signal = frame_diff_norm3(left_raw[:, :3])
    right_contact_signal = frame_diff_norm3(right_raw[:, :3])

    left_flag = detect_contact_hysteresis(
        left_contact_signal, t_enter=t_enter, t_exit=t_exit, enter_frames=enter_frames, exit_frames=exit_frames
    )
    right_flag = detect_contact_hysteresis(
        right_contact_signal, t_enter=t_enter, t_exit=t_exit, enter_frames=enter_frames, exit_frames=exit_frames
    )

    both_contact_raw = np.logical_and(left_flag == 1, right_flag == 1)
    both_contact_flag = keep_true_runs(both_contact_raw, min_len=both_contact_min_frames)
    # Exposed contact_flag is latched (no rollback): once in contact, always contact.
    contact_flag = latch_after_first_true(both_contact_flag).astype(np.int8)

    # FK kinematics for lift/carry stage transition.
    q32 = load_q32_series(table)
    if q32.shape[0] != n:
        raise ValueError(f"q32 length mismatch: q32={q32.shape[0]} vs ft={n}")
    left_ee_xyz, right_ee_xyz, ee_mean_xyz, ee_mean_speed = compute_ee_kinematics(q32, fk_context, fps=fps)
    ee_mean_z = ee_mean_xyz[:, 2] if ee_mean_xyz.shape[0] > 0 else np.zeros((0,), dtype=np.float32)

    stage_id, stage_label = compute_stage_labels(
        contact_flag,
        ee_mean_z,
        ee_mean_speed,
        lift_dz_min=lift_dz_min,
        lift_min_frames=lift_min_frames,
        carry_speed_max=carry_speed_max,
        carry_min_frames=carry_min_frames,
    )
    ee_sym = evaluate_ee_pose_symmetry(
        left_ee_xyz,
        right_ee_xyz,
        stage_id,
        x_max=ee_sym_x_max,
        z_max=ee_sym_z_max,
        y_abs_max=ee_sym_yabs_max,
        min_frames=ee_sym_min_frames,
        quantile=ee_sym_quantile,
    )
    stage_comp = evaluate_stage_completeness(stage_id)

    out = table
    out = upsert_column(out, "contact_flag", pa.array(contact_flag.tolist(), type=pa.int8()))
    out = upsert_stage_columns(out, stage_id, stage_label)
    out = upsert_ee_columns(out, left_ee_xyz, right_ee_xyz, ee_mean_xyz, ee_mean_speed)

    stats = {
        "frames": int(n),
        "left_contact_signal_mean": float(np.mean(left_contact_signal)) if n > 0 else 0.0,
        "right_contact_signal_mean": float(np.mean(right_contact_signal)) if n > 0 else 0.0,
        "contact_signal_mean": float(np.mean(0.5 * (left_contact_signal + right_contact_signal))) if n > 0 else 0.0,
        "contact_ratio": float(contact_flag.mean()) if n > 0 else 0.0,
        "both_contact_ratio": float(np.mean(both_contact_flag)) if n > 0 else 0.0,
        "stage_free_ratio": float(np.mean(stage_id == 0)) if n > 0 else 0.0,
        "stage_contact_ratio": float(np.mean(stage_id == 1)) if n > 0 else 0.0,
        "stage_lift_ratio": float(np.mean(stage_id == 2)) if n > 0 else 0.0,
        "stage_carry_ratio": float(np.mean(stage_id == 3)) if n > 0 else 0.0,
        "ee_mean_z_delta": float(np.max(ee_mean_z) - np.min(ee_mean_z)) if n > 0 else 0.0,
        "ee_speed_mean": float(np.mean(ee_mean_speed)) if n > 0 else 0.0,
        "ee_speed_max": float(np.max(ee_mean_speed)) if n > 0 else 0.0,
    }
    stats.update(ee_sym)
    stats.update(stage_comp)
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="FT clean pipeline (sensor-frame core mode)")
    parser.add_argument(
        "--input-dir",
        default="/home/SENSETIME/yanzichen/data/file/dataset/0206_ft_train",
        help="Input dataset root or chunk dir",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output chunk dir; default auto-generated as *_clean/.../chunk-000",
    )
    parser.add_argument(
        "--dry-run-stats",
        action="store_true",
        help="Only generate report/stats, do not write cleaned parquet files",
    )

    # Core-exposed controls
    parser.add_argument("--t-enter", type=float, default=3.0, help="Contact enter threshold on |ΔF_xyz|")
    parser.add_argument("--t-exit", type=float, default=1.5, help="Contact exit threshold on |ΔF_xyz|")
    parser.add_argument(
        "--both-contact-min-frames",
        type=int,
        default=1,
        help="Min consecutive both-hand contact frames",
    )
    parser.add_argument(
        "--lift-dz-min",
        type=float,
        default=0.015,
        help="Lift enter threshold on mean EE z rise after first contact (m)",
    )
    parser.add_argument(
        "--lift-min-frames",
        type=int,
        default=3,
        help="Min consecutive frames for lift stage",
    )
    parser.add_argument(
        "--carry-speed-max",
        type=float,
        default=0.03,
        help="Carry enter threshold on mean EE speed (m/s)",
    )
    parser.add_argument(
        "--carry-min-frames",
        type=int,
        default=8,
        help="Min consecutive low-speed frames for carry stage",
    )
    parser.add_argument(
        "--enable-ee-symmetry-filter",
        dest="enable_ee_symmetry_filter",
        action="store_true",
        default=True,
        help="Enable EE pose symmetry filtering (default: enabled)",
    )
    parser.add_argument(
        "--disable-ee-symmetry-filter",
        dest="enable_ee_symmetry_filter",
        action="store_false",
        help="Disable EE pose symmetry filtering",
    )
    parser.add_argument(
        "--ee-sym-x-max",
        type=float,
        default=0.06,
        help="Max allowed q(err_x)=q(|xL-xR|) in lift/carry frames (m)",
    )
    parser.add_argument(
        "--ee-sym-z-max",
        type=float,
        default=0.05,
        help="Max allowed q(err_z)=q(|zL-zR|) in lift/carry frames (m)",
    )
    parser.add_argument(
        "--ee-sym-yabs-max",
        type=float,
        default=0.06,
        help="Max allowed q(err_|y|)=q(||yL|-|yR||) in lift/carry frames (m)",
    )
    parser.add_argument(
        "--ee-sym-min-frames",
        type=int,
        default=6,
        help="Min lift/carry frames required for EE symmetry evaluation",
    )
    parser.add_argument(
        "--ee-sym-quantile",
        type=float,
        default=0.85,
        help="Quantile for EE symmetry errors (0~1)",
    )

    # Advanced knobs (hidden)
    parser.add_argument("--fps", type=float, default=30.0, help=argparse.SUPPRESS)
    parser.add_argument("--enter-frames", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--exit-frames", type=int, default=3, help=argparse.SUPPRESS)

    # Compatibility args from old FK/world-force version
    parser.add_argument("--enable-world-force", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--sensor-on", default="hand_base", help=argparse.SUPPRESS)
    parser.add_argument("--sensor-offset-left", default="0,0,-0.04", help=argparse.SUPPRESS)
    parser.add_argument("--sensor-offset-right", default="0,0,-0.04", help=argparse.SUPPRESS)
    parser.add_argument("--kin-common-path", default=DEFAULT_KIN_COMMON_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--urdf-path", default=DEFAULT_URDF_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--left-force-link", default="left_tcp_link", help=argparse.SUPPRESS)
    parser.add_argument("--right-force-link", default="right_tcp_link", help=argparse.SUPPRESS)
    parser.add_argument("--sensor-rpy-left", default="0,0,0", help=argparse.SUPPRESS)
    parser.add_argument("--sensor-rpy-right", default="0,0,0", help=argparse.SUPPRESS)

    args = parser.parse_args()
    if not (0.0 <= float(args.ee_sym_quantile) <= 1.0):
        raise SystemExit("--ee-sym-quantile must be in [0, 1]")
    dry_run_stats = bool(args.dry_run_stats)
    in_path = Path(args.input_dir)
    chunk_dir = resolve_chunk_dir(in_path)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(in_path, chunk_dir)
    if not dry_run_stats:
        out_dir.mkdir(parents=True, exist_ok=True)
    fk_context = build_fk_context(
        kin_common_path=Path(args.kin_common_path),
        urdf_path=Path(args.urdf_path),
        left_ee_link=args.left_force_link,
        right_ee_link=args.right_force_link,
    )

    parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode_*.parquet in {chunk_dir}")

    print("=" * 80)
    print("FT clean pipeline (sensor-frame)")
    print(f"input : {chunk_dir}")
    print(f"output: {out_dir}")
    print(f"episodes: {len(parquet_files)}")
    print(f"dry_run_stats: {dry_run_stats}")
    print("contact signal: raw sensor-frame |ΔF_xyz| = ||F_t - F_{t-1}|| (no bias/lowpass)")
    print("stage transition: contact->lift by EE z rise, lift->carry by EE low speed (FK)")
    print(f"ee_symmetry_filter: {bool(args.enable_ee_symmetry_filter)}")
    print("stage_completeness_filter: require free/contact/lift/carry all present")
    print("=" * 80)

    report_rows: list[dict] = []
    kept = 0
    dropped = 0
    fail_reason_counts = {
        "ee_symmetry_failed": 0,
        "stage_incomplete": 0,
        "both_failed": 0,
    }

    for pq_file in tqdm(parquet_files, desc="Process FT episodes", unit="episode"):
        table = pq.read_table(str(pq_file))
        cleaned, stats = process_one_episode(
            table,
            fk_context=fk_context,
            fps=args.fps,
            t_enter=args.t_enter,
            t_exit=args.t_exit,
            enter_frames=args.enter_frames,
            exit_frames=args.exit_frames,
            both_contact_min_frames=args.both_contact_min_frames,
            lift_dz_min=args.lift_dz_min,
            lift_min_frames=args.lift_min_frames,
            carry_speed_max=args.carry_speed_max,
            carry_min_frames=args.carry_min_frames,
            ee_sym_x_max=args.ee_sym_x_max,
            ee_sym_z_max=args.ee_sym_z_max,
            ee_sym_yabs_max=args.ee_sym_yabs_max,
            ee_sym_min_frames=args.ee_sym_min_frames,
            ee_sym_quantile=args.ee_sym_quantile,
        )

        out_name = f"{pq_file.stem}_clean.parquet"
        ee_sym_pass = (not bool(args.enable_ee_symmetry_filter)) or bool(stats.get("ee_symmetry_pass", False))
        stage_complete_pass = bool(stats.get("stage_has_all_4", False))
        keep = bool(ee_sym_pass and stage_complete_pass)
        fail_reasons: list[str] = []
        if not ee_sym_pass:
            fail_reasons.append("ee_symmetry_failed")
        if not stage_complete_pass:
            fail_reasons.append("stage_incomplete")

        if not keep:
            if len(fail_reasons) >= 2:
                fail_reason_counts["both_failed"] += 1
            elif fail_reasons[0] in fail_reason_counts:
                fail_reason_counts[fail_reasons[0]] += 1

        if keep:
            if not dry_run_stats:
                pq.write_table(cleaned, out_dir / out_name)
            kept += 1
        else:
            dropped += 1

        report_rows.append(
            {
                "episode": pq_file.name,
                "output": out_name,
                "kept": bool(keep),
                "filter_pass": bool(keep),
                "filter_fail_reasons": fail_reasons,
                "filter_eval": {
                    "ee_symmetry_pass": bool(stats.get("ee_symmetry_pass", False)),
                    "stage_has_all_4": bool(stats.get("stage_has_all_4", False)),
                },
                **stats,
            }
        )

    report = {
        "input_chunk_dir": str(chunk_dir),
        "output_chunk_dir": str(out_dir),
        "num_input_episodes": len(parquet_files),
        "num_kept_episodes": kept,
        "num_dropped_episodes": dropped,
        "filter_summary": {
            "enabled": {
                "ee_symmetry_filter": bool(args.enable_ee_symmetry_filter),
                "stage_completeness_filter": True,
            },
            "dropped_by_reason": fail_reason_counts,
        },
        "params": {
            "contact_signal": "raw_sensor_dforce_xyz_norm",
            "t_enter": args.t_enter,
            "t_exit": args.t_exit,
            "enter_frames": args.enter_frames,
            "exit_frames": args.exit_frames,
            "both_contact_min_frames": args.both_contact_min_frames,
            "lift_dz_min": args.lift_dz_min,
            "lift_min_frames": args.lift_min_frames,
            "carry_speed_max": args.carry_speed_max,
            "carry_min_frames": args.carry_min_frames,
            "enable_ee_symmetry_filter": bool(args.enable_ee_symmetry_filter),
            "ee_sym_x_max": args.ee_sym_x_max,
            "ee_sym_z_max": args.ee_sym_z_max,
            "ee_sym_yabs_max": args.ee_sym_yabs_max,
            "ee_sym_min_frames": args.ee_sym_min_frames,
            "ee_sym_quantile": args.ee_sym_quantile,
            "require_all_4_stages": True,
            "fps": args.fps,
            "filters_enabled": bool(args.enable_ee_symmetry_filter),
            "kin_common_path": str(args.kin_common_path),
            "urdf_path": str(args.urdf_path),
            "left_ee_link": args.left_force_link,
            "right_ee_link": args.right_force_link,
            "dry_run_stats": dry_run_stats,
        },
        "episodes": report_rows,
    }
    report_path = out_dir.parent.parent / "ft_clean_report.json" if out_dir.name == "chunk-000" else out_dir / "ft_clean_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Kept episodes   : {kept}")
    print(f"Dropped episodes: {dropped}")
    if dry_run_stats:
        print("Parquet writing : skipped (--dry-run-stats)")
    print(f"Report          : {report_path}")


if __name__ == "__main__":
    main()
