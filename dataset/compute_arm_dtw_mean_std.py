from __future__ import annotations

"""Compute DTW-aligned mean/std trajectories for a dual-arm task.

This script aligns all DOFs except the left/right hands to a reference
trajectory using DTW, resamples to a fixed length, and then reconstructs
full 32D outputs by filling hand joints with the mean initial values.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from dtaidistance import dtw

# ========================
# 配置
# ========================

# 32维动作映射（索引 -> 关节/部位）
# 头部(2): 0 head_pitch, 1 head_yaw
# 左臂(7): 2 left_shoulder_pitch, 3 left_shoulder_roll, 4 left_shoulder_yaw,
#         5 left_elbow_pitch, 6 left_wrist_yaw, 7 left_wrist_pitch, 8 left_wrist_roll
# 左手(6): 9 left_little_finger, 10 left_ring_finger, 11 left_middle_finger,
#         12 left_fore_finger, 13 left_thumb_bend, 14 left_thumb_rotation
# 右臂(7): 15 right_shoulder_pitch, 16 right_shoulder_roll, 17 right_shoulder_yaw,
#         18 right_elbow_pitch, 19 right_wrist_yaw, 20 right_wrist_pitch, 21 right_wrist_roll
# 右手(6): 22 right_little_finger, 23 right_ring_finger, 24 right_middle_finger,
#         25 right_fore_finger, 26 right_thumb_bend, 27 right_thumb_rotation
# 腰部(2): 28 waist_yaw, 29 waist_pitch
# 腿部(2): 30 hip_pitch, 31 knee_pitch

# DATA_DIR = Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/grab_stool_cut_train/data/chunk-000")
# OUT_DIR = Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/pick_mean_std/chunk-000_dtw")
DATA_DIR = Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/putbox_combined_cut/chunk-000")
OUT_DIR = Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/putbox_combined_cut/chunk-000_dtw")
TARGET_LEN = 100

# Outputs
OUT_MEAN = OUT_DIR / "mean_traj_32d_dtw100_trim.npy"
OUT_STD = OUT_DIR / "std_traj_32d_dtw100_trim.npy"
OUT_SYMMETRY = OUT_DIR / "symmetry_stats_dtw100_trim.csv"
OUT_ALIGNED = OUT_DIR / "aligned_all_full_dtw100_trim.npy"

# 对除双手外的所有维度做对齐：固定手部(9-14, 22-27)
HAND_IDXS = np.array(list(range(9, 15)) + list(range(22, 28)), dtype=np.int64)
ALIGN_IDXS = np.array([i for i in range(32) if i not in set(HAND_IDXS.tolist())], dtype=np.int64)
FIXED_IDXS = HAND_IDXS
TOTAL_DIMS = 32
LEFT_ARM_IDXS = np.array(list(range(2, 9)), dtype=np.int64)
RIGHT_ARM_IDXS = np.array(list(range(15, 22)), dtype=np.int64)
ARM_JOINT_NAMES = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
]

# ========================
# 数据加载
# ========================

def load_all_trajectories(data_dir):
    """Load all action trajectories and extract non-hand sequences."""
    trajs = []
    fixed_inits = []
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    print(f"Found {len(parquet_files)} episodes")

    for pq_file in tqdm(parquet_files, desc="Load episodes"):
        table = pq.read_table(pq_file, columns=["action"])
        data = table.to_pydict()
        actions_full = np.array(data["action"], dtype=np.float32)
        fixed_inits.append(actions_full[0, FIXED_IDXS])
        trajs.append(actions_full[:, ALIGN_IDXS])

    return trajs, np.stack(fixed_inits, axis=0)

# ========================
# 参考轨迹选择
# ========================

def pick_reference(trajs):
    """Pick the median-length trajectory as DTW reference."""
    lengths = [t.shape[0] for t in trajs]
    median_idx = np.argsort(lengths)[len(lengths)//2]
    ref = trajs[median_idx]
    print(f"Reference length: {ref.shape[0]}")
    return ref

# ========================
# DTW 对齐与重采样
# ========================

def dtw_align_and_resample(traj, ref, target_len=100):
    """DTW-align one trajectory to the reference and resample to target_len."""
    # use_ndim=True enables multivariate DTW
    path = dtw.warping_path(traj, ref, use_ndim=True)

    path = np.array(path)

    aligned = [[] for _ in range(ref.shape[0])]
    for i, j in path:
        aligned[j].append(traj[i])

    aligned_traj = []
    for group in aligned:
        if len(group) == 0:
            aligned_traj.append(aligned_traj[-1])
        else:
            aligned_traj.append(np.mean(group, axis=0))

    aligned_traj = np.stack(aligned_traj, axis=0)

    # Resample to a fixed length
    x_old = np.linspace(0, 1, aligned_traj.shape[0])
    x_new = np.linspace(0, 1, target_len)

    resampled = np.zeros((target_len, aligned_traj.shape[1]))
    for d in range(aligned_traj.shape[1]):
        resampled[:, d] = np.interp(x_new, x_old, aligned_traj[:, d])

    return resampled


def dtw_alignment_groups(traj, ref):
    """Return list of index groups mapping ref time to source indices."""
    path = dtw.warping_path(traj, ref, use_ndim=True)
    groups = [[] for _ in range(ref.shape[0])]
    for i, j in path:
        groups[j].append(i)
    return groups


def align_continuous(seq: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    aligned = []
    for grp in groups:
        if not grp:
            aligned.append(aligned[-1])
        else:
            aligned.append(np.mean(seq[grp], axis=0))
    return np.stack(aligned, axis=0)


def resample_continuous(aligned: np.ndarray, target_len: int) -> np.ndarray:
    if aligned.ndim == 1:
        aligned = aligned[:, None]
        squeeze = True
    else:
        squeeze = False

    x_old = np.linspace(0, 1, aligned.shape[0])
    x_new = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, aligned.shape[1]), dtype=np.float32)
    for d in range(aligned.shape[1]):
        resampled[:, d] = np.interp(x_new, x_old, aligned[:, d])

    return resampled[:, 0] if squeeze else resampled


def align_and_resample_discrete(seq: list, groups: list[list[int]], target_len: int) -> list:
    if not seq:
        return []
    aligned_idx = []
    for grp in groups:
        if not grp:
            aligned_idx.append(aligned_idx[-1])
        else:
            aligned_idx.append(grp[len(grp) // 2])
    resample_idx = np.round(np.linspace(0, len(aligned_idx) - 1, target_len)).astype(int)
    return [seq[aligned_idx[i]] for i in resample_idx]

# ========================
# 双臂对称性分析
# ========================

def analyze_arm_symmetry(aligned_all_full, *, output_csv):
    """Compute left-right arm symmetry based on abs(|L| - |R|)."""
    left_arm = aligned_all_full[:, :, LEFT_ARM_IDXS]
    right_arm = aligned_all_full[:, :, RIGHT_ARM_IDXS]
    abs_diff = np.abs(np.abs(left_arm) - np.abs(right_arm))
    abs_diff_mean = abs_diff.mean(axis=(0, 1))
    abs_diff_std = abs_diff.std(axis=(0, 1))

    symmetry_table = np.stack([abs_diff_mean, abs_diff_std], axis=1)
    header = "joint,abs_diff_mean,abs_diff_std"
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for name, row in zip(ARM_JOINT_NAMES, symmetry_table):
            f.write(f"{name},{row[0]:.6f},{row[1]:.6f}\n")

    print("\nSymmetry (left vs right) saved to:", output_csv)

# ========================
# 主流程
# ========================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trajs, fixed_inits = load_all_trajectories(DATA_DIR)
    ref = pick_reference(trajs)
    # Mean initial pose for non-arm joints, used to fill the fixed dimensions.
    fixed_init_mean = fixed_inits.mean(axis=0)

    aligned_all = []

    for traj in tqdm(trajs, desc="DTW align"):
        aligned = dtw_align_and_resample(traj, ref, TARGET_LEN)
        aligned_all.append(aligned)

    aligned_all = np.stack(aligned_all, axis=0)  # [N, 100, D_non_hand]

    mean_reduced = aligned_all.mean(axis=0)
    std_reduced = aligned_all.std(axis=0)

    mean_traj = np.zeros((TARGET_LEN, TOTAL_DIMS), dtype=np.float32)
    std_traj = np.zeros((TARGET_LEN, TOTAL_DIMS), dtype=np.float32)
    mean_traj[:, ALIGN_IDXS] = mean_reduced
    std_traj[:, ALIGN_IDXS] = std_reduced
    mean_traj[:, FIXED_IDXS] = fixed_init_mean[None, :]

    np.save(OUT_MEAN, mean_traj)
    np.save(OUT_STD, std_traj)

    print("\n=== DONE ===")
    print("Mean:", OUT_MEAN, mean_traj.shape)
    print("Std :", OUT_STD, std_traj.shape)

    # Reconstruct full 32D aligned trajectories for downstream plotting.
    aligned_all_full = np.zeros((aligned_all.shape[0], TARGET_LEN, TOTAL_DIMS), dtype=np.float32)
    aligned_all_full[:, :, ALIGN_IDXS] = aligned_all
    aligned_all_full[:, :, FIXED_IDXS] = fixed_init_mean[None, None, :]

    np.save(OUT_ALIGNED, aligned_all_full)
    analyze_arm_symmetry(aligned_all_full, output_csv=OUT_SYMMETRY)

    print("Aligned trajectories:", OUT_ALIGNED, aligned_all_full.shape)

    # ---------- Per-episode DTW align + resample ----------
    parquet_files = sorted(DATA_DIR.glob("episode_*.parquet"))
    for pq_file in tqdm(parquet_files, desc="Write DTW episodes"):
        table = pq.read_table(pq_file)
        data = table.to_pydict()

        actions = np.asarray(data["action"], dtype=np.float32)
        align_traj = actions[:, ALIGN_IDXS]
        groups = dtw_alignment_groups(align_traj, ref)

        resampled_actions = resample_continuous(align_continuous(actions, groups), TARGET_LEN)
        # Keep hand DOFs fixed to the mean initial values.
        resampled_actions[:, HAND_IDXS] = fixed_init_mean[None, :]

        out_data = {"action": resampled_actions.tolist()}

        if "observation.state" in data:
            state = np.asarray(data["observation.state"], dtype=np.float32)
            out_data["observation.state"] = resample_continuous(align_continuous(state, groups), TARGET_LEN).tolist()
        if "observation.velocity" in data:
            velocity = np.asarray(data["observation.velocity"], dtype=np.float32)
            out_data["observation.velocity"] = resample_continuous(
                align_continuous(velocity, groups), TARGET_LEN
            ).tolist()
        if "timestamp" in data:
            ts = np.asarray(data["timestamp"], dtype=np.float32)
            out_data["timestamp"] = resample_continuous(align_continuous(ts, groups), TARGET_LEN).tolist()
        if "frame_index" in data:
            out_data["frame_index"] = list(range(TARGET_LEN))
        if "index" in data:
            out_data["index"] = list(range(TARGET_LEN))
        if "episode_index" in data:
            out_data["episode_index"] = [data["episode_index"][0]] * TARGET_LEN
        if "task_index" in data:
            out_data["task_index"] = [data["task_index"][0]] * TARGET_LEN
        if "observation.images.cam_high" in data:
            imgs = data["observation.images.cam_high"]
            out_data["observation.images.cam_high"] = align_and_resample_discrete(imgs, groups, TARGET_LEN)

        out_table = pa.Table.from_pydict(out_data)
        pq.write_table(out_table, OUT_DIR / pq_file.name)

if __name__ == "__main__":
    main()
