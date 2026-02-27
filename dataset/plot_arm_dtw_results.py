"""Plot per-DOF trajectories and arm symmetry from precomputed outputs."""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ========================
# 配置
# ========================

OUT_MEAN = "/home/SENSETIME/yanzichen/data/file/openpi/dataset/mean/mean_traj_32d_dtw100_trim.npy"
OUT_STD = "/home/SENSETIME/yanzichen/data/file/openpi/dataset/mean/std_traj_32d_dtw100_trim.npy"
OUT_ALIGNED = "/home/SENSETIME/yanzichen/data/file/openpi/dataset/mean/aligned_all_full_dtw100_trim.npy"
OUT_SYMMETRY = "symmetry_stats_dtw100_trim.csv"
OUT_PLOT_DIR = Path("plots_per_dof_dtw100_trim")
OUT_PLOT_DIR.mkdir(exist_ok=True)
OUT_SYMMETRY_PLOT = OUT_PLOT_DIR / "symmetry_barplot.png"

TOTAL_DIMS = 32
ARM_JOINT_NAMES = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
]

DOF_INFO = [
    ("head_pitch", "head"),
    ("head_yaw", "head"),
    ("left_shoulder_pitch", "left_arm"),
    ("left_shoulder_roll", "left_arm"),
    ("left_shoulder_yaw", "left_arm"),
    ("left_elbow_pitch", "left_arm"),
    ("left_wrist_yaw", "left_arm"),
    ("left_wrist_pitch", "left_arm"),
    ("left_wrist_roll", "left_arm"),
    ("left_little_finger", "left_hand"),
    ("left_ring_finger", "left_hand"),
    ("left_middle_finger", "left_hand"),
    ("left_fore_finger", "left_hand"),
    ("left_thumb_bend", "left_hand"),
    ("left_thumb_rotation", "left_hand"),
    ("right_shoulder_pitch", "right_arm"),
    ("right_shoulder_roll", "right_arm"),
    ("right_shoulder_yaw", "right_arm"),
    ("right_elbow_pitch", "right_arm"),
    ("right_wrist_yaw", "right_arm"),
    ("right_wrist_pitch", "right_arm"),
    ("right_wrist_roll", "right_arm"),
    ("right_little_finger", "right_hand"),
    ("right_ring_finger", "right_hand"),
    ("right_middle_finger", "right_hand"),
    ("right_fore_finger", "right_hand"),
    ("right_thumb_bend", "right_hand"),
    ("right_thumb_rotation", "right_hand"),
    ("waist_yaw", "waist"),
    ("waist_pitch", "waist"),
    ("hip_pitch", "leg"),
    ("knee_pitch", "leg"),
]


def plot_per_dof(aligned_all_full, mean_traj):
    """Plot per-DOF traces with all trajectories and the mean."""
    T = mean_traj.shape[0]
    x = np.arange(T)

    for d in range(TOTAL_DIMS):
        plt.figure(figsize=(6, 4))
        for i in range(aligned_all_full.shape[0]):
            plt.plot(x, aligned_all_full[i, :, d], alpha=0.1, color="gray")
        plt.plot(x, mean_traj[:, d], linewidth=2, color="black")
        name, part = DOF_INFO[d]
        plt.title(f"DOF {d} - {part} - {name}")
        plt.xlabel("Task phase (0~100)")
        plt.ylabel("Joint value")
        plt.tight_layout()
        plt.savefig(OUT_PLOT_DIR / f"dof_{d:02d}.png")
        plt.close()

    print("Saved per-DOF plots to:", OUT_PLOT_DIR.resolve())


def plot_symmetry_barplot(csv_path, output_plot):
    """Plot symmetry stats (abs diff mean/std) per joint."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    abs_diff_mean = data["abs_diff_mean"]
    abs_diff_std = data["abs_diff_std"]
    x = np.arange(len(ARM_JOINT_NAMES))

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(x - 0.2, abs_diff_mean, width=0.4, label="abs diff mean", color="red")
    ax1.bar(x + 0.2, abs_diff_std, width=0.4, label="abs diff std", color="green")
    ax1.set_ylabel("Abs diff mean/std")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ARM_JOINT_NAMES, rotation=30, ha="right")
    ax1.set_title("Left vs Right Arm Symmetry (Abs Diff)")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(output_plot)
    plt.close(fig)
    print("Symmetry plot saved to:", output_plot)


def main():
    """Entry point for generating all plots."""
    mean_traj = np.load(OUT_MEAN)
    _ = np.load(OUT_STD)
    aligned_all_full = np.load(OUT_ALIGNED)

    plot_per_dof(aligned_all_full, mean_traj)
    plot_symmetry_barplot(OUT_SYMMETRY, OUT_SYMMETRY_PLOT)


if __name__ == "__main__":
    main()
