from __future__ import annotations

"""Query DTW mean/std to build a flow-matching warm start from current arm pose."""
"""python warm_start_from_dtw.py --chunk-size 16"""

import argparse
from pathlib import Path

import numpy as np

# DEFAULT_OUT_DIR = Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/putbox_combined_cut/chunk-000_dtw")
DEFAULT_OUT_DIR = Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/grab_stool_wrist_cut/chunk-000_dtw")

# Arm joint indices in the 32D action/state.
LEFT_ARM_IDXS = np.array(list(range(2, 9)), dtype=np.int64)
RIGHT_ARM_IDXS = np.array(list(range(15, 22)), dtype=np.int64)
ARM_IDXS = np.concatenate([LEFT_ARM_IDXS, RIGHT_ARM_IDXS], axis=0)

# Shoulder -> wrist decreasing weights (7 joints).
ARM_WEIGHTS = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
DEFAULT_MEAN = DEFAULT_OUT_DIR / "mean_traj_32d_dtw100_trim.npy"
DEFAULT_STD = DEFAULT_OUT_DIR / "std_traj_32d_dtw100_trim.npy"


def load_stats(mean_path: Path, std_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mean_traj = np.load(mean_path)
    std_traj = np.load(std_path)
    if mean_traj.shape != std_traj.shape:
        raise ValueError(f"mean/std shape mismatch: {mean_traj.shape} vs {std_traj.shape}")
    if mean_traj.ndim != 2 or mean_traj.shape[1] != 32:
        raise ValueError(f"expected mean/std shape (T, 32), got {mean_traj.shape}")
    return mean_traj, std_traj


def nearest_frame_index(
    mean_traj: np.ndarray, current_pose_32d: np.ndarray, *, arm_weights: np.ndarray
) -> int:
    """Find the closest frame by weighted arm distance."""
    current_pose_32d = np.asarray(current_pose_32d, dtype=np.float32)
    if current_pose_32d.shape != (32,):
        raise ValueError(f"current_pose_32d must be shape (32,), got {current_pose_32d.shape}")

    arm_target = current_pose_32d[ARM_IDXS]
    arm_mean = mean_traj[:, ARM_IDXS]

    # Apply per-joint weights to left and right arms.
    weights = np.concatenate([arm_weights, arm_weights], axis=0)
    diff = arm_mean - arm_target[None, :]
    weighted = diff * weights[None, :]
    dist = np.linalg.norm(weighted, axis=1)
    return int(np.argmin(dist))


def warm_start_from_stats(
    mean_traj: np.ndarray,
    std_traj: np.ndarray,
    current_pose_32d: np.ndarray,
    *,
    noise_scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Return a warm-start 32D pose and the matched frame index."""
    if rng is None:
        rng = np.random.default_rng()

    idx = nearest_frame_index(mean_traj, current_pose_32d, arm_weights=ARM_WEIGHTS)
    base = mean_traj[idx]
    noise = rng.standard_normal(size=base.shape).astype(np.float32)
    warm_start = base + noise * std_traj[idx] * noise_scale
    return warm_start.astype(np.float32), idx


def warm_start_chunk_from_stats(
    mean_traj: np.ndarray,
    std_traj: np.ndarray,
    current_pose_32d: np.ndarray,
    *,
    chunk_size: int,
    noise_scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Return a warm-start chunk (chunk_size, 32) and the matched frame index."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if rng is None:
        rng = np.random.default_rng()

    idx = nearest_frame_index(mean_traj, current_pose_32d, arm_weights=ARM_WEIGHTS)
    start = idx
    end = start + chunk_size
    if end > mean_traj.shape[0]:
        end = mean_traj.shape[0]
        start = max(0, end - chunk_size)

    base = mean_traj[start:end]
    noise = rng.standard_normal(size=base.shape).astype(np.float32)
    warm_start = base + noise * std_traj[start:end] * noise_scale
    return warm_start.astype(np.float32), idx


def plot_warm_start(
    mean_traj: np.ndarray,
    std_traj: np.ndarray,
    warm_start: np.ndarray,
    *,
    idx: int,
    out_path: Path | None,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    if warm_start.ndim == 1:
        warm_start = warm_start[None, :]

    def _plot_one_arm(name: str, arm_idxs: np.ndarray, labels: list[str], save_path: Path | None) -> None:
        arm_series = warm_start[:, arm_idxs]  # (T, 7)
        t = np.arange(arm_series.shape[0])

        fig, ax = plt.subplots(figsize=(8, 4))
        for i, label in enumerate(labels):
            ax.plot(t, arm_series[:, i], label=label)
        ax.set_xlabel("frame")
        ax.set_ylabel("value")
        ax.set_title(f"{name} arm warm start (frame {idx}, chunk={arm_series.shape[0]})")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    left_labels = [f"L{i}" for i in range(7)]
    right_labels = [f"R{i}" for i in range(7)]

    if out_path is not None:
        stem = out_path.stem
        suffix = out_path.suffix or ".png"
        left_path = out_path.with_name(f"{stem}_left{suffix}")
        right_path = out_path.with_name(f"{stem}_right{suffix}")
    else:
        left_path = None
        right_path = None

    _plot_one_arm("Left", LEFT_ARM_IDXS, left_labels, left_path)
    _plot_one_arm("Right", RIGHT_ARM_IDXS, right_labels, right_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start query from DTW mean/std stats.")
    parser.add_argument("--mean", default=str(DEFAULT_MEAN))
    parser.add_argument("--std", default=str(DEFAULT_STD))
    parser.add_argument("--current", help="Path to .npy with shape (32,)")
    parser.add_argument("--chunk-size", type=int, default=16, help="Number of frames to output.")
    parser.add_argument(
        "--output-mode",
        choices=["abs", "rel"],
        default="rel",
        help="abs: output absolute warm start; rel: output frame-to-frame delta (t - t-1).",
    )
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for all files.")
    parser.add_argument("--out", help="Optional .npy output filename (relative to out-dir if not absolute).")
    parser.add_argument("--plot", action="store_true", help="Show a warm start plot.")
    parser.add_argument("--plot-out", help="Optional plot filename (relative to out-dir if not absolute).")
    args = parser.parse_args()

    mean_traj, std_traj = load_stats(Path(args.mean), Path(args.std))
    if args.current:
        current_pose = np.load(args.current)
    else:
        current_pose = np.asarray(mean_traj[0], dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.chunk_size == 1:
        warm_start, idx = warm_start_from_stats(
            mean_traj, std_traj, current_pose, noise_scale=args.noise_scale, rng=rng
        )
    else:
        warm_start, idx = warm_start_chunk_from_stats(
            mean_traj,
            std_traj,
            current_pose,
            chunk_size=args.chunk_size,
            noise_scale=args.noise_scale,
            rng=rng,
        )
    if args.output_mode == "rel":
        if warm_start.ndim == 2:
            rel = np.zeros_like(warm_start)
            rel[1:] = warm_start[1:] - warm_start[:-1]
            warm_start = rel
        else:
            warm_start = np.zeros_like(warm_start)
    print(f"Matched frame index: {idx}")
    print("Warm start:")
    print(warm_start)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = out_dir / out_path
    else:
        out_path = out_dir / "warm_start.npy"
    np.save(out_path, warm_start)
    print(f"Warm start saved to: {out_path}")

    if args.plot or args.plot_out:
        if args.plot_out:
            plot_path = Path(args.plot_out)
            if not plot_path.is_absolute():
                plot_path = out_dir / plot_path
        else:
            plot_path = out_dir / "warm_start.png"
        plot_warm_start(mean_traj, std_traj, warm_start, idx=idx, out_path=plot_path, show=args.plot)


if __name__ == "__main__":
    main()
