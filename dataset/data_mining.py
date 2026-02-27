"""
Episode prefix trimming (joint-change):
- Cut at the first time BOTH image and action change for K consecutive steps.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# DEFAULT_DATA_ROOT = Path("/home/SENSETIME/yanzichen/data/file/dataset/putbox_cut_test/chunk-000")
# DEFAULT_OUT_DIR = Path("/home/SENSETIME/yanzichen/data/file/dataset/putbox_cut/chunk-000")
# DEFAULT_DATA_ROOT = Path("/home/SENSETIME/yanzichen/data/file/dataset/putbox_cut_test/pick")
# DEFAULT_OUT_DIR = Path("/home/SENSETIME/yanzichen/data/file/dataset/putbox_cut/pick")
DEFAULT_DATA_ROOT = Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/putbox_combined/data/chunk-000")
DEFAULT_OUT_DIR = Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/putbox_combined_cut/chunk-000")
CAM_KEY = "observation.images.cam_high"
ACTION_KEY = "action"
ARM_IDXS = np.array(list(range(2, 9)) + list(range(15, 22)), dtype=np.int64)

# Per-task parameter presets (joint-change mode).
TASK_PRESETS = {
    "pick": {
        "hash_size": 8,
        "frame_step": 1,
        "image_change_threshold": 1.0,
        "action_percentile": 85.0,
        "min_consistent_steps": 3,
    },
    "put": {
        "hash_size": 8,
        "frame_step": 1,
        "image_change_threshold": 0.0,
        "action_percentile": 80.0,
        "min_consistent_steps": 3,
    },
}

# =========================
# Utils
# =========================

def _try_import_pil():
    try:
        from PIL import Image
        return Image
    except Exception:
        return None


def ahash_digest(img_struct: dict, *, hash_size: int, cache: dict[str, str]) -> str | None:
    if img_struct is None:
        return None
    img_bytes = img_struct.get("bytes")
    img_path = img_struct.get("path")

    cache_key = None
    if img_bytes:
        cache_key = hashlib.sha1(img_bytes).hexdigest()
    elif img_path:
        cache_key = f"path:{img_path}"

    if cache_key and cache_key in cache:
        return cache[cache_key]

    image_mod = _try_import_pil()
    if image_mod is None:
        raise RuntimeError("PIL not available; install pillow for ahash")

    if img_bytes:
        img = image_mod.open(io.BytesIO(img_bytes)).convert("L")
    elif img_path:
        img = image_mod.open(img_path).convert("L")
    else:
        return None

    img = img.resize((hash_size, hash_size))
    arr = np.asarray(img, dtype=np.float32)
    mean = arr.mean()
    bits = arr > mean
    digest = "".join("1" if b else "0" for b in bits.flatten())
    if cache_key:
        cache[cache_key] = digest
    return digest


def hamming_distance(a: str | None, b: str | None) -> int | None:
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return None
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


# =========================
# Step 1: Auto calibrate action diff
# =========================

def collect_action_diffs(root: Path):
    diffs = []
    parquet_files = sorted(root.glob("episode_*.parquet"))
    print(f"Scanning {len(parquet_files)} episodes for action diff stats...")

    for pq_file in tqdm(parquet_files):
        table = pq.read_table(pq_file, columns=[ACTION_KEY])
        actions = np.asarray(table.column(0).to_pylist(), dtype=np.float32)
        for t in range(1, len(actions)):
            diff = np.abs(actions[t, ARM_IDXS] - actions[t - 1, ARM_IDXS])
            diffs.extend(diff.tolist())

    diffs = np.array(diffs)
    return diffs


# =========================
# Step 2: Trim logic
# =========================

def compute_trim_start(
    images: list[dict],
    actions: np.ndarray,
    *,
    hash_size: int,
    frame_step: int,
    image_change_threshold: float,
    action_diff_threshold: float,
    min_consistent_steps: int,
    cache: dict[str, str],
) -> int:
    """First time image+action changes for K consecutive steps -> cut at that start."""
    if not images or actions.size == 0:
        return 0
    prev_hash = ahash_digest(images[0], hash_size=hash_size, cache=cache)
    if prev_hash is None:
        return 0
    prev_action = actions[0]
    change_count = 0
    total_len = min(len(images), len(actions))
    for idx in range(frame_step, total_len, frame_step):
        img_hash = ahash_digest(images[idx], hash_size=hash_size, cache=cache)
        if img_hash is None:
            break
        ham = hamming_distance(prev_hash, img_hash)
        if ham is None:
            break
        image_changed = ham >= image_change_threshold
        action = actions[idx]
        diff = np.abs(action[ARM_IDXS] - prev_action[ARM_IDXS])
        action_changed = bool(np.any(diff > action_diff_threshold))
        if image_changed and action_changed:
            change_count += 1
            if change_count >= min_consistent_steps:
                start_idx = max(0, idx - (min_consistent_steps - 1) * frame_step)
                return min(start_idx, total_len - 1)
        else:
            change_count = 0
        prev_hash = img_hash
        prev_action = action
    return 0


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Industrial-grade episode prefix trimming")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--task", default="pick", choices=sorted(TASK_PRESETS.keys()))
    parser.add_argument("--hash-size", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=None)
    parser.add_argument("--image-change-threshold", type=float, default=None)
    parser.add_argument("--action-percentile", type=float, default=None)
    parser.add_argument("--min-consistent-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--report-out", default="trimmed_episodes.txt")
    args = parser.parse_args()

    preset = TASK_PRESETS[args.task]
    for name in (
        "hash_size",
        "frame_step",
        "image_change_threshold",
        "action_percentile",
        "min_consistent_steps",
    ):
        if getattr(args, name) is None:
            setattr(args, name, preset[name])

    print(
        f"[task={args.task}] hash_size={args.hash_size} frame_step={args.frame_step} "
        f"image_change_threshold={args.image_change_threshold} "
        f"action_percentile={args.action_percentile} min_consistent_steps={args.min_consistent_steps}"
    )

    root = Path(args.data_root)
    parquet_files = sorted(root.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode_*.parquet found in {root}")

    # ---------- Step 1: auto threshold ----------
    diffs = collect_action_diffs(root)
    diffs = np.asarray(diffs, dtype=np.float32)
    nonzero = diffs[diffs > 0]
    use_diffs = nonzero if nonzero.size > 0 else diffs
    threshold = float(np.percentile(use_diffs, args.action_percentile))
    print("\n=== Action diff statistics ===")
    print("Percentiles(all):", np.percentile(diffs, [5, 10, 25, 50, 75, 90, 95]))
    if nonzero.size > 0:
        print("Percentiles(nonzero):", np.percentile(nonzero, [5, 10, 25, 50, 75, 90, 95]))
    print(f"Using P{args.action_percentile} on {'nonzero' if nonzero.size > 0 else 'all'} diffs = {threshold:.6f}")

    # ---------- Output dir ----------
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"{root}_trimmed")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    trimmed = []
    total_removed = 0

    # ---------- Step 2: trim ----------
    for pq_file in tqdm(parquet_files, desc="Trim episodes"):
        full_table = pq.read_table(pq_file)
        images = full_table.column(CAM_KEY).to_pylist()
        actions = np.asarray(full_table.column(ACTION_KEY).to_pylist(), dtype=np.float32)

        start_idx = compute_trim_start(
            images,
            actions,
            hash_size=args.hash_size,
            frame_step=args.frame_step,
            image_change_threshold=args.image_change_threshold,
            action_diff_threshold=threshold,
            min_consistent_steps=args.min_consistent_steps,
            cache=cache,
        )

        if start_idx > 0:
            trimmed.append((pq_file.name, start_idx))
            total_removed += start_idx

        trimmed_table = full_table.slice(start_idx)
        pq.write_table(trimmed_table, out_dir / pq_file.name)

    # ---------- Report ----------
    report_path = Path(args.report_out)
    if not report_path.is_absolute():
        report_path = out_dir / report_path
    with open(report_path, "w") as f:
        for name, idx in trimmed:
            f.write(f"{name}\t{idx}\n")

    print("\n=== Episode Trim Summary ===")
    print(f"Data root: {root}")
    print(f"Output dir: {out_dir}")
    print(f"Total episodes: {len(parquet_files)}")
    print(f"Trimmed episodes: {len(trimmed)}")
    print(f"Total removed frames: {total_removed}")
    print(f"Action diff threshold (P{args.action_percentile}): {threshold:.6f}")
    print(f"Min consistent steps: {args.min_consistent_steps}")
    print("Mode: image_change AND action_change")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    main()
