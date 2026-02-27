#!/usr/bin/env python3
"""
Quick visualization for FT cleaning quality check.

For each cleaned episode parquet, generate one figure containing:
- contact_flag over time
- stage_id / stage_label over time
- tail_mean_fz / tail_var_fz / success from ft_clean_report.json (in title)

Also generate a summary figure across episodes:
- tail_mean_fz bar
- tail_var_fz bar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


STAGE_ID_TO_NAME = {
    0: "free-space",
    1: "contact",
    2: "lift",
    3: "carry",
}
STAGE_NAME_TO_ID = {v: k for k, v in STAGE_ID_TO_NAME.items()}


def resolve_chunk_dir(path: Path) -> Path:
    if any(path.glob("episode_*.parquet")):
        return path
    c1 = path / "data" / "chunk-000"
    if any(c1.glob("episode_*.parquet")):
        return c1
    c2 = path / "chunk-000"
    if any(c2.glob("episode_*.parquet")):
        return c2
    raise FileNotFoundError(f"No episode_*.parquet found under: {path}")


def resolve_report_path(chunk_dir: Path, report_path: str | None) -> Path | None:
    if report_path:
        p = Path(report_path)
        return p if p.exists() else None

    # Typical layout:
    # <root>/data/chunk-000  -> <root>/ft_clean_report.json
    p1 = chunk_dir.parent.parent / "ft_clean_report.json"
    if p1.exists():
        return p1

    p2 = chunk_dir / "ft_clean_report.json"
    if p2.exists():
        return p2
    return None


def load_report_map(report_file: Path | None) -> dict[str, dict]:
    if report_file is None:
        return {}
    data = json.loads(report_file.read_text())
    episode_rows = data.get("episodes", [])
    out = {}
    for row in episode_rows:
        output_name = str(row.get("output", ""))
        if output_name:
            out[output_name] = row
    return out


def to_int_stage(stage_id: np.ndarray | None, stage_label: list[str] | None, n: int) -> np.ndarray:
    if stage_id is not None:
        return np.asarray(stage_id, dtype=np.int8)
    if stage_label is not None:
        return np.array([STAGE_NAME_TO_ID.get(str(x), -1) for x in stage_label], dtype=np.int8)
    return np.full((n,), -1, dtype=np.int8)


def plot_one_episode(
    parquet_path: Path,
    out_png: Path,
    report_row: dict | None,
) -> None:
    table = pq.read_table(str(parquet_path))
    colnames = set(table.column_names)

    if "contact_flag" not in colnames:
        raise KeyError(f"{parquet_path.name} missing required column: contact_flag")

    contact = np.asarray(table.column("contact_flag").to_pylist(), dtype=np.int8)
    t = np.arange(contact.shape[0], dtype=np.int32)

    stage_id_arr: np.ndarray | None = None
    stage_label_arr: list[str] | None = None
    if "stage_id" in colnames:
        stage_id_arr = np.asarray(table.column("stage_id").to_pylist(), dtype=np.int8)
    if "stage_label" in colnames:
        stage_label_arr = [str(x) for x in table.column("stage_label").to_pylist()]
    stage = to_int_stage(stage_id_arr, stage_label_arr, n=contact.shape[0])

    tail_mean = float("nan")
    tail_var = float("nan")
    success = None
    if report_row:
        tail_mean = float(report_row.get("tail_mean_fz", float("nan")))
        tail_var = float(report_row.get("tail_var_fz", float("nan")))
        success = report_row.get("success", None)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].step(t, contact, where="post", linewidth=1.5)
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["FREE", "CONTACT"])
    axes[0].set_ylabel("contact_flag")
    axes[0].grid(alpha=0.25)

    axes[1].step(t, stage, where="post", linewidth=1.5, color="#d95f02")
    axes[1].set_ylabel("stage")
    axes[1].set_xlabel("frame")
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(
        [
            STAGE_ID_TO_NAME[0],
            STAGE_ID_TO_NAME[1],
            STAGE_ID_TO_NAME[2],
            STAGE_ID_TO_NAME[3],
        ]
    )
    axes[1].grid(alpha=0.25)

    stat_text = f"tail_mean_fz={tail_mean:.3f}, tail_var_fz={tail_var:.3f}"
    if success is not None:
        stat_text += f", success={bool(success)}"
    fig.suptitle(f"{parquet_path.name}\n{stat_text}", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_summary(report_map: dict[str, dict], out_png: Path) -> None:
    if not report_map:
        return

    names = sorted(report_map.keys())
    means = [float(report_map[n].get("tail_mean_fz", 0.0)) for n in names]
    vars_ = [float(report_map[n].get("tail_var_fz", 0.0)) for n in names]
    success = [bool(report_map[n].get("success", False)) for n in names]

    x = np.arange(len(names), dtype=np.int32)
    colors = ["#2ca02c" if ok else "#d62728" for ok in success]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    axes[0].bar(x, means, color=colors, alpha=0.9)
    axes[0].set_ylabel("tail_mean_fz")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_title("Tail Fz Mean (green=success, red=failure)")

    axes[1].bar(x, vars_, color=colors, alpha=0.9)
    axes[1].set_ylabel("tail_var_fz")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_title("Tail Fz Variance")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FT clean QC figures per episode")
    parser.add_argument(
        "--clean-dir",
        default="/home/SENSETIME/yanzichen/data/file/dataset/0206_ft_train_clean/data/chunk-000",
        help="Clean dataset root or chunk dir",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to ft_clean_report.json (optional; auto-detect if omitted)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for PNGs (default: <clean_root>/qc_plots)",
    )
    args = parser.parse_args()

    chunk_dir = resolve_chunk_dir(Path(args.clean_dir))
    report_file = resolve_report_path(chunk_dir, args.report)
    report_map = load_report_map(report_file)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # <clean_root>/data/chunk-000 -> <clean_root>/qc_plots
        out_dir = chunk_dir.parent.parent / "qc_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(chunk_dir.glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet found in: {chunk_dir}")

    print("=" * 80)
    print("FT QC plot")
    print(f"clean chunk: {chunk_dir}")
    print(f"report    : {report_file if report_file else 'N/A'}")
    print(f"out dir   : {out_dir}")
    print(f"episodes  : {len(files)}")
    print("=" * 80)

    for pq_file in tqdm(files, desc="Plot episodes", unit="episode"):
        out_png = out_dir / f"{pq_file.stem}_qc.png"
        row = report_map.get(pq_file.name, None)
        plot_one_episode(pq_file, out_png=out_png, report_row=row)

    summary_png = out_dir / "tail_stats_summary.png"
    plot_summary(report_map, summary_png)

    print("\nDone.")
    print(f"Per-episode plots: {out_dir}")
    if report_map:
        print(f"Summary plot    : {summary_png}")


if __name__ == "__main__":
    main()

