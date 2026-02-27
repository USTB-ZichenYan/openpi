#!/usr/bin/env python3
"""
Stage statistics evaluation for cleaned FT datasets.

Outputs:
- stage_stats_summary.json
- stage_stats_episodes.csv
- stage_ratio_overall.png
- stage_ratio_per_episode.png
"""

from __future__ import annotations

import argparse
import csv
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
STAGE_ORDER = [0, 1, 2, 3]
STAGE_COLORS = {
    0: "#4C78A8",
    1: "#F58518",
    2: "#54A24B",
    3: "#E45756",
}


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


def parse_stage_weights(raw: str | None) -> dict[int, float]:
    if not raw:
        return {0: 1.0, 1: 2.0, 2: 3.0, 3: 3.0}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--stage-weights must be a JSON dict, e.g. '{\"0\":1,\"1\":2}'")
    return {int(k): float(v) for k, v in parsed.items()}


def get_stage_ids(table: pq.Table) -> np.ndarray:
    cols = set(table.column_names)
    if "stage_id" in cols:
        return np.asarray(table.column("stage_id").to_pylist(), dtype=np.int8)
    if "stage_label" in cols:
        labels = [str(x) for x in table.column("stage_label").to_pylist()]
        name_to_id = {v: k for k, v in STAGE_ID_TO_NAME.items()}
        return np.array([name_to_id.get(x, -1) for x in labels], dtype=np.int8)
    raise KeyError("Missing stage_id/stage_label in parquet")


def get_contact_flag(table: pq.Table) -> np.ndarray | None:
    if "contact_flag" in table.column_names:
        return np.asarray(table.column("contact_flag").to_pylist(), dtype=np.int8)
    return None


def count_segments(binary_flag: np.ndarray) -> int:
    if binary_flag.size == 0:
        return 0
    # segment starts where 0->1
    prev = np.concatenate([np.array([0], dtype=np.int8), binary_flag[:-1]])
    starts = np.logical_and(prev == 0, binary_flag == 1)
    return int(np.sum(starts))


def plot_overall_ratio(stage_ratio: dict[int, float], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(STAGE_ORDER), dtype=np.int32)
    vals = [stage_ratio.get(s, 0.0) for s in STAGE_ORDER]
    colors = [STAGE_COLORS[s] for s in STAGE_ORDER]
    labels = [STAGE_ID_TO_NAME[s] for s in STAGE_ORDER]

    ax.bar(xs, vals, color=colors, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, max(1e-6, max(vals) * 1.2))
    ax.set_ylabel("ratio")
    ax.set_title("Overall Stage Ratio")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_episode_ratio(rows: list[dict], out_png: Path) -> None:
    names = [r["episode"] for r in rows]
    x = np.arange(len(names), dtype=np.int32)
    bottom = np.zeros((len(names),), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.4), 6))
    for s in STAGE_ORDER:
        vals = np.array([r[f"ratio_stage_{s}"] for r in rows], dtype=np.float64)
        ax.bar(x, vals, bottom=bottom, color=STAGE_COLORS[s], label=STAGE_ID_TO_NAME[s], alpha=0.9)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ratio")
    ax.set_title("Per-Episode Stage Ratio (Stacked)")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stage statistics on cleaned parquet episodes")
    parser.add_argument(
        "--clean-dir",
        default="/home/SENSETIME/yanzichen/data/file/dataset/0206_ft_train_clean/data/chunk-000",
        help="Clean dataset root or chunk dir",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <clean_root>/stage_eval)",
    )
    parser.add_argument(
        "--stage-weights",
        default='{"0":1.0,"1":2.0,"2":3.0,"3":3.0}',
        help="Stage weights JSON for weighted sampling ratio estimation",
    )
    args = parser.parse_args()

    chunk_dir = resolve_chunk_dir(Path(args.clean_dir))
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = chunk_dir.parent.parent / "stage_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(chunk_dir.glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet in {chunk_dir}")

    stage_weights = parse_stage_weights(args.stage_weights)

    print("=" * 80)
    print("Stage statistics evaluation")
    print(f"clean chunk : {chunk_dir}")
    print(f"out dir     : {out_dir}")
    print(f"episodes    : {len(files)}")
    print(f"stageWeight : {stage_weights}")
    print("=" * 80)

    per_episode: list[dict] = []
    total_stage = {s: 0 for s in STAGE_ORDER}
    total_frames = 0
    total_contact_frames = 0
    total_contact_segments = 0

    for p in tqdm(files, desc="Eval stage", unit="episode"):
        table = pq.read_table(str(p))
        stage = get_stage_ids(table)
        n = int(stage.shape[0])
        total_frames += n

        row = {
            "episode": p.name,
            "frames": n,
        }

        for s in STAGE_ORDER:
            c = int(np.sum(stage == s))
            total_stage[s] += c
            row[f"count_stage_{s}"] = c
            row[f"ratio_stage_{s}"] = float(c / n) if n > 0 else 0.0

        contact = get_contact_flag(table)
        if contact is not None:
            contact_frames = int(np.sum(contact == 1))
            contact_segments = count_segments(contact.astype(np.int8))
            total_contact_frames += contact_frames
            total_contact_segments += contact_segments
            row["contact_frames"] = contact_frames
            row["contact_ratio"] = float(contact_frames / n) if n > 0 else 0.0
            row["contact_segments"] = contact_segments
        else:
            row["contact_frames"] = -1
            row["contact_ratio"] = -1.0
            row["contact_segments"] = -1

        per_episode.append(row)

    stage_ratio = {s: (total_stage[s] / total_frames if total_frames > 0 else 0.0) for s in STAGE_ORDER}

    # Estimate weighted sampling ratios
    weighted_mass = {}
    for s in STAGE_ORDER:
        weighted_mass[s] = float(total_stage[s]) * float(stage_weights.get(s, 1.0))
    total_mass = sum(weighted_mass.values()) + 1e-12
    weighted_ratio = {s: weighted_mass[s] / total_mass for s in STAGE_ORDER}

    summary = {
        "clean_chunk_dir": str(chunk_dir),
        "num_episodes": len(files),
        "total_frames": total_frames,
        "total_stage_count": {str(s): int(total_stage[s]) for s in STAGE_ORDER},
        "overall_stage_ratio": {str(s): float(stage_ratio[s]) for s in STAGE_ORDER},
        "contact_frames_total": int(total_contact_frames),
        "contact_ratio_total": float(total_contact_frames / total_frames) if total_frames > 0 else 0.0,
        "contact_segments_total": int(total_contact_segments),
        "stage_weights": {str(k): float(v) for k, v in stage_weights.items()},
        "estimated_weighted_stage_ratio": {str(s): float(weighted_ratio[s]) for s in STAGE_ORDER},
    }

    # Save JSON
    summary_json = out_dir / "stage_stats_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    # Save CSV
    csv_file = out_dir / "stage_stats_episodes.csv"
    fieldnames = [
        "episode",
        "frames",
        "count_stage_0",
        "count_stage_1",
        "count_stage_2",
        "count_stage_3",
        "ratio_stage_0",
        "ratio_stage_1",
        "ratio_stage_2",
        "ratio_stage_3",
        "contact_frames",
        "contact_ratio",
        "contact_segments",
    ]
    with csv_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_episode:
            writer.writerow(row)

    # Save plots
    plot_overall_ratio(stage_ratio, out_dir / "stage_ratio_overall.png")
    plot_episode_ratio(per_episode, out_dir / "stage_ratio_per_episode.png")

    print("\nDone.")
    print(f"summary json : {summary_json}")
    print(f"episode csv  : {csv_file}")
    print(f"plots        : {out_dir / 'stage_ratio_overall.png'}")
    print(f"              {out_dir / 'stage_ratio_per_episode.png'}")


if __name__ == "__main__":
    main()

