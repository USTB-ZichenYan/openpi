"""Save frames around trim points for visual inspection."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _try_import_pil():
    try:
        from PIL import Image

        return Image
    except Exception:
        return None


def parse_report_line(line: str) -> tuple[str, int] | None:
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    name = parts[0]
    try:
        idx = int(parts[1])
    except ValueError:
        return None
    return name, idx


def load_image(img_struct: dict):
    image_mod = _try_import_pil()
    if image_mod is None:
        raise RuntimeError("PIL not available; install pillow to save images")
    if img_struct is None:
        return None
    img_bytes = img_struct.get("bytes")
    if img_bytes:
        return image_mod.open(io.BytesIO(img_bytes)).convert("RGB")
    img_path = img_struct.get("path")
    if img_path:
        return image_mod.open(img_path).convert("RGB")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Save frames around trim points.")
    parser.add_argument("--data-root", required=True, help="Directory with episode_*.parquet")
    parser.add_argument("--report", default="trimmed_episodes.txt")
    parser.add_argument("--out-dir", default="test_result")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in (l.strip() for l in f) if ln]

    for line in lines:
        parsed = parse_report_line(line)
        if parsed is None:
            continue
        name, cut_idx = parsed
        episode_path = data_root / name
        if not episode_path.exists():
            continue

        table = pq.read_table(episode_path, columns=["observation.images.cam_high"])
        images = table.column(0).to_pylist()

        idxs = [cut_idx - 2, cut_idx - 1, cut_idx, cut_idx + 1]
        idxs = [i for i in idxs if 0 <= i < len(images)]
        if not idxs:
            continue

        ep_out = out_dir / episode_path.stem
        ep_out.mkdir(parents=True, exist_ok=True)

        for i in idxs:
            img = load_image(images[i])
            if img is None:
                continue
            img.save(ep_out / f"frame_{i:06d}.png")

        np.save(ep_out / "cut_index.npy", np.asarray([cut_idx], dtype=np.int64))


if __name__ == "__main__":
    main()
