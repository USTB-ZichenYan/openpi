"""Trim episode prefixes using Qwen or aHash similarity and action diffs.

Rule (prefix trimming):
  For consecutive frames, if image_change < image_change_threshold
  AND action_diff > action_diff_threshold for K steps, then trim the prefix.

Outputs:
  - Trimmed parquet files (same schema) into out-dir
  - A report file with episode name and trim index
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

DEFAULT_DATA_ROOT = Path("/home/SENSETIME/yanzichen/data/file/dataset/grab_stool_train/data/chunk-000")
CAM_KEY = "observation.images.cam_high"
ACTION_KEY = "action"


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


def load_image_bytes(img_struct: dict) -> bytes | None:
    if img_struct is None:
        return None
    img_bytes = img_struct.get("bytes")
    if img_bytes:
        return img_bytes
    img_path = img_struct.get("path")
    if img_path:
        with open(img_path, "rb") as f:
            return f.read()
    return None


def qwen_image_similarity(
    img_a: bytes,
    img_b: bytes,
    *,
    model: str,
    api_key: str,
    url: str,
    timeout_s: int,
) -> float:
    """Return similarity in [0, 1], where 1 means very similar."""
    try:
        import requests
    except Exception as exc:
        raise RuntimeError("requests is required for Qwen API calls") from exc

    def _img_payload(img_bytes: bytes) -> dict:
        return {"image": f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"}

    prompt = (
        "You are given two images. Return a JSON object like {\"similarity\": 0} "
        "where similarity is an integer in [0,100], 100 means the images are very similar. "
        "Only output the JSON."
    )
    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                        _img_payload(img_a),
                        _img_payload(img_b),
                    ],
                }
            ]
        },
        "parameters": {"temperature": 0.0},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    text = ""
    output = data.get("output", {})
    if isinstance(output, dict):
        text = output.get("text", "") or ""
        if not text and "choices" in output:
            choices = output.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text = item["text"]
                            break
                if not text:
                    text = choices[0].get("text", "") or ""

    sim = 0.0
    if text:
        try:
            obj = json.loads(text)
            sim = float(obj.get("similarity", 0.0))
        except Exception:
            match = re.search(r"similarity\"?\s*[:=]\s*([0-9.]+)", text)
            if match:
                sim = float(match.group(1))

    return max(0.0, min(100.0, sim))


def collect_action_diffs(root: Path):
    diffs = []
    parquet_files = sorted(root.glob("episode_*.parquet"))
    print(f"Scanning {len(parquet_files)} episodes for action diff stats...")

    for pq_file in tqdm(parquet_files):
        table = pq.read_table(pq_file, columns=[ACTION_KEY])
        actions = np.asarray(table.column(0).to_pylist(), dtype=np.float32)
        for t in range(1, len(actions)):
            diff = np.linalg.norm(actions[t] - actions[t - 1])
            diffs.append(diff)

    diffs = np.array(diffs)
    return diffs


def compute_trim_start(
    images: list[dict],
    actions: np.ndarray,
    *,
    frame_step: int,
    image_change_threshold: float,
    action_diff_threshold: float,
    min_consistent_steps: int,
    compare_mode: str,
    hash_size: int,
    qwen_model: str,
    qwen_api_key: str,
    qwen_url: str,
    qwen_timeout_s: int,
    cache: dict[str, float],
    ahash_cache: dict[str, str],
    debug_counter: dict[str, int] | None,
) -> int:
    if not images or actions.size == 0:
        return 0

    prev_img = load_image_bytes(images[0])
    prev_hash = ahash_digest(images[0], hash_size=hash_size, cache=ahash_cache) if compare_mode == "ahash" else None
    prev_action = actions[0]
    start_idx = 0
    bad_count = 0

    for idx in range(frame_step, len(images), frame_step):
        image_change = None
        img_bytes = None
        if compare_mode == "qwen":
            img_bytes = load_image_bytes(images[idx])
            if prev_img is None or img_bytes is None:
                break
            cache_key = hashlib.sha1(prev_img + img_bytes).hexdigest()
            if cache_key in cache:
                similarity = cache[cache_key]
            else:
                similarity = qwen_image_similarity(
                    prev_img,
                    img_bytes,
                    model=qwen_model,
                    api_key=qwen_api_key,
                    url=qwen_url,
                    timeout_s=qwen_timeout_s,
                )
                cache[cache_key] = similarity
            image_change = 100.0 - similarity
        elif compare_mode == "ahash":
            img_hash = ahash_digest(images[idx], hash_size=hash_size, cache=ahash_cache)
            if prev_hash is None or img_hash is None:
                break
            ham = hamming_distance(prev_hash, img_hash)
            if ham is None:
                break
            image_change = float(ham)
            prev_hash = img_hash
        else:
            raise ValueError(f"Unknown compare_mode: {compare_mode}")
        action = actions[idx]
        diff = float(np.linalg.norm(action - prev_action))

        if image_change is not None:
            if compare_mode == "qwen":
                image_static = image_change <= image_change_threshold
            else:
                image_static = image_change < image_change_threshold
        else:
            image_static = False

        if debug_counter is not None and debug_counter.get("remaining", 0) > 0:
            print(
                f"[debug] step={idx} sim_change={image_change:.2f} "
                f"diff={diff:.4f} static={image_static}"
            )
            debug_counter["remaining"] -= 1

        if image_static and diff > action_diff_threshold:
            bad_count += 1
            if bad_count >= min_consistent_steps:
                start_idx = min(idx + 1, len(images) - 1)
        else:
            break

        if compare_mode == "qwen":
            prev_img = img_bytes
        prev_action = action

    return start_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Episode prefix trimming with Qwen/aHash similarity")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--frame-step", type=int, default=1)
    # For Qwen similarity in [0,100], 0.0 means only 100% similar is treated as "static".
    parser.add_argument("--image-change-threshold", type=float, default=0.0)
    parser.add_argument("--action-percentile", type=float, default=75.0)
    parser.add_argument("--min-consistent-steps", type=int, default=3)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--compare-mode", choices=["qwen", "ahash"], default="qwen")
    parser.add_argument("--hash-size", type=int, default=8, help="Used for aHash only")
    parser.add_argument("--qwen-model", default="qwen-vl-max")
    parser.add_argument("--qwen-url", default="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
    parser.add_argument("--qwen-timeout-s", type=int, default=30)
    parser.add_argument("--qwen-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--debug-sim-steps", type=int, default=0, help="Print first N similarity steps")
    args = parser.parse_args()

    api_key = ""
    if args.compare_mode == "qwen":
        api_key = os.environ.get(args.qwen_api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key in env: {args.qwen_api_key_env}")

    root = Path(args.data_root)
    parquet_files = sorted(root.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode_*.parquet found in {root}")

    diffs = collect_action_diffs(root)
    threshold = float(np.percentile(diffs, args.action_percentile))
    print("\n=== Action diff statistics ===")
    print("Percentiles:", np.percentile(diffs, [5, 10, 25, 50, 75, 90, 95]))
    print(f"Using P{args.action_percentile} as action_diff_threshold = {threshold:.4f}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"{root}_trimmed_{args.compare_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, float] = {}
    ahash_cache: dict[str, str] = {}
    trimmed = []
    total_removed = 0
    debug_counter = {"remaining": args.debug_sim_steps} if args.debug_sim_steps > 0 else None

    for pq_file in tqdm(parquet_files, desc="Trim episodes"):
        full_table = pq.read_table(pq_file)
        images = full_table.column(CAM_KEY).to_pylist()
        actions = np.asarray(full_table.column(ACTION_KEY).to_pylist(), dtype=np.float32)

        start_idx = compute_trim_start(
            images,
            actions,
            frame_step=args.frame_step,
            image_change_threshold=args.image_change_threshold,
            action_diff_threshold=threshold,
            min_consistent_steps=args.min_consistent_steps,
            compare_mode=args.compare_mode,
            hash_size=args.hash_size,
            qwen_model=args.qwen_model,
            qwen_api_key=api_key,
            qwen_url=args.qwen_url,
            qwen_timeout_s=args.qwen_timeout_s,
            cache=cache,
            ahash_cache=ahash_cache,
            debug_counter=debug_counter,
        )

        if start_idx > 0:
            trimmed.append((pq_file.name, start_idx))
            total_removed += start_idx

        trimmed_table = full_table.slice(start_idx)
        pq.write_table(trimmed_table, out_dir / pq_file.name)

    report_path = Path(args.report_out) if args.report_out else Path(f"trimmed_episodes_{args.compare_mode}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for name, idx in trimmed:
            f.write(f"{name}\t{idx}\n")

    print("\n=== Qwen Trim Summary ===")
    print(f"Data root: {root}")
    print(f"Output dir: {out_dir}")
    print(f"Total episodes: {len(parquet_files)}")
    print(f"Trimmed episodes: {len(trimmed)}")
    print(f"Total removed frames: {total_removed}")
    print(f"Action diff threshold (P{args.action_percentile}): {threshold:.4f}")
    print(f"Min consistent steps: {args.min_consistent_steps}")
    print(f"Compare mode: {args.compare_mode}")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    main()
