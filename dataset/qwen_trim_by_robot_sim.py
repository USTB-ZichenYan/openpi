from __future__ import annotations

"""Trim episode prefixes using Qwen image similarity.

Rule (prefix trimming):
  For consecutive frames, if image_change < image_change_threshold
  for K steps, then trim the prefix.

Outputs:
  - Trimmed parquet files (same schema) into out-dir
  - A report file with episode name and trim index
"""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

DEFAULT_DATA_ROOT = Path(
    "/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/grab_stool_train/data/chunk-000"
)
CAM_KEY = "observation.images.cam_high"

QWEN_LOCAL_DIR = (
    "/iag_ad_vepfs_volc/iag_ad_vepfs_volc/yanzichen/model/"
    "qwen2-vl-7b/models--Qwen--Qwen2-VL-7B-Instruct/"
    "snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
)
ROI_RATIO = 1.0

PROMPT = (
    "You are given two images. Return a JSON object like {\"similarity\": 0} "
    "where similarity is an integer in [0,100], 100 means the images are very similar. "
    "Only output the JSON."
)
GEN_KWARGS = {"max_new_tokens": 16}


def load_image_pil(img_struct: dict) -> Image.Image | None:
    if img_struct is None:
        return None
    if img_struct.get("bytes"):
        return Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB")
    if img_struct.get("path"):
        return Image.open(img_struct["path"]).convert("RGB")
    return None


def load_image_bytes(img_struct: dict) -> bytes | None:
    if img_struct is None:
        return None
    if img_struct.get("bytes"):
        return img_struct["bytes"]
    if img_struct.get("path"):
        with open(img_struct["path"], "rb") as f:
            return f.read()
    return None


def crop_bottom_roi(img: Image.Image, ratio: float) -> Image.Image:
    w, h = img.size
    y0 = int(h * (1 - ratio))
    return img.crop((0, y0, w, h))


def load_qwen_local_model(model_dir: str):
    print(f"[INFO] Loading Qwen2-VL from local dir: {model_dir}")

    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        print("[WARN] CUDA not available, running on CPU (slow).")
    model.eval()
    _sanitize_generation_config(model)
    return processor, model


def _sanitize_generation_config(model: AutoModelForVision2Seq) -> None:
    cfg = model.generation_config
    for name in ("temperature", "top_p", "top_k"):
        if hasattr(cfg, name):
            try:
                setattr(cfg, name, None)
            except Exception:
                pass
    if hasattr(cfg, "do_sample"):
        cfg.do_sample = False


def qwen_image_similarity(
    img_a_struct: dict,
    img_b_struct: dict,
    *,
    processor,
    model,
    roi_ratio: float,
    cache: dict[str, float],
    debug: bool,
) -> float:
    img_a_bytes = load_image_bytes(img_a_struct)
    img_b_bytes = load_image_bytes(img_b_struct)
    cache_key = None
    if img_a_bytes and img_b_bytes:
        cache_key = hashlib.sha1(img_a_bytes + img_b_bytes).hexdigest()
        if cache_key in cache:
            return cache[cache_key]

    img_a = load_image_pil(img_a_struct)
    img_b = load_image_pil(img_b_struct)
    if img_a is None or img_b is None:
        return 0.0

    img_a = crop_bottom_roi(img_a, ratio=roi_ratio)
    img_b = crop_bottom_roi(img_b, ratio=roi_ratio)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_a},
                {"type": "image", "image": img_b},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    try:
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[[img_a, img_b]],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **GEN_KWARGS)

    out_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    if debug:
        print(f"[RAW] {out_text}")

    sim = 0.0
    # Extract the last JSON object to avoid picking the example in the prompt.
    matches = re.findall(r"\{[^{}]*\}", out_text)
    if matches:
        try:
            obj = json.loads(matches[-1])
            sim = float(obj.get("similarity", 0.0))
        except Exception:
            sim = 0.0
            if debug:
                print("[PARSE] invalid JSON")
    else:
        if debug:
            print("[PARSE] no JSON found")
    sim = max(0.0, min(100.0, sim))
    if debug:
        print(f"[PARSE] sim={sim}")

    if cache_key:
        cache[cache_key] = sim
    return sim


def compute_trim_start(
    images: list[dict],
    *,
    chunk_size: int,
    image_change_threshold: float,
    min_consistent_steps: int,
    max_trim: int,
    processor,
    model,
    roi_ratio: float,
    cache: dict[str, float],
    debug_steps: int,
) -> int:
    if not images:
        return 0

    start_idx = 0
    bad_count = 0
    step = max(1, chunk_size)

    for start in range(0, len(images), step):
        end = min(start + step - 1, len(images) - 1)
        sim = qwen_image_similarity(
            images[start],
            images[end],
            processor=processor,
            model=model,
            roi_ratio=roi_ratio,
            cache=cache,
            debug=debug_steps > 0,
        )
        if debug_steps > 0:
            debug_steps -= 1

        image_change = 100.0 - sim
        image_static = image_change <= image_change_threshold

        if image_static:
            bad_count += 1
            if bad_count >= min_consistent_steps:
                start_idx = min(end + 1, len(images) - 1)
        else:
            break

    if max_trim > 0:
        start_idx = min(start_idx, max_trim)
    return start_idx


def main() -> None:
    print(f"[INFO] Script: {__file__}")
    parser = argparse.ArgumentParser(description="Episode prefix trimming with Qwen similarity")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--frame-step", type=int, default=1, help="(Deprecated) use --chunk-size instead.")
    parser.add_argument("--chunk-size", type=int, default=0, help="Compare every N frames as a chunk.")
    parser.add_argument("--image-change-threshold", type=float, default=0.0)
    parser.add_argument("--min-consistent-steps", type=int, default=3)
    parser.add_argument("--max-trim", type=int, default=0, help="Cap the trim index to at most this value.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--qwen-local-dir", default=QWEN_LOCAL_DIR)
    parser.add_argument("--roi-ratio", type=float, default=ROI_RATIO)
    parser.add_argument("--debug-sim-steps", type=int, default=0)
    args = parser.parse_args()

    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if args.chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0")
    if args.min_consistent_steps < 1:
        raise ValueError("--min-consistent-steps must be >= 1")
    if args.max_trim < 0:
        raise ValueError("--max-trim must be >= 0")
    if not (0.0 < args.roi_ratio <= 1.0):
        raise ValueError("--roi-ratio must be in (0, 1]")

    root = Path(args.data_root)
    parquet_files = sorted(root.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode_*.parquet found in {root}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"{root}_trimmed_qwen_local")
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_qwen_local_model(args.qwen_local_dir)
    cache: dict[str, float] = {}

    trimmed = []
    total_removed = 0

    for pq_file in tqdm(parquet_files, desc="Trim episodes"):
        full_table = pq.read_table(pq_file)
        images = full_table.column(CAM_KEY).to_pylist()

        chunk_size = args.chunk_size if args.chunk_size > 0 else args.frame_step
        start_idx = compute_trim_start(
            images,
            chunk_size=chunk_size,
            image_change_threshold=args.image_change_threshold,
            min_consistent_steps=args.min_consistent_steps,
            max_trim=args.max_trim,
            processor=processor,
            model=model,
            roi_ratio=args.roi_ratio,
            cache=cache,
            debug_steps=args.debug_sim_steps,
        )

        if start_idx > 0:
            trimmed.append((pq_file.name, start_idx))
            total_removed += start_idx

        trimmed_table = full_table.slice(start_idx)
        pq.write_table(trimmed_table, out_dir / pq_file.name)

    report_path = Path(args.report_out) if args.report_out else Path("trimmed_episodes_qwen_local.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for name, idx in trimmed:
            f.write(f"{name}\t{idx}\n")

    print("\n=== Qwen Trim Summary ===")
    print(f"Data root: {root}")
    print(f"Output dir: {out_dir}")
    print(f"Total episodes: {len(parquet_files)}")
    print(f"Trimmed episodes: {len(trimmed)}")
    print(f"Total removed frames: {total_removed}")
    print(f"Min consistent steps: {args.min_consistent_steps}")
    print(f"Max trim: {args.max_trim}")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    main()
