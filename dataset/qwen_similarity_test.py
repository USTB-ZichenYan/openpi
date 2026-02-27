"""Quick test: compute Qwen similarity for any two images in a folder."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re


def qwen_image_similarity(
    img_a: bytes,
    img_b: bytes,
    *,
    model: str,
    api_key: str,
    url: str,
    timeout_s: int,
) -> float:
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
    sim = max(0.0, min(100.0, sim))
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Qwen similarity on two images in a folder.")
    parser.add_argument("--dir", default="/home/SENSETIME/yanzichen/data/file/openpi/dataset/test_result/episode_000000")
    parser.add_argument("--img-a", default="")
    parser.add_argument("--img-b", default="")
    parser.add_argument("--qwen-model", default="qwen-vl-max")
    parser.add_argument(
        "--qwen-url",
        default="https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    )
    parser.add_argument("--qwen-timeout-s", type=int, default=30)
    parser.add_argument("--qwen-api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.qwen_api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key in env: {args.qwen_api_key_env}")

    img_dir = Path(args.dir)
    if args.img_a and args.img_b:
        img_a_path = Path(args.img_a)
        img_b_path = Path(args.img_b)
    else:
        imgs = sorted(p for p in img_dir.glob("*.png"))
        if len(imgs) < 2:
            raise FileNotFoundError("Need at least two .png images in the directory")
        img_a_path, img_b_path = imgs[0], imgs[1]

    img_a = img_a_path.read_bytes()
    img_b = img_b_path.read_bytes()

    sim = qwen_image_similarity(
        img_a,
        img_b,
        model=args.qwen_model,
        api_key=api_key,
        url=args.qwen_url,
        timeout_s=args.qwen_timeout_s,
    )
    print(f"Image A: {img_a_path}")
    print(f"Image B: {img_b_path}")
    print(f"Similarity (0-100): {sim:.2f}")


if __name__ == "__main__":
    main()
