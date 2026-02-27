#!/usr/bin/env python3
"""
通用数据转换脚本（带腕部相机 + 双臂力矩版本）- 整合不同日期/任务的数据为 LeRobot 格式（32维状态），并自动划分 train/test 两个数据集输出。

新特性：
- 支持三个相机：head_camera, left_hand_camera, right_hand_camera
- 按照 LeRobot 标准命名：cam_high, cam_left_wrist, cam_right_wrist
- 支持双臂六轴力矩：left_arm_6dof / right_arm_6dof（force+torque）

特点：
- 支持多种目录结构（兼容 20251224 特殊结构 + 其它日期通用结构）
- 支持两种"选择数据源"的方式（二选一）：
  1) --sources：显式指定 日期-关键词（与原脚本一致）
  2) --sources-except：指定 日期-排除关键词；该日期下 **所有不包含该关键词的 *_parser 文件夹** 都视为同一任务并合并
- 自动按 episode 级别随机划分：默认 90% train / 10% test
- 一次命令生成两个 LeRobot 数据集目录：<output_name>_train 和 <output_name>_test

使用示例：
  # 常规：只取 20260123 下 stool_grap 相关数据，生成 grab_stool_train / grab_stool_test
  python convert_generic_data_32d_train_test_wrist.py grab_stool \
      --task "pick up the box" \
      --sources 20260123-stool_grap \
      --min-frames 50

  # 特殊：取 20260123 下"除了包含 stool_grap 的文件夹以外"的所有 *_parser 文件夹，合并为一个任务
  # 生成 grab_detail_train / grab_detail_test
  python convert_generic_data_32d_train_test_wrist.py grab_detail_wrist \
      --task "pick up the box" \
      --sources-except 20260123-stool_grap \
      --min-frames 50
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# 数据根目录
DATA_ROOT = Path("/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data")


# =====================
# 数据源查找函数
# =====================

def find_status_files_20251224(base_dir: Path, task_keyword: str) -> List[Path]:
    """
    20251224 特殊目录结构
    格式: case*_parser/{task_keyword}/PickAndPLace/episode_*/status/status.json
    """
    status_files: list[Path] = []

    for case_dir in sorted(base_dir.glob("case*_parser")):
        if not case_dir.is_dir():
            continue

        task_dir = case_dir / task_keyword
        if not task_dir.exists():
            continue

        for episode_dir in task_dir.glob("PickAndPLace/episode_*"):
            status_file = episode_dir / "status" / "status.json"
            if status_file.exists():
                status_files.append(status_file)

    return status_files


def find_status_files_generic(base_dir: Path, keyword: str) -> List[Path]:
    """
    通用目录结构，自动尝试多种模式

    尝试顺序:
    1. keyword_*_parser (keyword 在开头，如: grab_*_parser)
    2. *_keyword_*_parser (keyword 在中间，如: *_parking_incoherent_grap_*_parser)
    """
    status_files: list[Path] = []

    # 模式1: keyword 在开头
    pattern1 = f"{keyword}_*_parser"
    for folder in sorted(base_dir.glob(pattern1)):
        if not folder.is_dir():
            continue
        for episode_dir in folder.glob("PickAndPLace/episode_*"):
            status_file = episode_dir / "status" / "status.json"
            if status_file.exists():
                status_files.append(status_file)

    # 如果找到了，直接返回
    if status_files:
        return status_files

    # 模式2: keyword 在中间
    pattern2 = f"*_{keyword}_*_parser"
    for folder in sorted(base_dir.glob(pattern2)):
        if not folder.is_dir():
            continue
        for episode_dir in folder.glob("PickAndPLace/episode_*"):
            status_file = episode_dir / "status" / "status.json"
            if status_file.exists():
                status_files.append(status_file)

    return status_files


def find_status_files_generic_except(base_dir: Path, exclude_keyword: str) -> List[Path]:
    """
    通用目录结构（适用于 20260123 这类日期文件夹）：
    - 扫描 base_dir 下所有 *_parser 文件夹
    - 排除 folder.name 中包含 exclude_keyword 的文件夹
    - 其余全部合并为一个任务

    例如：exclude_keyword="stool_grap"
      - 排除：*stool_grap*_parser
      - 保留：其它 *_parser（无论命名是否有固定关键词）
    """
    status_files: list[Path] = []

    for folder in sorted(base_dir.glob("*_parser")):
        if not folder.is_dir():
            continue
        if exclude_keyword and (exclude_keyword in folder.name):
            continue
        for episode_dir in folder.glob("PickAndPLace/episode_*"):
            status_file = episode_dir / "status" / "status.json"
            if status_file.exists():
                status_files.append(status_file)

    return status_files


def count_frames(status_file: Path) -> int:
    """统计单个 episode 的有效帧数（只检查 head_camera）"""
    try:
        with open(status_file, "r") as f:
            status_data = json.load(f)

        episode_dir = status_file.parent.parent
        image_dir = episode_dir / "camera" / "head_camera"

        if not image_dir.exists():
            return 0

        frame_count = 0
        for frame in status_data:
            img_path = image_dir / frame.get("color_image", "")
            if img_path.exists():
                frame_count += 1

        return frame_count
    except Exception:
        return 0


def parse_data_source(source_spec: str) -> Tuple[Path, str, bool]:
    """
    解析数据源规格字符串

    格式: <日期>-<关键词>

    返回: (base_dir, keyword, is_20251224)
    """
    parts = source_spec.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"无效的数据源格式: {source_spec}，应为 '日期-关键词'")

    date_str, keyword = parts
    base_dir = DATA_ROOT / date_str

    if not base_dir.exists():
        raise ValueError(f"数据目录不存在: {base_dir}")

    is_20251224 = (date_str == "20251224")
    return base_dir, keyword, is_20251224


# =====================
# 状态/速度提取（与 convert_turn_3_data_32d.py 一致）
# =====================

def extract_state_32d(frame: Dict) -> np.ndarray:
    """提取 32 维状态向量"""
    state_dict = frame["state"]

    head_joints = state_dict["head_joint"]
    head_state = np.array(
        [
            next(j["pos"] for j in head_joints if j["id"] == 2),
            next(j["pos"] for j in head_joints if j["id"] == 3),
        ],
        dtype=np.float32,
    )

    left_arm = np.array([joint["pos"] for joint in state_dict["left_arm_joint"]], dtype=np.float32)
    left_hand = np.array([finger["pos"] for finger in state_dict["left_hand"]], dtype=np.float32)
    right_arm = np.array([joint["pos"] for joint in state_dict["right_arm_joint"]], dtype=np.float32)
    right_hand = np.array([finger["pos"] for finger in state_dict["right_hand"]], dtype=np.float32)

    torso_joints = state_dict["torso_joint"]
    waist_state = np.array(
        [
            next(j["pos"] for j in torso_joints if j["id"] == 31),
            next(j["pos"] for j in torso_joints if j["id"] == 32),
        ],
        dtype=np.float32,
    )
    leg_state = np.array(
        [
            next(j["pos"] for j in torso_joints if j["id"] == 51),
            next(j["pos"] for j in torso_joints if j["id"] == 52),
        ],
        dtype=np.float32,
    )

    full_state = np.concatenate([head_state, left_arm, left_hand, right_arm, right_hand, waist_state, leg_state])
    assert full_state.shape == (32,), f"状态维度错误: {full_state.shape}"
    return full_state


def extract_velocity_32d(frame: Dict) -> np.ndarray:
    """提取 32 维速度向量"""
    state_dict = frame["state"]

    head_joints = state_dict["head_joint"]
    head_vel = np.array(
        [
            next(j["speed"] for j in head_joints if j["id"] == 2),
            next(j["speed"] for j in head_joints if j["id"] == 3),
        ],
        dtype=np.float32,
    )

    left_arm_vel = np.array([joint["speed"] for joint in state_dict["left_arm_joint"]], dtype=np.float32)
    left_hand_vel = np.array([finger["speed"] for finger in state_dict["left_hand"]], dtype=np.float32)
    right_arm_vel = np.array([joint["speed"] for joint in state_dict["right_arm_joint"]], dtype=np.float32)
    right_hand_vel = np.array([finger["speed"] for finger in state_dict["right_hand"]], dtype=np.float32)

    torso_joints = state_dict["torso_joint"]
    waist_vel = np.array(
        [
            next(j["speed"] for j in torso_joints if j["id"] == 31),
            next(j["speed"] for j in torso_joints if j["id"] == 32),
        ],
        dtype=np.float32,
    )
    leg_vel = np.array(
        [
            next(j["speed"] for j in torso_joints if j["id"] == 51),
            next(j["speed"] for j in torso_joints if j["id"] == 52),
        ],
        dtype=np.float32,
    )

    velocity = np.concatenate([head_vel, left_arm_vel, left_hand_vel, right_arm_vel, right_hand_vel, waist_vel, leg_vel])
    assert velocity.shape == (32,), f"速度维度错误: {velocity.shape}"
    return velocity


def _vec3_from_any(value: object) -> np.ndarray:
    """将任意输入解析为 3 维向量，失败时返回全 0。"""
    if isinstance(value, dict):
        return np.array(
            [
                float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)),
            ],
            dtype=np.float32,
        )

    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        try:
            return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
        except Exception:
            return np.zeros((3,), dtype=np.float32)

    return np.zeros((3,), dtype=np.float32)


def extract_arm_ft_6d(frame: Dict, arm_key: str) -> np.ndarray:
    """提取单臂 6 维 FT 向量: [fx, fy, fz, tx, ty, tz]。"""
    state_dict = frame.get("state", {})
    arm_ft = state_dict.get(arm_key, {})
    if not isinstance(arm_ft, dict):
        return np.zeros((6,), dtype=np.float32)

    force = _vec3_from_any(arm_ft.get("force"))
    torque = _vec3_from_any(arm_ft.get("torque"))
    ft = np.concatenate([force, torque]).astype(np.float32)
    assert ft.shape == (6,), f"{arm_key} FT维度错误: {ft.shape}"
    return ft


def load_image(image_path: Path, image_height: int, image_width: int) -> np.ndarray:
    """加载图像并统一到固定分辨率（默认 720x1280）。

    说明：LeRobot 的 feature shape 会做强校验。我们这里统一 resize，
    避免同一个数据集里混入不同分辨率导致 add_frame 报错。
    """
    img = Image.open(image_path).convert("RGB")
    if int(image_height) > 0 and int(image_width) > 0:
        # PIL 的 size 是 (width, height)
        if img.size != (int(image_width), int(image_height)):
            img = img.resize((int(image_width), int(image_height)), resample=Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def load_status_data(status_file: Path) -> List[Dict]:
    with open(status_file, "r") as f:
        return json.load(f)


# =====================
# LeRobot 数据集创建 + meta 修补
# =====================

def create_lerobot_dataset(dataset_path: Path, fps: int = 30, image_height: int = 720, image_width: int = 1280) -> LeRobotDataset:
    motors = [
        "head_pitch", "head_yaw",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow_pitch", "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll",
        "left_little_finger", "left_ring_finger", "left_middle_finger",
        "left_fore_finger", "left_thumb_bend", "left_thumb_rotation",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow_pitch", "right_wrist_yaw", "right_wrist_pitch", "right_wrist_roll",
        "right_little_finger", "right_ring_finger", "right_middle_finger",
        "right_fore_finger", "right_thumb_bend", "right_thumb_rotation",
        "waist_yaw", "waist_pitch",
        "hip_pitch", "knee_pitch",
    ]
    ft_axes = ["fx", "fy", "fz", "tx", "ty", "tz"]
    dual_arm_ft_axes = [
        "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
        "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
    ]

    # 新增腕部相机的 features
    features = {
        "observation.state": {"dtype": "float32", "shape": (32,), "names": [motors]},
        "action": {"dtype": "float32", "shape": (32,), "names": [motors]},
        "observation.velocity": {"dtype": "float32", "shape": (32,), "names": [motors]},
        "observation.left_arm_ft": {"dtype": "float32", "shape": (6,), "names": [ft_axes]},
        "observation.right_arm_ft": {"dtype": "float32", "shape": (6,), "names": [ft_axes]},
        "observation.dual_arm_ft": {"dtype": "float32", "shape": (12,), "names": [dual_arm_ft_axes]},
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": (3, int(image_height), int(image_width)),
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "image",
            "shape": (3, int(image_height), int(image_width)),
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "image",
            "shape": (3, int(image_height), int(image_width)),
            "names": ["channels", "height", "width"],
        },
    }

    if dataset_path.exists():
        print(f"\n删除已存在的数据集: {dataset_path}")
        shutil.rmtree(dataset_path)

    print(f"创建 LeRobot 数据集（带腕部相机 + 双臂FT）: {dataset_path}")
    dataset = LeRobotDataset.create(
        repo_id=str(dataset_path),
        fps=fps,
        robot_type="aloha",
        features=features,
        use_videos=False,
        tolerance_s=0.1,
    )
    return dataset


def _count_episodes_jsonl(dataset_path: Path) -> int:
    ep_path = dataset_path / "meta" / "episodes.jsonl"
    if not ep_path.exists():
        return 0
    n = 0
    with ep_path.open("r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def patch_info_splits(dataset_path: Path, split_name: str) -> None:
    """把 meta/info.json 的 splits 修成只有一个 split（train 或 test），方便后续脚本读取。"""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return
    try:
        info = json.loads(info_path.read_text())
    except Exception:
        return
    total_episodes = _count_episodes_jsonl(dataset_path)
    info["total_episodes"] = total_episodes
    info["splits"] = {split_name: f"0:{total_episodes}"}
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")


def fix_parquet_metadata(parquet_file: Path) -> bool:
    table = pq.read_table(str(parquet_file))
    schema = table.schema
    metadata = dict(schema.metadata or {})

    if b"huggingface" in metadata:
        hf_metadata = json.loads(metadata[b"huggingface"].decode())
        features = hf_metadata.get("info", {}).get("features", {})

        modified = False
        for _, value in features.items():
            if isinstance(value, dict) and value.get("_type") == "List":
                value["_type"] = "Sequence"
                modified = True

        if modified:
            metadata[b"huggingface"] = json.dumps(hf_metadata).encode()
            new_schema = schema.with_metadata(metadata)
            new_table = table.cast(new_schema)
            pq.write_table(new_table, str(parquet_file))
            return True
    return False


def process_episode(status_file: Path, dataset: LeRobotDataset, task_name: str, image_height: int, image_width: int) -> int:
    """处理单个 episode，加载三个相机的图像"""
    status_data = load_status_data(status_file)

    episode_dir = status_file.parent.parent
    
    # 检查第一帧是否有 images 字段和头部相机（头部相机是必须的）
    if not status_data or "images" not in status_data[0]:
        return 0
    
    first_frame_images = status_data[0]["images"]
    if "head_camera" not in first_frame_images:
        return 0

    all_states: list[np.ndarray] = []
    all_velocities: list[np.ndarray] = []
    all_head_images: list[np.ndarray] = []
    all_left_wrist_images: list[np.ndarray] = []
    all_right_wrist_images: list[np.ndarray] = []
    all_left_arm_ft: list[np.ndarray] = []
    all_right_arm_ft: list[np.ndarray] = []
    all_dual_arm_ft: list[np.ndarray] = []

    # 创建全黑图像的辅助函数
    def create_black_image(height: int, width: int) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    for frame in status_data:
        try:
            state = extract_state_32d(frame)
            velocity = extract_velocity_32d(frame)
            left_arm_ft = extract_arm_ft_6d(frame, "left_arm_6dof")
            right_arm_ft = extract_arm_ft_6d(frame, "right_arm_6dof")
            dual_arm_ft = np.concatenate([left_arm_ft, right_arm_ft]).astype(np.float32)

            # 从 frame['images'] 字段读取每个相机的图像路径
            if "images" not in frame:
                continue
            
            images = frame["images"]
            
            # 头部相机是必须的，如果没有则跳过
            if "head_camera" not in images:
                continue
            
            head_img_rel_path = images["head_camera"].get("color")
            if head_img_rel_path is None:
                continue
            
            head_img_path = episode_dir / head_img_rel_path
            if not head_img_path.exists():
                continue
            
            # 加载头部相机图像（必须成功）
            head_image = load_image(head_img_path, image_height=image_height, image_width=image_width)
            
            # 尝试加载左腕相机，失败则用全黑图像
            left_wrist_image = None
            if "left_hand_camera" in images:
                left_wrist_rel_path = images["left_hand_camera"].get("color")
                if left_wrist_rel_path is not None:
                    left_wrist_path = episode_dir / left_wrist_rel_path
                    if left_wrist_path.exists():
                        try:
                            left_wrist_image = load_image(left_wrist_path, image_height=image_height, image_width=image_width)
                        except Exception:
                            pass
            if left_wrist_image is None:
                left_wrist_image = create_black_image(image_height, image_width)
            
            # 尝试加载右腕相机，失败则用全黑图像
            right_wrist_image = None
            if "right_hand_camera" in images:
                right_wrist_rel_path = images["right_hand_camera"].get("color")
                if right_wrist_rel_path is not None:
                    right_wrist_path = episode_dir / right_wrist_rel_path
                    if right_wrist_path.exists():
                        try:
                            right_wrist_image = load_image(right_wrist_path, image_height=image_height, image_width=image_width)
                        except Exception:
                            pass
            if right_wrist_image is None:
                right_wrist_image = create_black_image(image_height, image_width)
            
            all_states.append(state)
            all_velocities.append(velocity)
            all_head_images.append(head_image)
            all_left_wrist_images.append(left_wrist_image)
            all_right_wrist_images.append(right_wrist_image)
            all_left_arm_ft.append(left_arm_ft)
            all_right_arm_ft.append(right_arm_ft)
            all_dual_arm_ft.append(dual_arm_ft)
        except Exception as e:
            print(f"  ⚠️  跳过异常帧: {e}")
            continue

    if len(all_states) == 0:
        return 0

    for i in range(len(all_states)):
        action = all_states[i + 1] if i < len(all_states) - 1 else all_states[i]
        frame_data = {
            "observation.state": torch.from_numpy(all_states[i]),
            "action": torch.from_numpy(action),
            "observation.velocity": torch.from_numpy(all_velocities[i]),
            "observation.left_arm_ft": torch.from_numpy(all_left_arm_ft[i]),
            "observation.right_arm_ft": torch.from_numpy(all_right_arm_ft[i]),
            "observation.dual_arm_ft": torch.from_numpy(all_dual_arm_ft[i]),
            "observation.images.cam_high": all_head_images[i],
            "observation.images.cam_left_wrist": all_left_wrist_images[i],
            "observation.images.cam_right_wrist": all_right_wrist_images[i],
            "task": task_name,
        }
        dataset.add_frame(frame_data)

    dataset.save_episode()
    return len(all_states)


def _filter_by_min_frames(status_files: list[Path], min_frames: int) -> list[Path]:
    if min_frames <= 0:
        return status_files
    filtered: list[Path] = []
    for status_file in tqdm(status_files, desc="检查帧数", unit="episode"):
        if count_frames(status_file) >= min_frames:
            filtered.append(status_file)
    return filtered


def _split_train_test(status_files: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(status_files))
    rng.shuffle(idxs)
    shuffled = [status_files[i] for i in idxs.tolist()]

    if len(shuffled) <= 1:
        return shuffled, []

    train_n = int(len(shuffled) * float(train_ratio))
    train_n = max(1, min(train_n, len(shuffled) - 1))  # 保证 test 至少 1 个
    return shuffled[:train_n], shuffled[train_n:]


def _convert_subset(
    subset_name: str,
    subset_files: list[Path],
    dataset_path: Path,
    task: str,
    fps: int,
    image_height: int,
    image_width: int,
) -> tuple[int, int]:
    """返回 (success_episodes, total_frames)"""
    if not subset_files:
        print(f"\n[WARN] {subset_name}: 没有 episode，跳过生成: {dataset_path}")
        return 0, 0

    dataset = create_lerobot_dataset(dataset_path, fps=fps, image_height=image_height, image_width=image_width)
    total_frames = 0
    success_count = 0
    for status_file in tqdm(subset_files, desc=f"转换 {subset_name}", unit="episode"):
        try:
            num_frames = process_episode(status_file, dataset, task, image_height=image_height, image_width=image_width)
            if num_frames > 0:
                total_frames += num_frames
                success_count += 1
        except Exception as e:
            print(f"\n❌ 处理失败: {status_file}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{subset_name}: 修复 parquet 元数据...")
    data_dir = dataset_path / "data"
    parquet_files = list(data_dir.rglob("*.parquet"))
    fixed_count = 0
    for pq_file in parquet_files:
        if fix_parquet_metadata(pq_file):
            fixed_count += 1
    print(f"{subset_name}: ✓ 修复了 {fixed_count}/{len(parquet_files)} 个 parquet 文件")
    return success_count, total_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通用数据转换脚本（32维 + 腕部相机）+ 自动划分 train/test（两个数据集输出）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("output_name", help="输出数据集名称（会生成 <name>_train 和 <name>_test）")
    parser.add_argument("--task", required=True, help="任务描述/提示词（写入 LeRobot 的 task 字段）")

    parser.add_argument("--sources", nargs="+", default=None, help="数据源列表，格式: 日期-关键词（与原脚本一致）")
    parser.add_argument(
        "--sources-except",
        nargs="+",
        default=None,
        help="排除模式：日期-排除关键词；该日期下所有不包含该关键词的 *_parser 文件夹都合并进一个任务",
    )

    parser.add_argument(
        "--output-dir",
        default="/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data",
        help="数据输出根目录（默认: /iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data）",
    )
    parser.add_argument("--min-frames", type=int, default=50, help="最小帧数阈值（默认: 50）")
    parser.add_argument("--fps", type=int, default=30, help="目标帧率（默认: 30）")
    parser.add_argument("--image-height", type=int, default=720, help="输出图像高度（默认: 720）")
    parser.add_argument("--image-width", type=int, default=1280, help="输出图像宽度（默认: 1280）")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="训练集比例（episode 级别，默认 0.9）")
    parser.add_argument("--seed", type=int, default=0, help="划分随机种子（默认 0）")

    args = parser.parse_args()

    if (not args.sources) and (not args.sources_except):
        raise SystemExit("必须指定 --sources 或 --sources-except 其中之一")
    if args.sources and args.sources_except:
        raise SystemExit("--sources 和 --sources-except 不能同时使用")

    output_root = Path(args.output_dir)

    train_dir = output_root / f"{args.output_name}_train"
    test_dir = output_root / f"{args.output_name}_test"

    print("=" * 80)
    print("通用数据转换脚本 - 32维配置 + 腕部相机 + 双臂FT（train/test 两个数据集输出）")
    print("=" * 80)
    print(f"输出(train): {train_dir}")
    print(f"输出(test):  {test_dir}")
    print(f"任务描述: {args.task}")
    print(f"最小帧数: {args.min_frames}")
    print(f"FPS: {args.fps}")
    print(f"图像分辨率: {args.image_height}x{args.image_width}")
    print(f"相机: cam_high (头部), cam_left_wrist (左手), cam_right_wrist (右手)")
    print(f"train_ratio: {args.train_ratio} seed: {args.seed}")
    if args.sources:
        print(f"模式: sources = {', '.join(args.sources)}")
    else:
        print(f"模式: sources-except = {', '.join(args.sources_except)}")
    print("=" * 80 + "\n")

    all_status_files: list[Path] = []

    if args.sources:
        for source_spec in args.sources:
            try:
                base_dir, keyword, is_20251224 = parse_data_source(source_spec)
                print(f"处理数据源: {source_spec}")
                print(f"  目录: {base_dir}")
                print(f"  关键词: {keyword}")
                print(f"  结构: {'20251224特殊结构' if is_20251224 else '通用结构'}")

                if is_20251224:
                    files = find_status_files_20251224(base_dir, keyword)
                else:
                    files = find_status_files_generic(base_dir, keyword)

                print(f"  找到: {len(files)} 个 episode")
                all_status_files.extend(files)
            except Exception as e:
                print(f"  ❌ 错误: {e}")
                continue
    else:
        for source_spec in args.sources_except:
            try:
                base_dir, keyword, is_20251224 = parse_data_source(source_spec)
                if is_20251224:
                    raise ValueError("20251224 特殊结构暂不支持 --sources-except（因为它不以 *_parser 文件夹组织任务）")

                print(f"处理排除数据源: {source_spec}")
                print(f"  目录: {base_dir}")
                print(f"  排除关键词: {keyword}")

                files = find_status_files_generic_except(base_dir, keyword)
                print(f"  找到: {len(files)} 个 episode（排除包含 {keyword!r} 的 *_parser 文件夹）")
                all_status_files.extend(files)
            except Exception as e:
                print(f"  ❌ 错误: {e}")
                continue

    if not all_status_files:
        print("\n❌ 没有找到任何数据！")
        return

    print(f"\n总共找到: {len(all_status_files)} 个 episode")

    print(f"\n过滤帧数 < {args.min_frames} 的 episode...")
    all_status_files = _filter_by_min_frames(all_status_files, args.min_frames)
    print(f"过滤后剩余: {len(all_status_files)} 个 episode")
    if not all_status_files:
        print("\n❌ 过滤后没有剩余数据！")
        return

    train_files, test_files = _split_train_test(all_status_files, train_ratio=args.train_ratio, seed=args.seed)
    print(f"\n划分完成：train={len(train_files)} test={len(test_files)}（episode 级别）")

    print("\n" + "=" * 80)
    print("开始转换 train/test 两个数据集")
    print("=" * 80 + "\n")

    train_success, train_frames = _convert_subset(
        "train",
        train_files,
        train_dir,
        args.task,
        args.fps,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    patch_info_splits(train_dir, "train")

    test_success, test_frames = _convert_subset(
        "test",
        test_files,
        test_dir,
        args.task,
        args.fps,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    patch_info_splits(test_dir, "test")

    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)
    print(f"train 数据集路径: {train_dir}")
    print(f"  成功转换: {train_success}/{len(train_files)} 个 episode")
    print(f"  总帧数: {train_frames}")
    print(f"test  数据集路径: {test_dir}")
    print(f"  成功转换: {test_success}/{len(test_files)} 个 episode")
    print(f"  总帧数: {test_frames}")
    print(f"任务描述: {args.task}")
    print(f"FPS: {args.fps}")
    print(f"状态维度: 32 (头2 + 左臂7 + 左手6 + 右臂7 + 右手6 + 腰2 + 腿2)")
    print("力矩维度: 左臂6 + 右臂6 + 合并12 (fx, fy, fz, tx, ty, tz)")
    print(f"相机: 3 个（head + left_wrist + right_wrist）")
    print("=" * 80)


if __name__ == "__main__":
    main()
