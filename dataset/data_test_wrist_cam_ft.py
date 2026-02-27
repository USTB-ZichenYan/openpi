import argparse
import pyarrow.parquet as pq
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import cv2

# ====== Camera keys ======
CAM_HIGH_KEY = "observation.images.cam_high"
CAM_LEFT_WRIST_KEY = "observation.images.cam_left_wrist"
CAM_RIGHT_WRIST_KEY = "observation.images.cam_right_wrist"
ACTION_KEY = "action"

# ====== FT keys (raw only) ======
LEFT_FT_RAW_KEY = "observation.left_arm_ft"
RIGHT_FT_RAW_KEY = "observation.right_arm_ft"
LEFT_EE_XYZ_KEY = "observation.left_ee_xyz"
RIGHT_EE_XYZ_KEY = "observation.right_ee_xyz"
CONTACT_FLAG_KEY = "contact_flag"
STAGE_ID_KEY = "stage_id"
STAGE_LABEL_KEY = "stage_label"

STAGE_LABELS = {
    0: "free",
    1: "contact",
    2: "lift",
    3: "carry",
}

# DATA_DIR = Path(
#     "/home/SENSETIME/yanzichen/data/file/dataset/grab_stool_wrist_cut/chunk-000"
# )


DATA_DIR = Path(
    "/home/SENSETIME/yanzichen/data/file/dataset/0206_ft_train/data/chunk-000"
)



# Avoid Tk toolbar icon resize crash in some environments.
mpl.rcParams["toolbar"] = "None"

# 32维动作映射（索引 -> 关节/部位）
# 头部(2): 0 head_pitch, 1 head_yaw
# 左臂(7): 2 left_shoulder_pitch, 3 left_shoulder_roll, 4 left_shoulder_yaw,
#         5 left_elbow_pitch, 6 left_wrist_yaw, 7 left_wrist_pitch, 8 left_wrist_roll
# 左手(6): 9 left_little_finger, 10 left_ring_finger, 11 left_middle_finger,
#         12 left_fore_finger, 13 left_thumb_bend, 14 left_thumb_rotation
# 右臂(7): 15 right_shoulder_pitch, 16 right_shoulder_roll, 17 right_shoulder_yaw,
#         18 right_elbow_pitch, 19 right_wrist_yaw, 20 right_wrist_pitch, 21 right_wrist_roll
# 右手(6): 22 right_little_finger, 23 right_ring_finger, 24 right_middle_finger,
#         25 right_fore_finger, 26 right_thumb_bend, 27 right_thumb_rotation
# 腰部(2): 28 waist_yaw, 29 waist_pitch
# 腿部(2): 30 hip_pitch, 31 knee_pitch

# ====== 关节 index（按你现在 32D 约定） ======
LEFT_ARM_IDX  = list(range(2, 9))    # 左臂 7 DoF
RIGHT_ARM_IDX = list(range(15, 22))  # 右臂 7 DoF
JOINT_NAMES = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
]


def load_episode(pq_path: Path):
    schema = pq.read_schema(pq_path)
    available = set(schema.names)
    columns = [
        c
        for c in [
            CAM_HIGH_KEY,
            CAM_LEFT_WRIST_KEY,
            CAM_RIGHT_WRIST_KEY,
            ACTION_KEY,
            LEFT_FT_RAW_KEY,
            RIGHT_FT_RAW_KEY,
            LEFT_EE_XYZ_KEY,
            RIGHT_EE_XYZ_KEY,
            CONTACT_FLAG_KEY,
            STAGE_ID_KEY,
            STAGE_LABEL_KEY,
        ]
        if c in available
    ]
    table = pq.read_table(pq_path, columns=columns)
    data = table.to_pydict()

    cam_high = [_decode_image(img) for img in data[CAM_HIGH_KEY]]
    cam_left_wrist = [_decode_image(img) for img in data[CAM_LEFT_WRIST_KEY]]
    cam_right_wrist = [_decode_image(img) for img in data[CAM_RIGHT_WRIST_KEY]]
    actions = np.asarray(data[ACTION_KEY], dtype=np.float32)

    left_arm = actions[:, LEFT_ARM_IDX]     # [T, 7]
    right_arm = actions[:, RIGHT_ARM_IDX]   # [T, 7]

    t = actions.shape[0]

    left_fx = None
    left_fy = None
    left_fz = None
    right_fx = None
    right_fy = None
    right_fz = None

    if LEFT_FT_RAW_KEY not in data or RIGHT_FT_RAW_KEY not in data:
        raise KeyError(f"Missing raw FT columns: {LEFT_FT_RAW_KEY}, {RIGHT_FT_RAW_KEY}")

    left_ft_raw = np.asarray(data[LEFT_FT_RAW_KEY], dtype=np.float32)
    right_ft_raw = np.asarray(data[RIGHT_FT_RAW_KEY], dtype=np.float32)
    left_fx = left_ft_raw[:, 0]
    left_fy = left_ft_raw[:, 1]
    left_fz = left_ft_raw[:, 2]
    right_fx = right_ft_raw[:, 0]
    right_fy = right_ft_raw[:, 1]
    right_fz = right_ft_raw[:, 2]

    contact_flag = None
    if CONTACT_FLAG_KEY in data:
        contact_flag = np.asarray(data[CONTACT_FLAG_KEY], dtype=np.float32)
    stage_id = None
    if STAGE_ID_KEY in data:
        stage_id = np.asarray(data[STAGE_ID_KEY], dtype=np.float32)
    stage_label = data.get(STAGE_LABEL_KEY)
    if contact_flag is None:
        if stage_id is not None:
            contact_flag = (np.asarray(stage_id, dtype=np.float32) > 0).astype(np.float32)
        else:
            contact_flag = np.zeros((t,), dtype=np.float32)
    if stage_id is None:
        stage_id = np.zeros((t,), dtype=np.float32)
    else:
        stage_id = np.asarray(stage_id, dtype=np.float32)

    if LEFT_EE_XYZ_KEY in data and RIGHT_EE_XYZ_KEY in data:
        left_ee_xyz = np.asarray(data[LEFT_EE_XYZ_KEY], dtype=np.float32)
        right_ee_xyz = np.asarray(data[RIGHT_EE_XYZ_KEY], dtype=np.float32)
        ee_source = "parquet"
    else:
        left_ee_xyz = np.zeros((t, 3), dtype=np.float32)
        right_ee_xyz = np.zeros((t, 3), dtype=np.float32)
        ee_source = "missing"

    if left_ee_xyz.shape[0] != t or right_ee_xyz.shape[0] != t:
        left_ee_xyz = np.zeros((t, 3), dtype=np.float32)
        right_ee_xyz = np.zeros((t, 3), dtype=np.float32)
        ee_source = "invalid_shape"

    left_ee_dxyz = left_ee_xyz - left_ee_xyz[:1]
    right_ee_dxyz = right_ee_xyz - right_ee_xyz[:1]

    ft = {
        "left_fx": left_fx,
        "left_fy": left_fy,
        "left_fz": left_fz,
        "right_fx": right_fx,
        "right_fy": right_fy,
        "right_fz": right_fz,
        "contact_flag": contact_flag,
        "stage_id": stage_id,
        "stage_label": stage_label,
        "left_ee_dxyz": left_ee_dxyz,
        "right_ee_dxyz": right_ee_dxyz,
        "ee_source": ee_source,
    }
    return cam_high, cam_left_wrist, cam_right_wrist, left_arm, right_arm, ft


def extract_stage_segments(stage_id: np.ndarray):
    s = np.asarray(stage_id, dtype=np.int32).reshape(-1)
    if s.size == 0:
        return []
    segments = []
    start = 0
    cur = int(s[0])
    for i in range(1, s.size):
        v = int(s[i])
        if v != cur:
            segments.append((cur, start, i - 1))
            start = i
            cur = v
    segments.append((cur, start, s.size - 1))
    return segments


def _decode_image(img):
    """Decode image entries from parquet into a numpy array suitable for imshow."""
    if isinstance(img, dict):
        if "bytes" in img and img["bytes"] is not None:
            return _decode_image(img["bytes"])
        if "path" in img and img["path"] is not None:
            path = str(img["path"])
            decoded = cv2.imread(path, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"Failed to read image from path: {path}")
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        raise ValueError("Image dict missing 'bytes' or 'path' fields.")

    if isinstance(img, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(img, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode image bytes.")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    if isinstance(img, str):
        decoded = cv2.imread(img, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Failed to read image from path: {img}")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    if isinstance(img, np.ndarray):
        return img

    # Fallback for list-like or scalar types.
    arr = np.array(img)
    if arr.dtype == object and arr.size == 1 and isinstance(arr.item(), (bytes, bytearray, memoryview)):
        return _decode_image(arr.item())
    return arr


def collect_parquet_files(data_dir: Path, selectors: Optional[list[str]]) -> list[Path]:
    if not selectors:
        return sorted(data_dir.glob("episode_*.parquet"))

    out: list[Path] = []
    seen: set[Path] = set()
    for raw in selectors:
        parts = [x.strip() for x in str(raw).split(",") if x.strip()]
        for token in parts:
            token_path = Path(token)
            candidates: list[Path] = []

            if token_path.is_absolute():
                if token_path.exists():
                    candidates = [token_path]
            else:
                if any(ch in token for ch in "*?[]"):
                    candidates = sorted(data_dir.glob(token))
                else:
                    candidate = data_dir / token
                    if candidate.exists():
                        candidates = [candidate]

            if not candidates:
                raise FileNotFoundError(f"No parquet matched selector: {token}")

            for c in candidates:
                c = c.resolve()
                if c.suffix != ".parquet":
                    continue
                if c not in seen:
                    seen.add(c)
                    out.append(c)

    return out


def play_episode(cam_high, cam_left_wrist, cam_right_wrist, left_arm, right_arm, ft, episode_name):
    cams = [
        ("cam_high", cam_high),
        ("cam_left_wrist", cam_left_wrist),
        ("cam_right_wrist", cam_right_wrist),
    ]
    lengths = [
        len(left_arm),
        len(right_arm),
        len(ft["left_fx"]),
        len(ft["right_fx"]),
        len(ft["contact_flag"]),
        len(ft["stage_id"]),
        len(ft["left_ee_dxyz"]),
        len(ft["right_ee_dxyz"]),
    ] + [len(images) for _, images in cams]
    T = min(lengths)

    plt.ion()
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0])
    fig.suptitle(f"Episode: {episode_name}")

    img_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    left_ax = fig.add_subplot(gs[1, :])
    right_ax = fig.add_subplot(gs[2, :])
    left_ft_ax = fig.add_subplot(gs[3, 0])
    right_ft_ax = fig.add_subplot(gs[3, 1])
    status_ax = fig.add_subplot(gs[3, 2])
    ee_x_ax = fig.add_subplot(gs[4, 0])
    ee_y_ax = fig.add_subplot(gs[4, 1])
    ee_z_ax = fig.add_subplot(gs[4, 2])

    paused = False
    restart = False
    step_delta = 0

    def on_key(event):
        nonlocal paused, restart, step_delta
        if event.key in (" ", "p"):
            paused = not paused
        elif event.key in ("r", "R"):
            restart = True
        elif event.key in ("left", "a"):
            step_delta = -1
            paused = True
        elif event.key in ("right", "d"):
            step_delta = 1
            paused = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    # 初始化图像
    img_plots = []
    for ax, (name, images) in zip(img_axes, cams):
        img_plot = ax.imshow(images[0])
        ax.axis("off")
        ax.set_title(name)
        img_plots.append(img_plot)

    # 初始化曲线
    left_lines = [
        left_ax.plot([], [], label=f"L_{name}")[0] for name in JOINT_NAMES
    ]
    right_lines = [
        right_ax.plot([], [], label=f"R_{name}")[0] for name in JOINT_NAMES
    ]
    left_ft_lines = [
        left_ft_ax.plot([], [], label="L_raw_Fx")[0],
        left_ft_ax.plot([], [], label="L_raw_Fy")[0],
        left_ft_ax.plot([], [], label="L_raw_Fz")[0],
    ]
    right_ft_lines = [
        right_ft_ax.plot([], [], label="R_raw_Fx")[0],
        right_ft_ax.plot([], [], label="R_raw_Fy")[0],
        right_ft_ax.plot([], [], label="R_raw_Fz")[0],
    ]
    status_lines = [
        status_ax.step([], [], where="post", label="contact_flag")[0],
        status_ax.step([], [], where="post", label="stage_id")[0],
    ]
    ee_x_lines = [
        ee_x_ax.plot([], [], label="L_dX")[0],
        ee_x_ax.plot([], [], "--", label="R_dX", alpha=0.9)[0],
    ]
    ee_y_lines = [
        ee_y_ax.plot([], [], label="L_dY")[0],
        ee_y_ax.plot([], [], "--", label="R_dY", alpha=0.9)[0],
    ]
    ee_z_lines = [
        ee_z_ax.plot([], [], label="L_dZ")[0],
        ee_z_ax.plot([], [], "--", label="R_dZ", alpha=0.9)[0],
    ]

    left_ax.set_xlim(0, T)
    right_ax.set_xlim(0, T)
    left_ax.set_title("Left arm joints")
    right_ax.set_title("Right arm joints")
    left_ax.legend(ncol=7, fontsize=8)
    right_ax.legend(ncol=7, fontsize=8)
    left_ft_ax.set_title("Left FT (raw sensor frame)")
    right_ft_ax.set_title("Right FT (raw sensor frame)")
    status_ax.set_title("Contact / Stage")
    left_ft_ax.legend(fontsize=8)
    right_ft_ax.legend(fontsize=8)
    status_ax.legend(fontsize=8)
    status_ax.set_ylim(-0.5, 3.5)
    status_ax.set_yticks([0, 1, 2, 3])
    status_ax.set_yticklabels([STAGE_LABELS[i] for i in [0, 1, 2, 3]])
    left_ft_ax.set_xlim(0, T - 1)
    right_ft_ax.set_xlim(0, T - 1)
    status_ax.set_xlim(0, T - 1)
    ee_x_ax.set_xlim(0, T - 1)
    ee_y_ax.set_xlim(0, T - 1)
    ee_z_ax.set_xlim(0, T - 1)
    ee_x_ax.set_title(f"EE dX (source={ft.get('ee_source', 'unknown')})")
    ee_y_ax.set_title("EE dY")
    ee_z_ax.set_title("EE dZ")
    ee_x_ax.set_ylabel("m")
    ee_y_ax.set_ylabel("m")
    ee_z_ax.set_ylabel("m")
    ee_x_ax.legend(fontsize=8)
    ee_y_ax.legend(fontsize=8)
    ee_z_ax.legend(fontsize=8)

    # 标记每个 stage 的起止帧
    stage_segments = extract_stage_segments(ft["stage_id"][:T])
    for stage, s, e in stage_segments:
        name = STAGE_LABELS.get(int(stage), f"stage{stage}")
        status_ax.axvline(x=s, color="tab:green", linestyle="--", linewidth=0.8, alpha=0.7)
        status_ax.axvline(x=e, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.7)
        mid = 0.5 * (s + e)
        status_ax.text(mid, float(stage) + 0.15, f"{name}[{s}-{e}]", fontsize=7, ha="center", va="bottom")
        left_ft_ax.axvline(x=s, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        left_ft_ax.axvline(x=e, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        right_ft_ax.axvline(x=s, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        right_ft_ax.axvline(x=e, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ee_x_ax.axvline(x=s, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ee_x_ax.axvline(x=e, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ee_y_ax.axvline(x=s, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ee_y_ax.axvline(x=e, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ee_z_ax.axvline(x=s, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        ee_z_ax.axvline(x=e, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

    def update_frame(frame_idx: int):
        for plot, (_, images) in zip(img_plots, cams):
            plot.set_data(images[frame_idx])
        for i in range(7):
            left_lines[i].set_data(range(frame_idx + 1), left_arm[: frame_idx + 1, i])
            right_lines[i].set_data(range(frame_idx + 1), right_arm[: frame_idx + 1, i])
        left_ft_lines[0].set_data(range(frame_idx + 1), ft["left_fx"][: frame_idx + 1])
        left_ft_lines[1].set_data(range(frame_idx + 1), ft["left_fy"][: frame_idx + 1])
        left_ft_lines[2].set_data(range(frame_idx + 1), ft["left_fz"][: frame_idx + 1])
        right_ft_lines[0].set_data(range(frame_idx + 1), ft["right_fx"][: frame_idx + 1])
        right_ft_lines[1].set_data(range(frame_idx + 1), ft["right_fy"][: frame_idx + 1])
        right_ft_lines[2].set_data(range(frame_idx + 1), ft["right_fz"][: frame_idx + 1])
        status_lines[0].set_data(range(frame_idx + 1), ft["contact_flag"][: frame_idx + 1])
        status_lines[1].set_data(range(frame_idx + 1), ft["stage_id"][: frame_idx + 1])
        ee_x_lines[0].set_data(range(frame_idx + 1), ft["left_ee_dxyz"][: frame_idx + 1, 0])
        ee_x_lines[1].set_data(range(frame_idx + 1), ft["right_ee_dxyz"][: frame_idx + 1, 0])
        ee_y_lines[0].set_data(range(frame_idx + 1), ft["left_ee_dxyz"][: frame_idx + 1, 1])
        ee_y_lines[1].set_data(range(frame_idx + 1), ft["right_ee_dxyz"][: frame_idx + 1, 1])
        ee_z_lines[0].set_data(range(frame_idx + 1), ft["left_ee_dxyz"][: frame_idx + 1, 2])
        ee_z_lines[1].set_data(range(frame_idx + 1), ft["right_ee_dxyz"][: frame_idx + 1, 2])

        left_ax.relim()
        left_ax.autoscale_view()
        right_ax.relim()
        right_ax.autoscale_view()
        left_ft_ax.relim()
        left_ft_ax.autoscale_view()
        left_ft_ax.set_xlim(0, T - 1)
        right_ft_ax.relim()
        right_ft_ax.autoscale_view()
        right_ft_ax.set_xlim(0, T - 1)
        status_ax.relim()
        status_ax.autoscale_view()
        status_ax.set_xlim(0, T - 1)
        status_ax.set_ylim(-0.5, 3.5)
        ee_x_ax.relim()
        ee_x_ax.autoscale_view()
        ee_x_ax.set_xlim(0, T - 1)
        ee_y_ax.relim()
        ee_y_ax.autoscale_view()
        ee_y_ax.set_xlim(0, T - 1)
        ee_z_ax.relim()
        ee_z_ax.autoscale_view()
        ee_z_ax.set_xlim(0, T - 1)

    t = 0
    while t < T and plt.fignum_exists(fig.number):
        while paused and plt.fignum_exists(fig.number):
            if restart:
                restart = False
                t = 0
                for i in range(7):
                    left_lines[i].set_data([], [])
                    right_lines[i].set_data([], [])
                for line in left_ft_lines + right_ft_lines + status_lines + ee_x_lines + ee_y_lines + ee_z_lines:
                    line.set_data([], [])
                for plot, (_, images) in zip(img_plots, cams):
                    plot.set_data(images[0])
                left_ax.relim()
                left_ax.autoscale_view()
                right_ax.relim()
                right_ax.autoscale_view()
                left_ft_ax.relim()
                left_ft_ax.autoscale_view()
                right_ft_ax.relim()
                right_ft_ax.autoscale_view()
                status_ax.relim()
                status_ax.autoscale_view()
                status_ax.set_ylim(-0.5, 3.5)
                ee_x_ax.relim()
                ee_x_ax.autoscale_view()
                ee_y_ax.relim()
                ee_y_ax.autoscale_view()
                ee_z_ax.relim()
                ee_z_ax.autoscale_view()
                plt.pause(0.01)
                continue

            if step_delta != 0:
                t = max(0, min(T - 1, t + step_delta))
                step_delta = 0
                update_frame(t)
                plt.pause(0.01)
                continue

            plt.pause(0.1)

        update_frame(t)
        plt.pause(0.03)
        t += 1

    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize wrist cams + FT + stage + EE pose")
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory containing episode_*.parquet",
    )
    parser.add_argument(
        "--parquet",
        action="append",
        default=None,
        help=(
            "Parquet selector (repeatable): absolute path, filename under --data-dir, "
            "or glob pattern like 'episode_000123*.parquet'"
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    parquet_files = collect_parquet_files(data_dir, args.parquet)
    print(f"Data dir: {data_dir}")
    print(f"Found {len(parquet_files)} episodes")

    for pq_file in parquet_files:
        print(f"\n▶ Playing {pq_file.name}")
        cam_high, cam_left_wrist, cam_right_wrist, left_arm, right_arm, ft = load_episode(pq_file)
        segs = extract_stage_segments(ft["stage_id"])
        if segs:
            seg_text = ", ".join(
                f"{STAGE_LABELS.get(int(st), st)}[{s}-{e}]"
                for st, s, e in segs
            )
            print(f"  stage segments: {seg_text}")
        print(f"  ee source: {ft.get('ee_source', 'unknown')}")

        if len(cam_high) < 10:
            print("  ⚠️ Episode too short, skip")
            continue

        play_episode(cam_high, cam_left_wrist, cam_right_wrist, left_arm, right_arm, ft, pq_file.name)

        key = input("Press ENTER for next episode, or q to quit: ")
        if key.lower().startswith("q"):
            break


if __name__ == "__main__":
    main()
