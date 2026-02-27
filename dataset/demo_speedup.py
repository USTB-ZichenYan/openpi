"""
DemoSpeedup: entropy-guided action chunk acceleration for demonstration data.

This script is a practical implementation of the pseudocode workflow:
1) estimate per-step entropy from proxy action samples,
2) denoise entropy with IsolationForest-like preprocessing,
3) get precision labels with HDBSCAN-like clustering,
4) piecewise downsample action suffixes into fixed-length speedup chunks.

Input:
  episode parquet files with an `action` column of shape [T, action_dim].

Output:
  parquet files with added columns:
    - action_speedup: [T, K, action_dim]
    - entropy: [T]
    - entropy_clean: [T]
    - entropy_precision: [T]
    - entropy_critical: [T] (0/1)
    - rbd_copy: replicated copy id
    - rbd_offset: temporal offset used by this replicated copy

  With RBD enabled, each output parquet contains concatenated rows from
  multiple replicated sub-trajectories (different offsets).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path(
    "/home/SENSETIME/yanzichen/data/file/dataset/20260206_grab/0206_1_grab_test/data/chunk-000"
)
DEFAULT_OUT_DIR = Path(
    "/home/SENSETIME/yanzichen/data/file/openpi/dataset/speedup_outputs/0206_1_grab_test/chunk-000"
)
ACTION_KEY = "action"


@dataclass(frozen=True)
class EntropyLabels:
    # precision 越大表示越“确定”（低熵）；critical_indices 这里定义为高熵集合。
    precision: np.ndarray  # [T], higher means higher precision
    critical_indices: np.ndarray  # 1D indices of critical timesteps
    cluster_labels: np.ndarray  # [T], HDBSCAN labels or fallback labels


def _proxy_sample_chunk(
    actions: np.ndarray,
    start_idx: int,
    chunk_size: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one proxy chunk by perturbing the action suffix with Gaussian noise."""
    # 从 t 开始截取长度 K 的动作块，不足 K 就用最后一个动作补齐，保证后续 shape 恒定。
    end = min(actions.shape[0], start_idx + chunk_size)
    chunk = actions[start_idx:end]
    if chunk.shape[0] < chunk_size:
        pad = np.repeat(chunk[-1:, :], chunk_size - chunk.shape[0], axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)

    if noise_std <= 0.0:
        return chunk.astype(np.float32)
    # 用噪声近似“生成式策略的随机采样”，用于估计该时刻动作分布离散程度。
    noise = rng.normal(loc=0.0, scale=noise_std, size=chunk.shape).astype(np.float32)
    return (chunk + noise).astype(np.float32)


def _gaussian_entropy_from_samples(samples: np.ndarray, eps: float = 1e-6) -> float:
    """
    Estimate differential entropy from sampled action chunks.
    samples: [N, K, A]
    """
    # 将每个 chunk [K, A] 展平为向量，样本矩阵 shape = [N, K*A]。
    flat = samples.reshape(samples.shape[0], -1).astype(np.float64)
    dim = flat.shape[1]
    if samples.shape[0] <= 1:
        return 0.0

    cov = np.cov(flat, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
        dim = 1

    cov = cast(np.ndarray, cov)
    # 数值稳定：给协方差加一个很小的对角线项，避免奇异矩阵。
    cov = cov + np.eye(dim, dtype=np.float64) * eps
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return 0.0
    return float(0.5 * (dim * (1.0 + np.log(2.0 * np.pi)) + logdet))


def get_entropy(
    actions: np.ndarray,
    *,
    num_samples: int = 32,
    chunk_size: int = 16,
    noise_std: float = 0.03,
    seed: int = 0,
) -> np.ndarray:
    """Compute per-step entropy list H_t from proxy action samples."""
    if actions.ndim != 2:
        raise ValueError(f"`actions` must be 2D [T, A], got shape={actions.shape}")
    if actions.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if num_samples < 2:
        raise ValueError("`num_samples` must be >= 2.")

    rng = np.random.default_rng(seed)
    entropies = np.zeros((actions.shape[0],), dtype=np.float32)
    for t in range(actions.shape[0]):
        # 对每个时刻 t 采样 N 个候选 chunk，并计算该时刻熵 H_t。
        sampled = np.stack(
            [
                _proxy_sample_chunk(
                    actions=actions,
                    start_idx=t,
                    chunk_size=chunk_size,
                    noise_std=noise_std,
                    rng=rng,
                )
                for _ in range(num_samples)
            ],
            axis=0,
        )
        entropies[t] = _gaussian_entropy_from_samples(sampled)
    return entropies


def preprocess_entropy_with_isolation_forest(
    entropy: np.ndarray, *, contamination: float = 0.08, seed: int = 0
) -> np.ndarray:
    """
    Isolation-forest denoising for H_list.
    If sklearn is unavailable, fallback to percentile clipping.
    """
    values = entropy.astype(np.float32).copy()
    if values.size == 0:
        return values

    try:
        from sklearn.ensemble import IsolationForest  # type: ignore

        # 用孤立森林识别异常熵点，再裁剪到 inlier 区间抑制尖峰噪声。
        model = IsolationForest(
            contamination=contamination,
            random_state=seed,
            n_estimators=200,
        )
        pred = model.fit_predict(values.reshape(-1, 1))
        inlier = pred == 1
        if not np.any(inlier):
            return values
        lo = float(values[inlier].min())
        hi = float(values[inlier].max())
        values = np.clip(values, lo, hi)
        return values
    except Exception:
        # 无 sklearn 时退化到分位裁剪，行为更稳但不如 IF 精细。
        lo = float(np.percentile(values, 5.0))
        hi = float(np.percentile(values, 95.0))
        return np.clip(values, lo, hi)


def h_dbscan_cluster(
    entropy: np.ndarray,
    *,
    min_cluster_size: int = 8,
    min_samples: int = 8,
    critical_quantile_fallback: float = 75.0,
) -> EntropyLabels:
    """
    Label precision {P, C} from entropy sequence.
    C is picked as the highest-entropy dense cluster (or quantile fallback).
    """
    values = entropy.astype(np.float32)
    n = values.shape[0]
    if n == 0:
        return EntropyLabels(
            precision=np.zeros((0,), dtype=np.float32),
            critical_indices=np.zeros((0,), dtype=np.int64),
            cluster_labels=np.zeros((0,), dtype=np.int32),
        )

    # 连续精度分数：低熵 => precision 高；高熵 => precision 低。
    vmin = float(values.min())
    vmax = float(values.max())
    precision = 1.0 - (values - vmin) / (max(vmax - vmin, 1e-8))
    precision = np.clip(precision, 0.0, 1.0).astype(np.float32)

    standardized = (values - float(values.mean())) / (float(values.std()) + 1e-8)
    features = standardized.reshape(-1, 1)

    critical_indices: np.ndarray
    labels: np.ndarray
    try:
        # Primary path: use HDBSCAN (package name: hdbscan).
        import hdbscan  # type: ignore

        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method="eom",
        ).fit_predict(features).astype(np.int32)
    except Exception:
        try:
            # Fallback path: DBSCAN if hdbscan is unavailable.
            from sklearn.cluster import DBSCAN  # type: ignore

            labels = DBSCAN(eps=0.35, min_samples=min_samples).fit_predict(features).astype(np.int32)
        except Exception:
            # Last fallback: high-quantile thresholding.
            threshold = float(np.percentile(values, critical_quantile_fallback))
            critical_indices = np.where(values >= threshold)[0].astype(np.int64)
            labels = np.where(values >= threshold, 1, 0).astype(np.int32)
            return EntropyLabels(
                precision=precision,
                critical_indices=critical_indices,
                cluster_labels=labels,
            )

    cluster_ids = [cid for cid in np.unique(labels).tolist() if cid != -1]
    if cluster_ids:
        # 当前实现把“平均熵最高”的簇视为 critical（更可加速的区域）。
        cluster_mean = {cid: float(values[labels == cid].mean()) for cid in cluster_ids}
        critical_cluster = max(cluster_mean, key=cluster_mean.get)
        critical_indices = np.where(labels == critical_cluster)[0].astype(np.int64)
    else:
        threshold = float(np.percentile(values, critical_quantile_fallback))
        critical_indices = np.where(values >= threshold)[0].astype(np.int64)

    return EntropyLabels(
        precision=precision,
        critical_indices=critical_indices,
        cluster_labels=labels,
    )


def piecewise_downsample_actions(
    actions_suffix: np.ndarray,
    *,
    critical_indices: np.ndarray,
    global_start_idx: int,
    r_high: int = 4,
    r_low: int = 2,
    start_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample actions in sub-trajectories with precision labels.
    If [i, i+r_high) belongs to critical set C, use r_high; else r_low.
    """
    if actions_suffix.ndim != 2:
        raise ValueError(f"`actions_suffix` must be 2D [L, A], got shape={actions_suffix.shape}")
    if actions_suffix.shape[0] == 0:
        return actions_suffix.copy(), np.zeros((0,), dtype=np.int64)
    if r_high <= 0 or r_low <= 0:
        raise ValueError("r_high and r_low must be positive.")

    cset = set(critical_indices.tolist())
    indices: list[int] = []
    i = max(0, start_offset)
    length = actions_suffix.shape[0]
    while i < length:
        global_i = global_start_idx + i
        window = range(global_i, global_i + r_high)
        # 若当前位置处在连续高熵窗口，则用大步长 r_high；否则用小步长 r_low。
        use_high = all(w in cset for w in window if w < global_start_idx + length)
        indices.append(i)
        i += r_high if use_high else r_low

    # 强制保留末帧，避免轨迹终点信息被跳过。
    if indices[-1] != length - 1:
        indices.append(length - 1)

    local_indices = np.array(indices, dtype=np.int64)
    return actions_suffix[local_indices], local_indices


def select_piecewise_indices(
    *,
    total_len: int,
    critical_indices: np.ndarray,
    r_high: int,
    r_low: int,
    start_offset: int,
) -> np.ndarray:
    """Select timestep indices for one replicated trajectory copy."""
    if total_len <= 0:
        return np.zeros((0,), dtype=np.int64)

    cset = set(critical_indices.tolist())
    i = max(0, start_offset)
    picked: list[int] = []
    while i < total_len:
        window = range(i, i + r_high)
        # 与 piecewise_downsample_actions 同一规则，但作用于整条轨迹索引选择。
        use_high = all(w in cset for w in window if w < total_len)
        picked.append(i)
        i += r_high if use_high else r_low

    if picked[-1] != total_len - 1:
        picked.append(total_len - 1)
    return np.array(picked, dtype=np.int64)


def _to_fixed_chunk(actions: np.ndarray, k: int) -> np.ndarray:
    # 训练目标需要固定长度 K：长则截断，短则重复末帧补齐。
    if actions.shape[0] >= k:
        return actions[:k].astype(np.float32)
    pad = np.repeat(actions[-1:, :], k - actions.shape[0], axis=0)
    return np.concatenate([actions, pad], axis=0).astype(np.float32)


def accelerate_episode(
    actions: np.ndarray,
    *,
    chunk_size: int,
    num_samples: int,
    noise_std: float,
    r_high: int,
    r_low: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build speedup chunk targets per timestep.

    Returns:
      action_speedup: [T, K, A]
      entropy_raw: [T]
      entropy_clean: [T]
      entropy_precision: [T]
      critical_mask: [T] {0,1}
    """
    # Stage 1: 熵估计（每个时刻一个 H_t）。
    entropy_raw = get_entropy(
        actions,
        num_samples=num_samples,
        chunk_size=chunk_size,
        noise_std=noise_std,
        seed=seed,
    )
    # Stage 2: 熵去噪 + 高熵区域标注。
    entropy_clean = preprocess_entropy_with_isolation_forest(entropy_raw, seed=seed)
    labels = h_dbscan_cluster(entropy_clean)

    t_max, action_dim = actions.shape
    speedup_chunks = np.zeros((t_max, chunk_size, action_dim), dtype=np.float32)
    # Stage 3: 逐时刻构建 speedup chunk 监督信号。
    for t in range(t_max):
        suffix = actions[t:, :]
        downsampled, _ = piecewise_downsample_actions(
            suffix,
            critical_indices=labels.critical_indices,
            global_start_idx=t,
            r_high=r_high,
            r_low=r_low,
        )
        speedup_chunks[t] = _to_fixed_chunk(downsampled, chunk_size)

    critical_mask = np.zeros((t_max,), dtype=np.int8)
    critical_mask[labels.critical_indices] = 1
    return (
        speedup_chunks,
        entropy_raw.astype(np.float32),
        entropy_clean.astype(np.float32),
        labels.precision.astype(np.float32),
        critical_mask,
    )


def run_demo_speedup(
    *,
    data_root: Path,
    out_dir: Path,
    chunk_size: int,
    num_samples: int,
    noise_std: float,
    r_high: int,
    r_low: int,
    rbd_copies: int,
    max_episodes: int | None,
    seed: int,
) -> None:
    parquet_files = sorted(data_root.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode_*.parquet found in {data_root}")
    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {data_root}")
    print(f"Output dir: {out_dir}")
    print(f"Episodes: {len(parquet_files)}")
    print(
        "Config: "
        f"chunk_size={chunk_size} num_samples={num_samples} noise_std={noise_std} "
        f"r_high={r_high} r_low={r_low} rbd_copies={rbd_copies}"
    )

    for idx, pq_file in enumerate(tqdm(parquet_files, desc="DemoSpeedup")):
        table = pq.read_table(pq_file)
        actions = np.asarray(table.column(ACTION_KEY).to_pylist(), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0:
            print(f"[Skip] {pq_file.name}: invalid actions shape {actions.shape}")
            continue

        action_speedup, entropy_raw, entropy_clean, entropy_precision, critical_mask = accelerate_episode(
            actions=actions,
            chunk_size=chunk_size,
            num_samples=num_samples,
            noise_std=noise_std,
            r_high=r_high,
            r_low=r_low,
            seed=seed + idx,
        )

        critical_indices = np.where(critical_mask > 0)[0].astype(np.int64)
        rbd_tables: list[pa.Table] = []
        copy_count = max(1, int(rbd_copies))
        total_len = actions.shape[0]

        for copy_id in range(copy_count):
            # RBD: 每个 copy 用不同起始 offset 抽样，减少“跳帧后观测覆盖下降”。
            offset = copy_id % max(r_high, 1)
            selected = select_piecewise_indices(
                total_len=total_len,
                critical_indices=critical_indices,
                r_high=r_high,
                r_low=r_low,
                start_offset=offset,
            )
            if selected.size == 0:
                continue

            # 先按 selected 取原始行，再追加 speedup/entropy/RBD 元信息列。
            base = table.take(pa.array(selected.tolist(), type=pa.int64()))
            action_speedup_sel = action_speedup[selected]
            entropy_sel = entropy_raw[selected]
            entropy_clean_sel = entropy_clean[selected]
            entropy_precision_sel = entropy_precision[selected]
            critical_sel = critical_mask[selected]
            copy_id_col = np.full((selected.shape[0],), copy_id, dtype=np.int32)
            offset_col = np.full((selected.shape[0],), offset, dtype=np.int32)

            base = base.append_column("action_speedup", pa.array(action_speedup_sel.tolist()))
            base = base.append_column("entropy", pa.array(entropy_sel.tolist(), type=pa.float32()))
            base = base.append_column("entropy_clean", pa.array(entropy_clean_sel.tolist(), type=pa.float32()))
            base = base.append_column("entropy_precision", pa.array(entropy_precision_sel.tolist(), type=pa.float32()))
            base = base.append_column("entropy_critical", pa.array(critical_sel.tolist(), type=pa.int8()))
            base = base.append_column("rbd_copy", pa.array(copy_id_col.tolist(), type=pa.int32()))
            base = base.append_column("rbd_offset", pa.array(offset_col.tolist(), type=pa.int32()))
            rbd_tables.append(base)

        if not rbd_tables:
            print(f"[Skip] {pq_file.name}: no valid RBD rows.")
            continue
        out_table = pa.concat_tables(rbd_tables)

        out_path = out_dir / pq_file.name
        pq.write_table(out_table, out_path)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DemoSpeedup implementation for parquet demonstrations")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=30, help="K in A_t_speedup[:K]")
    parser.add_argument("--num-samples", type=int, default=32, help="N proxy samples for entropy")
    parser.add_argument("--noise-std", type=float, default=0.03, help="Proxy sampling noise std")
    parser.add_argument("--r-high", type=int, default=4, help="high-entropy step size (larger = faster)")
    parser.add_argument("--r-low", type=int, default=2, help="low-entropy step size (smaller = finer)")
    parser.add_argument("--rbd-copies", type=int, default=None, help="replication count for RBD; default uses r_high")
    parser.add_argument("--max-episodes", type=int, default=None, help="debug option")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rbd_copies = args.rbd_copies if args.rbd_copies is not None else args.r_high

    run_demo_speedup(
        data_root=args.data_root,
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        num_samples=args.num_samples,
        noise_std=args.noise_std,
        r_high=args.r_high,
        r_low=args.r_low,
        rbd_copies=rbd_copies,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
