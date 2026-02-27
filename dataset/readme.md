Dataset Utilities

This folder contains four scripts used in the dual-arm dataset workflow: trim noisy prefixes, DTW align/resample, plot, and warm-start query.

1) data_mining.py
   - Purpose: Trim episode prefixes at the first point where BOTH image and action change
     for K consecutive steps (change-driven, not static-driven).
   - Input: `episode_*.parquet` under a single chunk directory.
   - Output: A trimmed chunk directory with the same parquet schema and a trim report.
   - Example:
     ```
     /home/SENSETIME/yanzichen/anaconda3/envs/nuplan/bin/python dataset/data_mining.py \
       --data-root /home/SENSETIME/yanzichen/data/file/dataset/grab_stool_train/data/chunk-000 \
       --task pick \
       --image-change-threshold 2.0 \
       --action-percentile 80 \
       --min-consistent-steps 3 \
       --out-dir /home/SENSETIME/yanzichen/data/file/dataset/grab_stool_train/data/chunk-000_trimmed \
       --report-out trimmed_episodes.txt
     ```

2) compute_arm_dtw_mean_std.py
   - Purpose: DTW-align and resample each episode to a fixed length (default 100), then save:
     - Per-episode aligned parquet files in a new chunk directory.
     - Mean/std trajectories and symmetry stats.
     - NOTE: Aligns all DOFs except hand joints (9–14, 22–27). Hand joints are fixed to
       the mean initial values across episodes.
   - Input: Trimmed chunk directory (default `chunk-000_trimmed`).
   - Output: DTW-aligned chunk directory (default `chunk-000_dtw`) plus stats files.
   - Example:
     ```
     /home/SENSETIME/yanzichen/anaconda3/envs/nuplan/bin/python dataset/compute_arm_dtw_mean_std.py
     ```

3) plot_arm_dtw_results.py
   - Purpose: Plot per-DOF trajectories and symmetry bars from DTW outputs.
   - Input: `mean_traj_32d_dtw100_trim.npy`, `std_traj_32d_dtw100_trim.npy`,
     `aligned_all_full_dtw100_trim.npy`, and `symmetry_stats_dtw100_trim.csv`.
   - Output: `plots_per_dof_dtw100_trim/` with per-DOF plots and a symmetry bar plot.
   - Example:
     ```
     /home/SENSETIME/yanzichen/anaconda3/envs/nuplan/bin/python dataset/plot_arm_dtw_results.py
     ```

4) warm_start_from_dtw.py
   - Purpose: Query DTW mean/std to generate a warm-start 32D pose or a chunk
     (chunk_size × 32) for flow matching.
   - Input: `mean_traj_32d_dtw100_trim.npy` and `std_traj_32d_dtw100_trim.npy`.
   - Output: Saves `warm_start.npy` (and optional plots) under `--out-dir`,
     prints the matched frame index.
   - Example:
     ```
     /home/SENSETIME/yanzichen/anaconda3/envs/nuplan/bin/python dataset/warm_start_from_dtw.py \
       --mean mean_traj_32d_dtw100_trim.npy \
       --std std_traj_32d_dtw100_trim.npy \
       --chunk-size 16 \
       --plot
     ```

Notes
- Image hashing uses aHash and requires Pillow when running `data_mining.py`.
- DTW alignment uses `dtaidistance`.
