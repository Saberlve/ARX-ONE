#!/usr/bin/env python3
"""分析 LeRobot 格式数据的同步质量。直接读取 parquet，支持指定 repo_id 和 episode。"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_episode_parquet(root: Path, episode_index: int):
    """根据 info.json 里的 data_path 格式找到 parquet 文件。"""
    info_path = root / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    chunks_size = info.get("chunks_size", 1000)
    chunk = episode_index // chunks_size
    data_path_tmpl = info["data_path"]
    fpath = root / data_path_tmpl.format(episode_chunk=chunk, episode_index=episode_index)

    if not fpath.is_file():
        return None
    return pd.read_parquet(fpath)


def analyze_episode(df: pd.DataFrame, fps: int, episode_index: int):
    ts_cols = [
        "sync_timestamp.head_ns",
        "sync_timestamp.left_wrist_ns",
        "sync_timestamp.right_wrist_ns",
        "sync_timestamp.left_arm_ns",
        "sync_timestamp.right_arm_ns",
    ]

    # 检查列是否存在
    available_cols = [c for c in ts_cols if c in df.columns]
    if not available_cols:
        print(f"Episode {episode_index}: 找不到 sync_timestamp 列，请确认是用新版采集代码保存的。")
        return None

    total_frames = len(df)
    target_dt = 1000.0 / fps

    print(f"\n{'='*60}")
    print(f"Episode {episode_index:04d} | 总帧数: {total_frames} | 目标 fps: {fps}")
    print(f"{'='*60}")

    # 1. 各模态的帧间隔稳定性
    print("\n【帧率稳定性】")
    sync_scores = []
    dt_stats = {}
    for col in available_cols:
        arr = df[col].to_numpy().astype(np.int64).flatten()
        dts = np.diff(arr).astype(np.float64) / 1e6  # ms
        dt_stats[col] = dts
        avg_dt = float(np.mean(dts))
        std_dt = float(np.std(dts))
        max_dt = float(np.max(dts))
        fps_actual = 1000.0 / avg_dt if avg_dt > 0 else 0
        over_threshold = int(np.sum(dts > target_dt + 15.0))
        print(f"  {col.split('.')[-1]:20s} 平均fps={fps_actual:.1f} 平均dt={avg_dt:.1f}ms std={std_dt:.2f}ms 最大dt={max_dt:.1f}ms 超时帧={over_threshold}/{len(dts)}")
        if std_dt < 3:
            sync_scores.append(2)
        elif std_dt < 5:
            sync_scores.append(1)
        else:
            sync_scores.append(0)

    # 2. 同一帧内各模态时间戳的最大差异（真正的同步指标）
    print("\n【模态间同步质量（同一帧内最大时间差）】")
    mat = np.stack([df[c].to_numpy().astype(np.int64).flatten() for c in available_cols], axis=1)
    max_diff_ns = np.max(mat, axis=1) - np.min(mat, axis=1)
    max_diff_ms = max_diff_ns / 1e6

    avg_diff = float(np.mean(max_diff_ms))
    p95_diff = float(np.percentile(max_diff_ms, 95))
    max_diff = float(np.max(max_diff_ms))
    bad_frames = int(np.sum(max_diff_ms > 10.0))
    print(f"  平均差异={avg_diff:.2f}ms P95={p95_diff:.2f}ms 最大差异={max_diff:.1f}ms")
    print(f"  差异>10ms 的帧: {bad_frames}/{total_frames} ({100*bad_frames/total_frames:.1f}%)")
    if avg_diff < 3:
        sync_scores.append(2)
    elif avg_diff < 5:
        sync_scores.append(1)
    else:
        sync_scores.append(0)

    # 3. timestamp 列的均匀性（LeRobot 自带的 timestamp）
    if "timestamp" in df.columns:
        ts_arr = df["timestamp"].to_numpy().astype(np.float64)
        dts_lerobot = np.diff(ts_arr) * 1000.0  # ms
        std_lerobot = float(np.std(dts_lerobot))
        print(f"\n【LeRobot timestamp 均匀性】std={std_lerobot:.2f}ms")
        if std_lerobot < 3:
            sync_scores.append(2)
        elif std_lerobot < 5:
            sync_scores.append(1)
        else:
            sync_scores.append(0)

    # 4. 综合评级
    score = sum(sync_scores)
    max_score = len(sync_scores) * 2
    print(f"\n【综合评级】{score}/{max_score}", end=" ")
    if score >= max_score - 1:
        print("=> 同步质量很好，可直接用于训练。")
    elif score >= max_score // 2:
        print("=> 同步基本可用，建议优化后再大规模采集。")
    else:
        print("=> 同步较差，建议检查 CAN/相机/ROS 节点延迟，或修改采集逻辑。")

    return {
        "episode_index": episode_index,
        "dt_stats": dt_stats,
        "max_diff_ms": max_diff_ms,
        "timestamp_ms": dts_lerobot if "timestamp" in df.columns else None,
    }


def plot_episode(result, fps: int, out_path: Path):
    episode_index = result["episode_index"]
    dt_stats = result["dt_stats"]
    max_diff_ms = result["max_diff_ms"]
    timestamp_ms = result["timestamp_ms"]

    n_plots = 2 + (1 if timestamp_ms is not None else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    target_dt = 1000.0 / fps
    frame_indices = np.arange(len(max_diff_ms))

    # 图1: 各模态 dt
    ax = axes[0]
    for col, dts in dt_stats.items():
        label = col.split(".")[-1]
        ax.plot(frame_indices[1:], dts, label=label, alpha=0.7)
    ax.axhline(target_dt, color="black", linestyle="--", linewidth=1, label=f"target ({target_dt:.1f}ms)")
    ax.axhline(target_dt + 15.0, color="red", linestyle="--", linewidth=1, label="timeout threshold")
    ax.set_ylabel("dt (ms)")
    ax.set_title(f"Episode {episode_index:04d} - Inter-frame interval by modality")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, max(target_dt + 15.0, 60))

    # 图2: 模态间同步差异
    ax = axes[1]
    ax.plot(frame_indices, max_diff_ms, color="purple", alpha=0.8)
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1, label="10 ms threshold")
    ax.set_ylabel("max diff (ms)")
    ax.set_xlabel("frame index")
    ax.set_title("Cross-modality sync diff within same frame")
    ax.legend()

    # 图3: LeRobot timestamp 均匀性
    if timestamp_ms is not None:
        ax = axes[2]
        ax.plot(frame_indices[1:], timestamp_ms, color="green", alpha=0.8)
        ax.set_ylabel("dt (ms)")
        ax.set_xlabel("frame index")
        ax.set_title("LeRobot timestamp interval")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"图表已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="分析 LeRobot 数据同步质量")
    parser.add_argument("--root_path", type=str, required=True, help="LeRobot 数据根目录，例如 ./All_datas")
    parser.add_argument("--repo_id", type=str, required=True, help="数据集名称，例如 pickXtimes_v21")
    parser.add_argument("--episodes", type=str, default="latest2", help="要分析的 episode，例如 '0,1' 或 'latest2'")
    args = parser.parse_args()

    root = Path(args.root_path) / args.repo_id
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        print(f"找不到 {info_path}，请检查 root_path 和 repo_id")
        sys.exit(1)

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    fps = info.get("fps", 30)
    total_episodes = info.get("total_episodes", 0)

    # 解析 episode 列表
    if args.episodes.lower().startswith("latest"):
        n = int(args.episodes.replace("latest", ""))
        episode_indices = list(range(max(0, total_episodes - n), total_episodes))
    else:
        episode_indices = [int(x.strip()) for x in args.episodes.split(",")]

    out_dir = root / "sync_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in episode_indices:
        df = load_episode_parquet(root, ep_idx)
        if df is None:
            print(f"Episode {ep_idx:04d} 不存在，跳过。")
            continue
        result = analyze_episode(df, fps, ep_idx)
        if result:
            plot_episode(result, fps, out_dir / f"episode_{ep_idx:04d}_sync.png")


if __name__ == "__main__":
    main()
