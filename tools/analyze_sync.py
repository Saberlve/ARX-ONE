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
    """根据 info.json 里的 data_path 格式找到 episode 对应的 parquet 文件。"""
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
    # 这 5 列是 collect_ledatav21.py 在 create_empty_dataset() 里注册并在 add_frame() 时写入的原始 ROS 时间戳
    ts_cols = [
        "sync_timestamp.head_ns",
        "sync_timestamp.left_wrist_ns",
        "sync_timestamp.right_wrist_ns",
        "sync_timestamp.left_arm_ns",
        "sync_timestamp.right_arm_ns",
    ]

    # 兼容旧数据：如果没有 sync_timestamp 列，说明是用旧版采集代码生成的，跳过分析
    available_cols = [c for c in ts_cols if c in df.columns]
    if not available_cols:
        print(f"Episode {episode_index}: 找不到 sync_timestamp 列，请确认是用新版采集代码保存的。")
        return None

    total_frames = len(df)
    target_dt = 1000.0 / fps  # 例如 30fps 时 target_dt = 33.33 ms

    print(f"\n{'='*60}")
    print(f"Episode {episode_index:04d} | 总帧数: {total_frames} | 目标 fps: {fps}")
    print(f"{'='*60}")

    # PART 1: 各模态的帧间隔稳定性
    # 原理：对 sync_timestamp.xxx_ns 这一列做 np.diff()，得到该模态在相邻两帧之间的时间间隔。
    #       如果 std 很小，说明该模态的 ROS 回调很规律；如果 std 很大，说明采集主循环有卡顿或跳帧。
    print("\n【帧率稳定性】")
    sync_scores = []
    dt_stats = {}
    for col in available_cols:
        arr = df[col].to_numpy().astype(np.int64).flatten()
        dts = np.diff(arr).astype(np.float64) / 1e6  # 纳秒 -> 毫秒
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

    # PART 2: 同一帧内各模态时间戳的最大差异（真正的同步指标）
    # 原理：把一帧的 5 个时间戳横向排列，取 max(ts) - min(ts)。
    #       这个值越大，说明 "被保存为同一帧" 的 head 图、wrist 图、arm 状态实际上来自 ROS 里时间跨度越大的窗口。
    #       对 VLA 模型来说，>10ms 就可能导致图和状态对不上号。
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

    # PART 2.5: action 与 state 之间的时间间隔
    # 原理：collect_ledatav21.py 保存的是 (state_t, action_t+1)。
    #       parquet 第 k 行的 sync_timestamp.left_arm_ns 是 state_t 的时间戳 ts_k；
    #       而第 k 行的 action 实际上对应第 k+1 行的 ts_{k+1}。
    #       因此 action-state 间隔 = np.diff(left_arm_ns)。
    #       理想情况下这个间隔应严格等于 target_dt（如 33.3ms），
    #       若 std 很大或平均值偏离 target_dt，说明 action 并不是真正的 "t+1 时刻"。
    action_state_dt_ms = None
    if "sync_timestamp.left_arm_ns" in available_cols:
        arm_ts = df["sync_timestamp.left_arm_ns"].to_numpy().astype(np.int64).flatten()
        action_state_dt_ms = np.diff(arm_ts).astype(np.float64) / 1e6
        avg_as_dt = float(np.mean(action_state_dt_ms))
        std_as_dt = float(np.std(action_state_dt_ms))
        max_as_dt = float(np.max(action_state_dt_ms))
        min_as_dt = float(np.min(action_state_dt_ms))
        short_frames = int(np.sum(action_state_dt_ms < target_dt - 5.0))  # 明显小于 target_dt
        print("\n【action 与 state 的时间间隔】")
        print(f"  平均值={avg_as_dt:.2f}ms std={std_as_dt:.2f}ms 最小={min_as_dt:.1f}ms 最大={max_as_dt:.1f}ms")
        print(f"  间隔明显小于 {target_dt:.1f}ms 的帧: {short_frames}/{len(action_state_dt_ms)}")
        if std_as_dt < 3 and abs(avg_as_dt - target_dt) < 3:
            sync_scores.append(2)
        elif std_as_dt < 5 and abs(avg_as_dt - target_dt) < 5:
            sync_scores.append(1)
        else:
            sync_scores.append(0)

    # PART 3: LeRobot 自带的 timestamp 列均匀性
    # 注意：这个 timestamp 在旧版代码里是 frame_index / fps 的固定值，std 为 0；
    #       在新版 Lerobot 里可能更真实，但这里只作为一个辅助指标。
    dts_lerobot = None
    if "timestamp" in df.columns:
        ts_arr = df["timestamp"].to_numpy().astype(np.float64)
        dts_lerobot = np.diff(ts_arr) * 1000.0  # s -> ms
        std_lerobot = float(np.std(dts_lerobot))
        print(f"\n【LeRobot timestamp 均匀性】std={std_lerobot:.2f}ms")
        if std_lerobot < 3:
            sync_scores.append(2)
        elif std_lerobot < 5:
            sync_scores.append(1)
        else:
            sync_scores.append(0)

    # PART 4: 综合评级
    # 每项指标满分 2 分（优秀），1 分（可用），0 分（较差）。
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
        "action_state_dt_ms": action_state_dt_ms,
        "timestamp_ms": dts_lerobot,
    }


def plot_episode(result, fps: int, out_path: Path):
    """绘制 4 张子图，直观展示同步质量。"""
    episode_index = result["episode_index"]
    dt_stats = result["dt_stats"]
    max_diff_ms = result["max_diff_ms"]
    action_state_dt_ms = result["action_state_dt_ms"]
    timestamp_ms = result["timestamp_ms"]

    n_plots = 3 + (1 if timestamp_ms is not None else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    target_dt = 1000.0 / fps
    # max_diff_ms 长度等于总帧数；dt_stats 里的数组长度是 total_frames-1（因为做了 diff）
    frame_indices = np.arange(len(max_diff_ms))
    ax_idx = 0

    # SUBPLOT 1: 各模态的 dt 时序曲线
    # 解读：5 条彩色线代表 5 个模态。如果它们紧紧贴在一起，说明各自队列的消费节奏一致；
    #       如果某条线经常“冒尖”超过红线，说明该模态有跳帧。
    ax = axes[ax_idx]
    ax_idx += 1
    for col, dts in dt_stats.items():
        label = col.split(".")[-1]
        ax.plot(frame_indices[1:], dts, label=label, alpha=0.7)
    ax.axhline(target_dt, color="black", linestyle="--", linewidth=1, label=f"target ({target_dt:.1f}ms)")
    ax.axhline(target_dt + 15.0, color="red", linestyle="--", linewidth=1, label="timeout threshold")
    ax.set_ylabel("dt (ms)")
    ax.set_title(f"Episode {episode_index:04d} - Inter-frame interval by modality")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, max(target_dt + 15.0, 60))

    # SUBPLOT 2: 同一帧内的模态间最大时间差（最关键的图）
    # 解读：紫色线越贴近 0 越好。红线是 10ms，超过红线意味着这帧的图和机械臂状态不是同一时刻的快照。
    ax = axes[ax_idx]
    ax_idx += 1
    ax.plot(frame_indices, max_diff_ms, color="purple", alpha=0.8)
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1, label="10 ms threshold")
    ax.set_ylabel("max diff (ms)")
    ax.set_title("Cross-modality sync diff within same frame")
    ax.legend()

    # SUBPLOT 3: action 与 state 的时间间隔
    # 解读：黄色线应该紧贴黑线（target_dt）。如果黄色线忽高忽低或整体偏离黑线，
    #       说明 action 并不是严格来自 "t+1 时刻"，而是被 pop() 机制随机拉近了/拉远了。
    if action_state_dt_ms is not None:
        ax = axes[ax_idx]
        ax_idx += 1
        ax.plot(frame_indices[1:], action_state_dt_ms, color="orange", alpha=0.8, label="action-state dt")
        ax.axhline(target_dt, color="black", linestyle="--", linewidth=1, label=f"target ({target_dt:.1f}ms)")
        ax.set_ylabel("dt (ms)")
        ax.set_title("Action-to-prev-state interval (should be ~target dt)")
        ax.legend()

    # SUBPLOT 4: LeRobot 自带 timestamp 的间隔
    if timestamp_ms is not None:
        ax = axes[ax_idx]
        ax_idx += 1
        ax.plot(frame_indices[1:], timestamp_ms, color="green", alpha=0.8)
        ax.set_ylabel("dt (ms)")
        ax.set_xlabel("frame index")
        ax.set_title("LeRobot timestamp interval")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"图表已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="分析 LeRobot 数据同步质量")
    parser.add_argument("dataset_root", type=str, help="数据集根目录，例如 /path/to/All_datas/pickXtimes_v21")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        print(f"找不到 {info_path}，请检查传入的路径是否正确")
        sys.exit(1)

    # 从 meta/info.json 读取 fps 和 total_episodes，然后遍历 0..total_episodes-1
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    fps = info.get("fps", 30)
    total_episodes = info.get("total_episodes", 0)

    if total_episodes == 0:
        print("该数据集没有 episode")
        sys.exit(0)

    out_dir = root / "sync_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"发现 {total_episodes} 个 episode，开始分析...")
    for ep_idx in range(total_episodes):
        df = load_episode_parquet(root, ep_idx)
        if df is None:
            print(f"Episode {ep_idx:04d} 不存在，跳过。")
            continue
        result = analyze_episode(df, fps, ep_idx)
        if result:
            plot_episode(result, fps, out_dir / f"episode_{ep_idx:04d}_sync.png")


if __name__ == "__main__":
    main()
