"""
verify_gripper_actions.py
--------------------------
Loads a collected LeRobot dataset episode and plots the comparison between:
  - observation.state  (follower arm's actual gripper position)
  - action             (leader arm's commanded gripper position)

For left gripper (index 6) and right gripper (index 13).

When grasping a hard / rigid object, the follower will be physically resisted
and its actual state will DIVERGE from the commanded action. A clear gap in the
plot confirms that:
  1. The bug fix is working (action = leader command, not follower qpos), and
  2. The dataset correctly captures the command vs. reality discrepancy.

Usage:
    python verify_gripper_actions.py --root_path ./All_datas/test --repo_id my_dataset
    python verify_gripper_actions.py --root_path ./All_datas/test --repo_id my_dataset --episode 3
    python verify_gripper_actions.py --root_path ./All_datas/test --repo_id my_dataset --episode 3 --save
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── index constants (must match motor list in collect_ledatav21.py) ──────────
LEFT_GRIPPER_IDX  = 6
RIGHT_GRIPPER_IDX = 13


def load_episode_parquet(root_path: Path, repo_id: str, episode: int):
    """
    Load observation.state and action columns for one episode from the
    LeRobot parquet files stored under:
        <root_path>/<repo_id>/data/chunk-*/episode_<XXXXXX>.parquet
    Falls back to a single merged parquet if per-episode files are absent.
    """
    import pandas as pd

    base = root_path / repo_id
    # LeRobot v2.1 layout: data/chunk-000/episode_000000.parquet
    # print(f"------------> base: {base}")
    pattern = f"episode_{episode:06d}.parquet"
    matches = sorted(base.rglob(pattern))

    if not matches:
        # Try flat layout (older versions)
        flat = base / "data" / f"episode_{episode:06d}.parquet"
        if flat.exists():
            matches = [flat]

    if not matches:
        raise FileNotFoundError(
            f"No parquet file found for episode {episode} under {base}.\n"
            f"Searched for: **/{pattern}"
        )

    df = pd.read_parquet(matches[0])

    required = {"observation.state", "action"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Parquet file is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Columns may be stored as lists/arrays per row — convert to 2-D arrays
    def to_array(col):
        first = col.iloc[0]
        if hasattr(first, '__len__'):
            return np.array(col.tolist(), dtype=np.float32)
        return col.values.reshape(-1, 1).astype(np.float32)

    state  = to_array(df["observation.state"])   # (T, 14)
    action = to_array(df["action"])              # (T, 14)

    return state, action


def load_episode_hdf5(root_path: Path, repo_id: str, episode: int):
    """
    Fallback loader for HDF5 layout (some older collect scripts save .hdf5).
    """
    import h5py

    base = root_path / repo_id
    candidates = sorted(base.rglob(f"episode_{episode:04d}.hdf5")) + \
                 sorted(base.rglob(f"episode_{episode:06d}.hdf5"))

    if not candidates:
        raise FileNotFoundError(f"No HDF5 file found for episode {episode} under {base}.")

    with h5py.File(candidates[0], "r") as f:
        state  = f["observation.state"][:]   # (T, 14)
        action = f["action"][:]              # (T, 14)

    return state.astype(np.float32), action.astype(np.float32)


def load_episode(root_path: Path, repo_id: str, episode: int):
    """Try parquet first, then HDF5."""
    try:
        return load_episode_parquet(root_path, repo_id, episode)
    except (FileNotFoundError, ImportError):
        pass
    return load_episode_hdf5(root_path, repo_id, episode)


def compute_stats(state_grip, action_grip):
    diff = action_grip - state_grip
    return {
        "max_divergence"  : float(np.max(np.abs(diff))),
        "mean_divergence" : float(np.mean(np.abs(diff))),
        "frames_diverged" : int(np.sum(np.abs(diff) > 0.05)),   # > ~3°
        "total_frames"    : len(diff),
    }


def plot_gripper_comparison(state, action, episode: int, save_path=None):
    T       = state.shape[0]
    frames  = np.arange(T)

    left_state   = state[:, LEFT_GRIPPER_IDX]
    left_action  = action[:, LEFT_GRIPPER_IDX]
    right_state  = state[:, RIGHT_GRIPPER_IDX]
    right_action = action[:, RIGHT_GRIPPER_IDX]

    left_stats  = compute_stats(left_state,  left_action)
    right_stats = compute_stats(right_state, right_action)

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Gripper: Actual State vs. Commanded Action  |  Episode {episode}",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax_left_main  = fig.add_subplot(gs[0, 0])
    ax_right_main = fig.add_subplot(gs[0, 1])
    ax_left_diff  = fig.add_subplot(gs[1, 0])
    ax_right_diff = fig.add_subplot(gs[1, 1])
    ax_left_hist  = fig.add_subplot(gs[2, 0])
    ax_right_hist = fig.add_subplot(gs[2, 1])

    COLOR_STATE  = "#2196F3"   # blue  — follower actual
    COLOR_ACTION = "#F44336"   # red   — leader commanded
    COLOR_DIFF   = "#FF9800"   # orange — divergence

    # ── helper: main overlay plot ─────────────────────────────────────────────
    def plot_main(ax, state_g, action_g, title, stats):
        ax.plot(frames, state_g,  color=COLOR_STATE,  lw=1.5, label="Actual state (follower)")
        ax.plot(frames, action_g, color=COLOR_ACTION, lw=1.5, linestyle="--", label="Commanded action (leader)")
        ax.fill_between(frames, state_g, action_g,
                        alpha=0.15, color=COLOR_DIFF, label="Divergence gap")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Gripper position (rad)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        # annotation box
        info = (f"Max divergence : {stats['max_divergence']:.4f} rad\n"
                f"Mean divergence: {stats['mean_divergence']:.4f} rad\n"
                f"Frames diverged: {stats['frames_diverged']}/{stats['total_frames']}")
        ax.text(0.02, 0.04, info, transform=ax.transAxes, fontsize=7.5,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.85))

    # ── helper: difference plot ───────────────────────────────────────────────
    def plot_diff(ax, state_g, action_g, title):
        diff = action_g - state_g
        ax.plot(frames, diff, color=COLOR_DIFF, lw=1.2)
        ax.axhline(0, color="black", lw=0.8, linestyle=":")
        ax.fill_between(frames, diff, 0,
                        where=(np.abs(diff) > 0.05),
                        color=COLOR_DIFF, alpha=0.3,
                        label="|divergence| > 0.05 rad")
        ax.set_title(f"{title} — Difference (action − state)", fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Δ position (rad)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── helper: histogram ────────────────────────────────────────────────────
    def plot_hist(ax, state_g, action_g, title):
        diff = action_g - state_g
        ax.hist(diff, bins=40, color=COLOR_DIFF, edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", lw=1.2, linestyle="--")
        ax.set_title(f"{title} — Divergence distribution", fontsize=10)
        ax.set_xlabel("action − state (rad)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # ── draw all panels ───────────────────────────────────────────────────────
    plot_main(ax_left_main,  left_state,  left_action,  "LEFT Gripper",  left_stats)
    plot_main(ax_right_main, right_state, right_action, "RIGHT Gripper", right_stats)
    plot_diff(ax_left_diff,  left_state,  left_action,  "LEFT Gripper")
    plot_diff(ax_right_diff, right_state, right_action, "RIGHT Gripper")
    plot_hist(ax_left_hist,  left_state,  left_action,  "LEFT Gripper")
    plot_hist(ax_right_hist, right_state, right_action, "RIGHT Gripper")

    # ── console summary ───────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  Episode {episode} — Gripper Divergence Summary")
    print("="*55)
    for side, stats in [("LEFT ", left_stats), ("RIGHT", right_stats)]:
        print(f"  {side}  max divergence : {stats['max_divergence']:.4f} rad")
        print(f"         mean divergence: {stats['mean_divergence']:.4f} rad")
        print(f"         frames diverged: {stats['frames_diverged']} / {stats['total_frames']}")
        print()

    divergence_detected = (left_stats["max_divergence"]  > 0.05 or
                           right_stats["max_divergence"] > 0.05)
    if divergence_detected:
        print("  ✅ Divergence detected — bug fix is working correctly.")
        print("     The dataset captures leader commands, not follower qpos.")
    else:
        print("  ⚠️  No significant divergence detected.")
        print("     Either the task had no hard contacts, or action = state still.")
    print("="*55 + "\n")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Verify gripper state vs commanded action divergence in a collected episode."
    )
    parser.add_argument("--root_path", type=Path, required=True,
                        help="Dataset root path (e.g. ./All_datas/test323_old)")
    parser.add_argument("--repo_id",   type=str,  required=True,
                        help="Dataset repo id (subfolder name)")
    parser.add_argument("--episode",   type=int,  default=0,
                        help="Episode index to inspect (default: 0)")
    parser.add_argument("--save",      action="store_true",
                        help="Save plot to file instead of displaying it")
    args = parser.parse_args()

    print(f"\nLoading episode {args.episode} from {args.root_path / args.repo_id} ...")
    try:
        state, action = load_episode(args.root_path, args.repo_id, args.episode)
    except (FileNotFoundError, KeyError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print(f"  Loaded {state.shape[0]} frames, {state.shape[1]} joints.")

    save_path = None
    if args.save:
        out_dir = args.root_path / args.repo_id
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"gripper_verify_ep{args.episode:04d}.png"

    plot_gripper_comparison(state, action, args.episode, save_path=save_path)


if __name__ == "__main__":
    main()


''' ziqi 3.23 add:
python ./src/edlsrobot/datasets/verify_gripper_actions_ziqi.py --root_path ./All_datas/test323_old --repo_id '' --episode 0
'''

'''
python ./src/edlsrobot/datasets/verify_gripper_actions_ziqi.py --root_path ./All_datas/pour_tea400 --repo_id '' --episode 0
'''
