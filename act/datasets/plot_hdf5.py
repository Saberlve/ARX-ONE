"""
Plot /action vs /observations/qpos from a teleoperation HDF5 file.
Usage: python3 plot_action_vs_qpos.py <path_to_episode.hdf5>
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
HDF5_PATH = sys.argv[1] if len(sys.argv) > 1 else "episode_2.hdf5"
JOINT_NAMES = None   # Set to a list of strings to override auto-generated names
                     # e.g. ["L_shoulder_pan", "L_shoulder_lift", ..., "R_wrist_roll"]
# ──────────────────────────────────────────────────────────────────────────────


def load_data(path: str):
    with h5py.File(path, "r") as f:
        action = np.array(f["/action"])             # (T, D)
        qpos   = np.array(f["/observations/qpos"])  # (T, D)
    assert action.shape == qpos.shape, (
        f"Shape mismatch: action={action.shape}, qpos={qpos.shape}"
    )
    return action, qpos


def make_joint_names(n_joints: int):
    """Auto-generate joint names for a dual-arm robot (14 = 7+7, 12 = 6+6, etc.)"""
    half = n_joints // 2
    left  = [f"L_joint_{i+1}" for i in range(half)]
    right = [f"R_joint_{i+1}" for i in range(n_joints - half)]
    return left + right


def plot(action: np.ndarray, qpos: np.ndarray, joint_names: list, save_path: str):
    T, D = action.shape
    timesteps = np.arange(T)

    ncols = 2
    nrows = (D + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 7, nrows * 3), constrained_layout=True)
    fig.suptitle("Action vs Joint Position (qpos) — per joint", fontsize=15, fontweight="bold")

    gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for i in range(D):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.plot(timesteps, qpos[:, i],   label="qpos (obs)",   color="#2196F3", linewidth=1.4)
        ax.plot(timesteps, action[:, i], label="action (cmd)", color="#FF5722",
                linewidth=1.4, linestyle="--", alpha=0.85)
        ax.set_title(joint_names[i], fontsize=10)
        ax.set_xlabel("Timestep", fontsize=8)
        ax.set_ylabel("Value (rad)", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Hide any unused subplot slots
    for i in range(D, nrows * ncols):
        fig.add_subplot(gs[i // ncols, i % ncols]).set_visible(False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {save_path}")
    plt.show()


def main():
    path = HDF5_PATH
    print(f"Loading: {path}")
    action, qpos = load_data(path)
    T, D = action.shape
    print(f"  Timesteps : {T}")
    print(f"  Joints    : {D}")

    names = JOINT_NAMES or make_joint_names(D)
    if len(names) != D:
        raise ValueError(f"JOINT_NAMES has {len(names)} entries but data has {D} joints")

    out = Path(path).stem + "_action_vs_qpos.png"
    plot(action, qpos, names, out)


if __name__ == "__main__":
    main()