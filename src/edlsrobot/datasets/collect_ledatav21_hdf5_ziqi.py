# -- coding: UTF-8
"""
collect_ziqi.py
===============
Dual-format teleoperation data collection for the ARX dual-arm robot.
Saves both LeRobot v2.1 (video) and HDF5 (ACT-compatible) formats in parallel.

Key fixes over collect_ledatav21.py
────────────────────────────────────
1. observation.state is always the RAW (unclipped) qpos.
   The original code did `prev_state = curr_state` where curr_state was already
   clipped, so the stored observation was wrong.  Here we track `prev_raw_state`
   and `prev_raw_eef` (unclipped) separately from the clipped action values.

2. Both action spaces are saved correctly:
     action      (joint-space) = gripper-clipped qpos    of frame t+1
     action_eef  (EEF-space)   = gripper-clipped eef     of frame t+1
     observation.state         = raw (unclipped) qpos    of frame t
     observation.eef           = raw (unclipped) eef     of frame t

3. HDF5 files are written alongside the LeRobot dataset so ACT / π0-style
   loaders can consume the data without any post-conversion step.

Inherited from collect_ledatav21.py (unchanged)
───────────────────────────────────────────────
- 数据跳帧 detection (per-camera and per-arm timestamp delta > 50 ms → discard)
- Camera/action-queue synchronisation timeout guard (10 retries)
- Joystick button-2 mid-episode discard
- LeRobot v2.1 dataset creation / resume logic
- Time-record JSONL logging per episode
"""

import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE  = Path(__file__).resolve()
ROOT  = FILE.parents[3]   # repo root:  .../ROS2_AC-one_Play/
ROOT1 = FILE.parents[2]   # src root:   .../ROS2_AC-one_Play/src/
for _p in [str(ROOT), str(ROOT1)]:
    if _p not in sys.path:
        sys.path.append(_p)
os.chdir(str(ROOT))

import time
import json
import shutil
import threading
import argparse
import dataclasses
from copy import deepcopy
from typing import Literal, Optional

import cv2
import h5py
import numpy as np
import pyttsx3
import rclpy
import yaml

from edlsrobot.datasets.lerobot_v21.lerobot_dataset import LeRobotDataset
from act.utils.ros_operator import Rate, RosOperator
from act.utils.setup_loader import setup_loader

np.set_printoptions(linewidth=200)

# ── Voice engine ─────────────────────────────────────────────────────────────
voice_engine = pyttsx3.init()
voice_engine.setProperty('voice', 'en')
voice_engine.setProperty('rate', 120)
voice_lock = threading.Lock()

# ── Global episode state ──────────────────────────────────────────────────────
joy_key_3 = False   # True → discard current episode and retry
init_pos   = None   # arm pose at episode start, used for early-stop detection


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {yaml_file}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML - {e}")
        return None


def voice_process(engine, line):
    with voice_lock:
        engine.say(line)
        engine.runAndWait()
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# LeRobot v2.1 dataset creation
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos:                      bool          = True
    tolerance_s:                     float         = 0.0001
    image_writer_processes:          int           = 3
    image_writer_threads_per_camera: int           = 4
    video_backend:                   Optional[str] = None
    video_encoding_batch_size:       int           = 1

DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    args,
    mode:           Literal["video", "image"] = "video",
    dataset_config: DatasetConfig             = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """Build the LeRobot feature schema and create (or resume) the dataset."""

    motors = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
        "left_wrist_angle", "left_wrist_rotate", "left_gripper",
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
        "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]
    eef_names = [
        "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper",
        "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper",
    ]

    features = {
        # ── Observations (always RAW, never clipped) ──────────────────────
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": list(motors),
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": list(motors),
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": list(motors),
        },
        # ── Joint-space action (gripper-clipped, next frame) ──────────────
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": list(motors),
        },
    }

    if args.use_eef:
        # Raw EEF observation at time t
        features["observation.eef"] = {
            "dtype": "float32",
            "shape": (14,),
            "names": eef_names,
        }
        # Clipped EEF action (= commanded pose at t+1)
        features["action_eef"] = {
            "dtype": "float32",
            "shape": (14,),
            "names": eef_names,
        }

    if args.use_base:
        features["action_base"] = {
            "dtype": "float32",
            "shape": (13,),
            "names": "base",
        }
        features["action_velocity"] = {
            "dtype": "float32",
            "shape": (4,),
            "names": "base_vel",
        }

    cameras = args.camera_names
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    if args.resume:
        dataset = LeRobotDataset(args.repo_id, root=args.root_path)
        if cameras:
            dataset.start_image_writer(
                num_processes=dataset_config.image_writer_processes,
                num_threads=dataset_config.image_writer_threads_per_camera * len(cameras),
            )
    else:
        root_local = Path(args.root_path) / args.repo_id
        if root_local.exists():
            shutil.rmtree(root_local)
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.frame_rate,
            root=args.root_path,
            robot_type=args.robot,
            features=features,
            use_videos=dataset_config.use_videos,
            tolerance_s=dataset_config.tolerance_s,
            image_writer_processes=dataset_config.image_writer_processes,
            image_writer_threads=dataset_config.image_writer_threads_per_camera * len(cameras),
            video_backend=dataset_config.video_backend,
        )

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Wait-for-trigger (joystick button-0 to start)
# ─────────────────────────────────────────────────────────────────────────────

def collect_detect(args, start_episode, voice_engine, ros_operator):
    """Block until the operator presses the start button (or presses a key)."""
    global init_pos, joy_key_3

    rate = Rate(args.frame_rate)
    print(f"Preparing to record episode {start_episode}")

    for i in range(1, -1, -1):
        print(f"\rwaiting {i} to start recording", end='')
        voice_process(voice_engine, f"{i + 1}")

    print(f"\nStart recording program...")

    if args.key_collect:
        input("Enter any key to record: ")
    else:
        init_done = False
        while not init_done and rclpy.ok():
            obs_dict = ros_operator.get_observation()
            if obs_dict is None:
                print("synchronization frame")
                rate.sleep()
                continue

            with ros_operator.joy_lock:
                triggered = dict(ros_operator.triggered_joys)
                ros_operator.triggered_joys.clear()

            if 0 in triggered:
                init_done  = True
                init_pos   = obs_dict['qpos']   # raw qpos at start
            if 2 in triggered:
                pass    # delete is handled inside collect_and_save

            if init_done:
                voice_process(voice_engine, f"{start_episode % 100}")
            rate.sleep()

        time.sleep(1)
        voice_process(voice_engine, "go")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 save helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compress_and_pad_images(data_dict, camera_names, use_depth, quality=50):
    """JPEG-compress and zero-pad images in-place; return padded sizes."""

    def _process(key_prefix):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        all_sizes = []
        for cam in camera_names:
            key = f'/observations/{key_prefix}/{cam}'
            encoded = []
            for img in data_dict[key]:
                _, enc = cv2.imencode('.jpg', img, encode_param)
                encoded.append(enc)
                all_sizes.append(len(enc))
            data_dict[key] = encoded
        padded = max(all_sizes)
        for cam in camera_names:
            key = f'/observations/{key_prefix}/{cam}'
            data_dict[key] = [
                np.pad(enc, (0, padded - len(enc)), constant_values=0)
                for enc in data_dict[key]
            ]
        return padded

    padded_rgb   = _process('images')
    padded_depth = _process('images_depth') if use_depth else 0
    return padded_rgb, padded_depth


def _save_hdf5(args, hdf5_buf, dataset_path_stem):
    """Write one episode buffer to <dataset_path_stem>.hdf5 (called in a thread)."""
    data_size = len(hdf5_buf['/action'])
    if data_size == 0:
        print("HDF5: empty episode buffer — skipping.")
        return

    padded_rgb, padded_depth = _compress_and_pad_images(
        hdf5_buf, args.camera_names, args.use_depth_image
    )

    STATE_DIM = 14
    EEF_DIM   = 14

    t0 = time.time()
    os.makedirs(os.path.dirname(dataset_path_stem), exist_ok=True)

    with h5py.File(dataset_path_stem + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim']  = False
        root.attrs['task'] = str(args.task)

        obs  = root.create_group('observations')
        imgs = obs.create_group('images')
        if args.use_depth_image:
            deps = obs.create_group('images_depth')

        # Image datasets
        for cam in args.camera_names:
            imgs.create_dataset(cam, (data_size, padded_rgb),   'uint8', chunks=(1, padded_rgb))
            if args.use_depth_image:
                deps.create_dataset(cam, (data_size, padded_depth), 'uint8', chunks=(1, padded_depth))

        # Scalar observation datasets
        obs.create_dataset('qpos',          (data_size, STATE_DIM))
        obs.create_dataset('qvel',          (data_size, STATE_DIM))
        obs.create_dataset('effort',        (data_size, STATE_DIM))
        obs.create_dataset('eef',           (data_size, EEF_DIM))
        obs.create_dataset('robot_base',    (data_size, 6))
        obs.create_dataset('base_velocity', (data_size, 4))

        # Action datasets
        root.create_dataset('action',          (data_size, STATE_DIM))
        root.create_dataset('action_eef',      (data_size, EEF_DIM))
        root.create_dataset('action_base',     (data_size, 6))
        root.create_dataset('action_velocity', (data_size, 4))

        # Write
        obs['qpos'][...]          = np.array(hdf5_buf['/observations/qpos'])
        obs['qvel'][...]          = np.array(hdf5_buf['/observations/qvel'])
        obs['effort'][...]        = np.array(hdf5_buf['/observations/effort'])
        obs['eef'][...]           = np.array(hdf5_buf['/observations/eef'])
        obs['robot_base'][...]    = np.array(hdf5_buf['/observations/robot_base'])
        obs['base_velocity'][...] = np.array(hdf5_buf['/observations/base_velocity'])
        root['action'][...]          = np.array(hdf5_buf['/action'])
        root['action_eef'][...]      = np.array(hdf5_buf['/action_eef'])
        root['action_base'][...]     = np.array(hdf5_buf['/action_base'])
        root['action_velocity'][...] = np.array(hdf5_buf['/action_velocity'])

        for cam in args.camera_names:
            imgs[cam][...] = np.array(hdf5_buf[f'/observations/images/{cam}'])
            if args.use_depth_image:
                deps[cam][...] = np.array(hdf5_buf[f'/observations/images_depth/{cam}'])

    print(f"\033[32mHDF5 saved in {time.time() - t0:.1f}s → {dataset_path_stem}.hdf5\033[0m")


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop (one episode)
# ─────────────────────────────────────────────────────────────────────────────

def collect_and_save(args, dataset, ros_operator, voice_engine, episode_num, hdf5_dir):
    """
    Collect one episode and save it in both LeRobot v2.1 and HDF5 formats.

    Action / observation convention
    ────────────────────────────────
    At every timestep t we write the pair:

        observation.*   ←  raw sensor readings at time t   (NEVER clipped)
        action          ←  gripper-clipped qpos  at time t+1  (joint space)
        action_eef      ←  gripper-clipped eef   at time t+1  (EEF space)

    Gripper clipping rationale: the gripper has no force feedback, so the
    raw qpos can drift above the open threshold.  We clip it to 0 to match
    what was actually commanded.  This clipping applies ONLY to the action
    fields — the observation.state / observation.eef fields always store the
    true sensor reading.
    """
    global joy_key_3, init_pos

    rate          = Rate(args.frame_rate)
    GRIPPER_IDX   = [6, 13]        # positions in the 14-dim vectors
    GRIPPER_CLOSE = -1.3           # above this → treat as open (command = 0)

    # ── 数据跳帧 tracking ─────────────────────────────────────────────────────
    last_head_ts = last_left_ts = last_right_ts = None
    last_left_arm_ts = last_right_arm_ts = None
    time_records = []

    # ── "Previous frame" buffers — always RAW, never clipped ─────────────────
    prev_raw_state   = None   # raw qpos  at t
    prev_raw_eef     = None   # raw eef   at t
    prev_vel         = None   # qvel      at t
    prev_effort      = None   # effort    at t
    prev_image_head  = None
    prev_image_left  = None
    prev_image_right = None

    # ── HDF5 per-episode accumulator ──────────────────────────────────────────
    hdf5_buf = {
        '/observations/qpos':          [],
        '/observations/qvel':          [],
        '/observations/effort':        [],
        '/observations/eef':           [],
        '/observations/robot_base':    [],
        '/observations/base_velocity': [],
        '/action':                     [],
        '/action_eef':                 [],
        '/action_base':                [],
        '/action_velocity':            [],
    }
    for cam in args.camera_names:
        hdf5_buf[f'/observations/images/{cam}'] = []
        if args.use_depth_image:
            hdf5_buf[f'/observations/images_depth/{cam}'] = []

    count          = 0
    sync_fail_cnt  = 0
    _ZERO_14       = np.zeros(14, dtype=np.float32)
    _ZERO_6        = np.zeros(6,  dtype=np.float32)
    _ZERO_4        = np.zeros(4,  dtype=np.float32)

    while count < args.max_timesteps and rclpy.ok() and not joy_key_3:
        loop_t0 = time.perf_counter()

        # ── Joystick mid-episode delete ───────────────────────────────────
        with ros_operator.joy_lock:
            triggered = dict(ros_operator.triggered_joys)
            ros_operator.triggered_joys.clear()
        if 2 in triggered:
            joy_key_3 = True
            voice_process(voice_engine, f"delete {episode_num}")
            break

        # ── Get observation ───────────────────────────────────────────────
        t0       = time.perf_counter()
        obs_dict = ros_operator.get_observation(ts=count)
        t1       = time.perf_counter()
        act_dict = ros_operator.get_action()

        if obs_dict is None or act_dict is None:
            print("Synchronization frame")
            rate.sleep()
            sync_fail_cnt += 1
            if sync_fail_cnt > 10:
                print("Camera/action queue timeout — discarding episode.")
                joy_key_3 = True
                if dataset is not None:
                    dataset.clear_episode_buffer()
                break
            continue
        sync_fail_cnt = 0

        # ── 数据跳帧: per-source timestamp delta check ────────────────────
        curr_head_ts      = obs_dict["img_ts"]["head"]
        curr_left_ts      = obs_dict["img_ts"]["left_wrist"]
        curr_right_ts     = obs_dict["img_ts"]["right_wrist"]
        curr_left_arm_ts  = obs_dict["arm_ts"]["left_arm"]
        curr_right_arm_ts = obs_dict["arm_ts"]["right_arm"]

        head_dt_ms = left_dt_ms = right_dt_ms = left_arm_dt_ms = right_arm_dt_ms = None
        if last_head_ts      is not None: head_dt_ms      = (curr_head_ts      - last_head_ts)      / 1e6
        if last_left_ts      is not None: left_dt_ms      = (curr_left_ts      - last_left_ts)      / 1e6
        if last_right_ts     is not None: right_dt_ms     = (curr_right_ts     - last_right_ts)     / 1e6
        if last_left_arm_ts  is not None: left_arm_dt_ms  = (curr_left_arm_ts  - last_left_arm_ts)  / 1e6
        if last_right_arm_ts is not None: right_arm_dt_ms = (curr_right_arm_ts - last_right_arm_ts) / 1e6

        last_head_ts = curr_head_ts;  last_left_ts = curr_left_ts;  last_right_ts = curr_right_ts
        last_left_arm_ts = curr_left_arm_ts;  last_right_arm_ts = curr_right_arm_ts

        time_records.append({
            "frame_index":     int(count),
            "head_ts_ns":      int(curr_head_ts),
            "head_dt_ms":      None if head_dt_ms      is None else float(head_dt_ms),
            "left_dt_ms":      None if left_dt_ms      is None else float(left_dt_ms),
            "right_dt_ms":     None if right_dt_ms     is None else float(right_dt_ms),
            "left_arm_dt_ms":  None if left_arm_dt_ms  is None else float(left_arm_dt_ms),
            "right_arm_dt_ms": None if right_arm_dt_ms is None else float(right_arm_dt_ms),
            "get_observation_ms": float((t1 - t0) * 1000.0),
        })

        dt_list = [head_dt_ms, left_dt_ms, right_dt_ms, left_arm_dt_ms, right_arm_dt_ms]
        if any(x is not None and x > 50 for x in dt_list):
            joy_key_3 = True
            print(f"Frame-skip detected (dt > 50 ms): {dt_list}")
            voice_process(voice_engine, f"timeout, delete {episode_num}")
            break

        # ── Current RAW observations (never clipped) ──────────────────────
        curr_raw_state = deepcopy(obs_dict["qpos"].astype(np.float32))
        curr_vel       = deepcopy(obs_dict["qvel"].astype(np.float32))
        curr_effort    = deepcopy(obs_dict["effort"].astype(np.float32))
        # Fix: was obs_dict["eff"] (typo) in collect_ledatav21.py
        curr_raw_eef   = deepcopy(obs_dict["eef"].astype(np.float32)) if args.use_eef else None

        # ── Clipped versions → used ONLY as the commanded action ──────────
        # Gripper has no force feedback: clip raw reading to 0 when above
        # the open threshold so the stored action matches the real command.
        curr_action_joint = curr_raw_state.copy()
        for idx in GRIPPER_IDX:
            if curr_action_joint[idx] > GRIPPER_CLOSE:
                curr_action_joint[idx] = 0.0

        curr_action_eef = None
        if args.use_eef:
            curr_action_eef = curr_raw_eef.copy()
            for idx in GRIPPER_IDX:
                if curr_action_eef[idx] > GRIPPER_CLOSE:
                    curr_action_eef[idx] = 0.0

        # ── Early-stop: robot has returned to init pose ───────────────────
        # We check BEFORE writing so the final return-to-home frame pair
        # is still saved (prev, curr) before we break.
        should_stop = False
        if count > 100 and init_pos is not None:
            if all(abs(v - i) <= 0.1 for v, i in zip(curr_raw_state, init_pos)):
                print("Robot returned to init — episode complete.")
                should_stop = True

        # ── Write frame pair (obs_t, action_{t→t+1}) ─────────────────────
        # Only once we have a complete "previous" frame available.
        if prev_raw_state is not None:
            curr_images = {
                "head":        obs_dict["images"]["head"],
                "left_wrist":  obs_dict["images"]["left_wrist"],
                "right_wrist": obs_dict["images"]["right_wrist"],
            }
            prev_images = {
                "head":        prev_image_head,
                "left_wrist":  prev_image_left,
                "right_wrist": prev_image_right,
            }

            # ── LeRobot frame ─────────────────────────────────────────────
            frame = {
                # ── observations at time t (always RAW) ──
                "observation.state":             prev_raw_state,
                "observation.velocity":          prev_vel,
                "observation.effort":            prev_effort,
                # ── joint-space action = clipped qpos at t+1 ──
                "action":                        curr_action_joint,
                # ── camera images at time t ──
                "observation.images.head":       prev_images["head"],
                "observation.images.left_wrist": prev_images["left_wrist"],
                "observation.images.right_wrist":prev_images["right_wrist"],
                "task": args.task,
            }
            if args.use_eef:
                # raw EEF observation at t
                frame["observation.eef"] = prev_raw_eef
                # clipped EEF action = commanded pose at t+1
                # Fix: was "action_eff" (typo) in collect_ledatav21.py
                frame["action_eef"]      = curr_action_eef
            if args.use_base:
                frame["action_base"]     = deepcopy(obs_dict["action_base"].astype(np.float32))
                frame["action_velocity"] = deepcopy(obs_dict["base_velocity"].astype(np.float32))

            if dataset is not None:
                dataset.add_frame(frame)

            # ── HDF5 accumulator ──────────────────────────────────────────
            hdf5_buf['/observations/qpos'].append(prev_raw_state)
            hdf5_buf['/observations/qvel'].append(prev_vel)
            hdf5_buf['/observations/effort'].append(prev_effort)
            hdf5_buf['/observations/eef'].append(
                prev_raw_eef if args.use_eef else _ZERO_14.copy()
            )
            hdf5_buf['/observations/robot_base'].append(
                deepcopy(obs_dict["robot_base"].astype(np.float32))
                if "robot_base" in obs_dict else _ZERO_6.copy()
            )
            hdf5_buf['/observations/base_velocity'].append(
                deepcopy(obs_dict["base_velocity"].astype(np.float32))
                if "base_velocity" in obs_dict else _ZERO_4.copy()
            )
            hdf5_buf['/action'].append(curr_action_joint)
            hdf5_buf['/action_eef'].append(
                curr_action_eef if args.use_eef else _ZERO_14.copy()
            )
            hdf5_buf['/action_base'].append(
                deepcopy(obs_dict["action_base"].astype(np.float32))
                if args.use_base else _ZERO_6.copy()
            )
            hdf5_buf['/action_velocity'].append(
                deepcopy(obs_dict["base_velocity"].astype(np.float32))
                if args.use_base else _ZERO_4.copy()
            )
            for cam in args.camera_names:
                hdf5_buf[f'/observations/images/{cam}'].append(prev_images[cam])
                # depth is a pass for now

        # ── Advance "previous" pointers ───────────────────────────────────
        # Always track the RAW observations — never the clipped ones.
        prev_raw_state   = curr_raw_state
        prev_raw_eef     = curr_raw_eef
        prev_vel         = curr_vel
        prev_effort      = curr_effort
        prev_image_head  = obs_dict["images"]["head"]
        prev_image_left  = obs_dict["images"]["left_wrist"]
        prev_image_right = obs_dict["images"]["right_wrist"]

        print(f"Frame data: {count}")
        count += 1

        if should_stop:
            break

        if not rclpy.ok():
            exit(-1)

        rate.sleep()

    # ── Episode end: save or discard ──────────────────────────────────────────

    def _save_time_records(ep_num, records):
        save_dir = Path("./time_logs")
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"episode_{ep_num:04d}_time.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved time records → {path}")

    if joy_key_3:
        if dataset is not None:
            dataset._wait_image_writer()
            dataset.clear_episode_buffer()
        print(f"\033[34mDiscarded episode {episode_num}\033[0m")
    else:
        _save_time_records(episode_num, time_records)

        # ── LeRobot v2.1 ──────────────────────────────────────────────────
        if dataset is not None:
            dataset.save_episode()

        # ── HDF5 (in a background thread so we don't block the next episode)
        hdf5_stem = os.path.join(hdf5_dir, f"episode_{episode_num}")
        threading.Thread(
            target=_save_hdf5,
            args=(args, hdf5_buf, hdf5_stem),
            daemon=True,
        ).start()

        voice_process(voice_engine, f"Save {episode_num}")
        print(f"\033[32m\nSaved episode {episode_num} ✓\033[0m\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    global joy_key_3

    setup_loader(ROOT)
    rclpy.init()

    config       = load_yaml(args.config)
    ros_operator = RosOperator(args, config, in_collect=True)

    spin_running = True

    def _spin_loop(node):
        while spin_running and rclpy.ok():
            try:
                rclpy.spin_once(node, timeout_sec=0.001)
            except rclpy._rclpy_pybind11.InvalidHandle:
                break

    spin_thread = threading.Thread(target=_spin_loop, args=(ros_operator,), daemon=True)
    spin_thread.start()

    dataset = create_empty_dataset(args, mode="video")

    def _get_resume_episode_num(info_json_path):
        with open(info_json_path, "r", encoding="utf-8") as f:
            return json.load(f).get("total_episodes", 0)

    num_episodes = 1000 if args.episode_nums == -1 else args.episode_nums

    try:
        episode_num = 0
        if args.resume:
            info_path = Path(args.root_path) / args.repo_id / "meta" / "info.json"
            if info_path.exists():
                episode_num = _get_resume_episode_num(info_path)
        num_episodes = episode_num + num_episodes

        hdf5_dir = os.path.join(args.root_path, args.repo_id, "hdf5")
        os.makedirs(hdf5_dir, exist_ok=True)
        print(f"HDF5 output directory: {hdf5_dir}")

        while episode_num < num_episodes and rclpy.ok():
            print(f"\n{'─'*60}")
            print(f"  Episode {episode_num}")
            print(f"{'─'*60}")

            collect_detect(args, episode_num, voice_engine, ros_operator)
            collect_and_save(args, dataset, ros_operator, voice_engine, episode_num, hdf5_dir)

            if joy_key_3:
                # retry same episode index
                joy_key_3 = False
            else:
                episode_num += 1

        time.sleep(0.5)
        voice_process(voice_engine, "Over")

    except KeyboardInterrupt:
        print("KeyboardInterrupt — stopping.")

    finally:
        if dataset is not None:
            dataset.shutdown_pool()
            if hasattr(dataset, "stop_image_writer"):
                dataset.stop_image_writer()

        spin_running = False
        spin_thread.join(timeout=1.0)
        voice_engine.stop()
        ros_operator.destroy_node()
        rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _str2bool(v: str) -> bool:
    """
    Proper CLI boolean parser.
    Fixes the silent argparse trap where type=bool makes bool("False") == True
    because any non-empty string is truthy in Python.

    Accepts: true/false, yes/no, 1/0, on/off  (case-insensitive)
    Usage:   --resume false   --use_eef true
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('true',  'yes', '1', 'on'):
        return True
    if v.lower() in ('false', 'no',  '0', 'off'):
        return False
    raise argparse.ArgumentTypeError(
        f"Boolean value expected, got '{v}'. Use: true/false, yes/no, 1/0"
    )


def parse_arguments(known=False):
    parser = argparse.ArgumentParser(
        description="Dual-format (LeRobot v2.1 + HDF5) teleoperation data collector"
    )

    # Dataset paths
    parser.add_argument('--root_path',    type=str,
                        default=str(Path.joinpath(ROOT, './All_datas/test')),
                        help='Root directory for LeRobot dataset + HDF5 sub-folder')
    parser.add_argument('--repo_id',      type=str, default="",
                        help='Dataset sub-directory / LeRobot repo ID')
    parser.add_argument('--datasets',     type=str,
                        default=str(Path.joinpath(ROOT, 'datasets')),
                        help='(Legacy) HDF5 dataset directory - kept for compatibility')

    # Collection parameters
    parser.add_argument('--episode_nums',  type=int, default=100,
                        help='Number of episodes to collect (-1 = unlimited)')
    parser.add_argument('--max_timesteps', type=int, default=1500,
                        help='Maximum frames per episode')
    parser.add_argument('--frame_rate',    type=int, default=25,
                        help='Collection rate in Hz')

    # Config
    parser.add_argument('--config', type=str,
                        default=str(Path.joinpath(ROOT, 'act/data/config.yaml')),
                        help='ROS operator config YAML')

    # Cameras
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist'],
                        default=['head', 'left_wrist', 'right_wrist'])
    parser.add_argument('--use_depth_image', action='store_true',
                        help='Record depth images (saved as zero-padded JPEG in HDF5)')

    # Modalities  - NOTE: use _str2bool, never type=bool (bool("False") == True in Python!)
    parser.add_argument('--use_eef',    type=_str2bool, default=False,
                        help='Save end-effector (Cartesian) action space alongside joint space')
    parser.add_argument('--use_base',   action='store_true',
                        help='Record mobile base odometry and velocity commands')
    parser.add_argument('--use_vel',    type=_str2bool, default=True,
                        help='Record joint velocities')
    parser.add_argument('--use_effort', type=_str2bool, default=True,
                        help='Record joint efforts')

    # Robot / collection options
    parser.add_argument('--record',      choices=['Distance', 'Speed'], default='Distance')
    parser.add_argument('--robot',       choices=['Single', 'Lift', 'X7', 'Acone'], default='Acone')
    parser.add_argument('--resume',      type=_str2bool, default=True,
                        help='Resume collection from the last saved episode index. '
                             'Pass --resume false to start fresh.')
    parser.add_argument('--key_collect', action='store_true',
                        help='Use keyboard Enter instead of joystick button to start each episode')
    parser.add_argument('--is_compress', type=_str2bool, default=True,
                        help='JPEG-compress images when writing HDF5')

    # Task metadata
    parser.add_argument('--task', type=str, default='',
                        help='Natural-language task description stored in every frame')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    voice_engine.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Usage examples
# ─────────────────────────────────────────────────────────────────────────────
'''
New collection (joint space only):
    python ./src/edlsrobot/datasets/collect_ledatav21_hdf5_ziqi.py \
        --root_path ./All_datas/test \
        --repo_id   pick_teapot_v1 \
        --task      "Pick up the teapot and pour the tea into the glass." \
        --resume    False

New collection (joint + EEF space):
    python ./src/edlsrobot/datasets/collect_ledatav21_hdf5_ziqi.py \
        --root_path ./All_datas/test \
        --repo_id   pick_teapot_v1 \
        --use_eef   True \
        --task      "Pick up the teapot and pour the tea into the glass." \
        --resume    False

Resume existing collection:
    python ./src/edlsrobot/datasets/collect_ledatav21_hdf5_ziqi.py \
        --root_path ./All_datas/test \
        --repo_id   pick_teapot_v1 \
        --resume    True

Output layout:
    ./All_datas/pick_teapot/pick_teapot_v1/
        ├── data/                   ← LeRobot v2.1 parquet + videos
        ├── meta/info.json
        ├── hdf5/
        │   ├── episode_0.hdf5      ← ACT-compatible HDF5
        │   ├── episode_1.hdf5
        │   └── ...
        └── time_logs/              ← per-frame timestamp JSONL (in CWD)
'''