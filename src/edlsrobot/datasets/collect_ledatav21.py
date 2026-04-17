# -- coding: UTF-8
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
ROOT1 = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    sys.path.append(str(ROOT1))
    os.chdir(str(ROOT))

import time
import argparse
import rclpy
import cv2
import yaml
import threading
import json

import numpy as np
from copy import deepcopy
from edlsrobot.datasets.lerobot_v21.lerobot_dataset import LeRobotDataset
from typing import Literal
import dataclasses
import shutil
from typing import Optional

from act.utils.ros_operator import Rate, RosOperator
from act.utils.setup_loader import setup_loader


np.set_printoptions(linewidth=200)

try:
    import pyttsx3
except ModuleNotFoundError:
    pyttsx3 = None


class _SilentVoiceEngine:
    def say(self, _line):
        return None

    def runAndWait(self):
        return None

    def stop(self):
        return None


voice_engine = _SilentVoiceEngine()
if pyttsx3 is not None:
    voice_engine = pyttsx3.init()
    voice_engine.setProperty('voice', 'en')
    voice_engine.setProperty('rate', 120)  # 设置语速

voice_lock = threading.Lock()
joy_key_3 = False


def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {yaml_file}")

        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file - {e}")

        return None

def voice_process(voice_engine, line):
    # with voice_lock:
    #     voice_engine.say(line)
    #     voice_engine.runAndWait()
    print(line)

    return

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 3      #(3,8)(3,9)(4,5)(4,8)(5,4)(6,4)
    image_writer_threads_per_camera: int = 4
    video_backend: Optional[str] = None
    video_encoding_batch_size: int = 1

DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset(args,
    mode: Literal["video", "image"] = "video",
    has_depth_image: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]
    features = {
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
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": list(motors),
        },
    }

    if args.use_eef:
        features["action_eef"] = {
            "dtype": "float32",
            "shape": (14,),
            "names": "eef",
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
        
    if args.use_depth_image:
        pass

    cameras = args.camera_names

    # STEP A: 添加 "sync_timestamp.*" 特征到 LeRobot 数据集元数据
    # 这些字段不会参与模型训练，只用于事后分析：一帧内 head/left_wrist/right_wrist/left_arm/right_arm 的 ROS 原始时间戳是否对齐
    for cam in cameras:
        features[f"sync_timestamp.{cam}_ns"] = {"dtype": "int64", "shape": (1,)}
    features["sync_timestamp.left_arm_ns"] = {"dtype": "int64", "shape": (1,)}
    features["sync_timestamp.right_arm_ns"] = {"dtype": "int64", "shape": (1,)}

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    # 创建lerobotdataset数据集格式
    if args.resume:
        dataset = LeRobotDataset(
            args.repo_id,
            root=args.root_path,
        )

        if cameras and len(cameras) > 0:
            dataset.start_image_writer(
                num_processes=dataset_config.image_writer_processes,
                num_threads=dataset_config.image_writer_threads_per_camera * len(cameras),
            )
        # sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        #创建空datasets数据
        root_local = Path(args.root_path) / args.repo_id
        # root_local.mkdir(parents=True, exist_ok=False)
        if Path(Path(root_local)).exists():
            shutil.rmtree(root_local)

        dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.frame_rate,
        root=root_local,
        robot_type=args.robot,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads_per_camera * len(cameras),
        video_backend=dataset_config.video_backend,
    )

    return dataset

def collect_detect(args, start_episode, voice_engine, ros_operator):
    global init_pos, joy_key_3

    rate = Rate(args.frame_rate)
    print(f"Preparing to record episode {start_episode}")

    # 倒计时
    for i in range(1, -1, -1):
        print(f"\rwaiting {i} to start recording", end='')
        # time.sleep(0.3)
        voice_process(voice_engine, f"{i+1}")

    print(f"\nStart recording program...")

    # 键盘触发录制
    if args.key_collect:
        input("Enter any key to record :")
    else:
        init_done = False

        while not init_done and rclpy.ok():
            obs_dict = ros_operator.get_observation()
            if obs_dict == None:
                print("synchronization frame")
                rate.sleep()
                continue

            # action = obs_dict['eef']
            action = obs_dict['qpos']

            # 减少不必要的循环
            with ros_operator.joy_lock:
                triggered = dict(ros_operator.triggered_joys)
                ros_operator.triggered_joys.clear()

            if 0 in triggered:
                init_done = True
                init_pos = action
            if 2 in triggered:
                # joy_key_3 = True
                # voice_process(voice_engine, f"delete {start_episode}")
                pass


            if init_done:
                voice_process(voice_engine, f"{start_episode % 100}")
            rate.sleep()
        time.sleep(1)
        voice_process(voice_engine, "go")

        return True

def collect_and_save(args, dataset, ros_operator, voice_engine, episode_num):
    rate = Rate(args.frame_rate)
    # 初始化机器人基础位置
    # ros_operator.init_robot_base_pose()
    gripper_idx = [6, 13]
    gripper_close = -0.5    # left开最大-3.425, right开最大 -3.323；
    global joy_key_3
    
    # prev_* 用来保存上一帧的数据，最终构成 (state_t, image_t, action_t+1) 的训练样本
    prev_state, prev_vel, prev_effort = None, None, None
    prev_head_ts, prev_left_ts, prev_right_ts = None, None, None
    prev_left_arm_ts, prev_right_arm_ts = None, None
    # last_* 只用于检测当前帧相比上一帧的时间间隔是否跳帧/超时
    last_head_ts = None
    last_left_ts = None
    last_right_ts = None
    last_left_arm_ts = None
    last_right_arm_ts = None
    time_records = []

    count = 0
    i = 0
    while (count < args.max_timesteps) and rclpy.ok() and not joy_key_3:
        loop_t0 = time.perf_counter()
        #监听按键3,删除这个批次,
        with ros_operator.joy_lock:
            triggered = dict(ros_operator.triggered_joys)
            ros_operator.triggered_joys.clear()

        if 2 in triggered:
            joy_key_3 = True
            voice_process(voice_engine, f"delete {episode_num}")
            break
        # STEP B: 从 ROS 队列里取观测(observation)和动作(action)
        # 注意：ros_operator 内部对每个队列都是独立 pop() "最新"的一条，没有按时间戳对齐
        # 这意味着 obs_dict 里的 head 图、left_wrist 图、right_arm 状态等可能来自不同时刻
        t0 = time.perf_counter()
        obs_dict = ros_operator.get_observation(ts=count)
        t1 = time.perf_counter()
        # print(f"===========[get_observation] {(t1 - t0)*1000:.1f} ms")
        action_dict = ros_operator.get_action()

        # 同步帧检测：如果任一队列暂时为空，就等待；连续 10 次为空则抛异常
        if obs_dict is None or action_dict is None:
            print("Synchronization frame")
            rate.sleep()
            i += 1
            if i > 10:
                print("no camera queue !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                dataset.clear_episode_buffer()
                raise RuntimeError("Camera/action queue timeout")
            continue

        # STEP C: 提取当前循环拿到的各模态原始 ROS 时间戳（单位：纳秒）
        # 这些时间戳来自 ros_operator 内部队列的 pop()，天然可能存在跨模态时间差
        curr_head_ts = obs_dict["img_ts"]["head"]
        curr_left_ts = obs_dict["img_ts"]["left_wrist"]
        curr_right_ts = obs_dict["img_ts"]["right_wrist"]
        curr_left_arm_ts = obs_dict["arm_ts"]["left_arm"]
        curr_right_arm_ts = obs_dict["arm_ts"]["right_arm"]
        # STEP D: 计算相邻两帧之间同一模态的时间间隔 dt，用于检测跳帧/超时
        # 例如 head_dt_ms = 44.5 表示 head 相机这两帧之间间隔了 44.5ms，大于 33.3ms 说明有跳帧
        head_dt_ms = None
        left_dt_ms = None
        right_dt_ms = None
        left_arm_dt_ms = None
        right_arm_dt_ms = None
        if last_head_ts is not None:
            head_dt_ms = (curr_head_ts - last_head_ts) / 1e6
        if last_left_ts is not None:
            left_dt_ms = (curr_left_ts - last_left_ts) / 1e6
        if last_right_ts is not None:
            right_dt_ms = (curr_right_ts - last_right_ts) / 1e6
        if last_left_arm_ts is not None:
            left_arm_dt_ms = (curr_left_arm_ts - last_left_arm_ts) / 1e6
        if last_right_arm_ts is not None:
            right_arm_dt_ms = (curr_right_arm_ts - last_right_arm_ts) / 1e6

        # 更新 last_* 供下一帧计算 dt 使用
        last_head_ts = curr_head_ts
        last_left_ts = curr_left_ts
        last_right_ts = curr_right_ts
        last_left_arm_ts = curr_left_arm_ts
        last_right_arm_ts = curr_right_arm_ts

        # STEP E: 把本帧的时间日志写入 time_records（仅用于事后 human 查看，分析脚本现在读 parquet）
        time_records.append({
            "frame_index": int(count),
            "head_ts_ns": int(curr_head_ts),
            "head_dt_ms": None if head_dt_ms is None else float(head_dt_ms),
            "left_dt_ms": None if left_dt_ms is None else float(left_dt_ms),
            "right_dt_ms": None if right_dt_ms is None else float(right_dt_ms),
            "left_arm_dt_ms": None if left_arm_dt_ms is None else float(left_arm_dt_ms),
            "right_arm_dt_ms": None if right_arm_dt_ms is None else float(right_arm_dt_ms),
            "get_observation_ms": float((t1 - t0) * 1000.0),
        })
        # STEP F: 超时检测
        # 如果某一模态的 dt 超过 threshold（例如 30fps 下是 48.3ms），认为发生了严重跳帧，丢弃本 episode
        dt_list = [head_dt_ms, left_dt_ms, right_dt_ms, left_arm_dt_ms, right_arm_dt_ms]
        dt_threshold_ms = (1000.0 / args.frame_rate) + 15.0
        if any(x is not None and x > dt_threshold_ms for x in dt_list):
            joy_key_3 = True
            print(f"---------timeout (threshold={dt_threshold_ms:.1f} ms): ", dt_list)
            voice_process(voice_engine, f"timeout, delete {episode_num}")
            break

        # STEP G: 提取当前从臂状态（将被用作下一帧的 action）
        curr_state = deepcopy(obs_dict["qpos"].astype(np.float32))
        curr_vel = deepcopy(obs_dict["qvel"].astype(np.float32))
        curr_effort = deepcopy(obs_dict["effort"].astype(np.float32))
        if args.use_eef:
            curr_eef = deepcopy(obs_dict["eff"].astype(np.float32))
        # 夹爪动作处理
        for idx in gripper_idx:
            curr_state[idx] = 0 if curr_state[idx] > gripper_close else curr_state[idx]
            if args.use_eef:
                curr_eef[idx] = 0 if curr_eef[idx] > gripper_close else curr_eef[idx]

        # 检查是否超过100帧，并判断是否应该停止
        if count > 100:
            if all(abs(val - init) <= 0.1 for val, init in zip(curr_state, init_pos)):
                print("------------------------ collect over ---------------------------------")
                break   #回到初始位置，提前结束收集。
        
        # STEP H: 构造训练样本 frame。注意这里使用的是 "prev_*"（上一帧的数据）作为 observation，
        # 而 "curr_state"（当前帧的从臂状态）作为 action。
        # 这是一种 (state_t, image_t, action_t+1) 的保存方式，常用于 Diffusion Policy / ACT。
        # 这意味着 action 和 observation 之间存在一个固定步长的时序偏移。
        if prev_state is not None:
            frame = {
                "observation.state": prev_state,              # t 时刻的从臂状态
                "observation.velocity": prev_vel,             # t 时刻的速度
                "observation.effort": prev_effort,            # t 时刻的电流
                "action": curr_state,                         # t+1 时刻的从臂状态（作为动作标签）
                "observation.images.head": prev_image_head,   # t 时刻的图像
                "observation.images.left_wrist": prev_image_left,
                "observation.images.right_wrist": prev_image_right,
                # 下面 5 个 sync_timestamp 记录的是 t 时刻各模态的 ROS 原始时间戳，
                # 用于事后分析 "同一帧内 head 图和 left_arm 状态到底差了多少毫秒"
                "sync_timestamp.head_ns": np.array([prev_head_ts], dtype=np.int64),
                "sync_timestamp.left_wrist_ns": np.array([prev_left_ts], dtype=np.int64),
                "sync_timestamp.right_wrist_ns": np.array([prev_right_ts], dtype=np.int64),
                "sync_timestamp.left_arm_ns": np.array([prev_left_arm_ts], dtype=np.int64),
                "sync_timestamp.right_arm_ns": np.array([prev_right_arm_ts], dtype=np.int64),
            }
            # add task
            frame["task"] = args.task
            #
            if args.use_eef:
                frame["action_eff"] = curr_eef
            #
            if args.use_base:
                frame["action_base"] = deepcopy(obs_dict["action_base"].astype(np.float32))
                frame["action_velocity"] = deepcopy(obs_dict["base_velocity"].astype(np.float32))
                
            if args.use_depth_image:
                pass
            
            # 添加元素
            t_add0 = time.perf_counter()
            if dataset is not None:
                dataset.add_frame(frame)
            t_add1 = time.perf_counter()
            # print(f"------[add_frame] {(t_add1 - t_add0)*1000:.1f} ms")

        # STEP I: 把当前帧的数据存储为 prev_*，供下一循环构造 (state_t, action_t+1) 使用
        prev_state = curr_state
        prev_vel = curr_vel
        prev_effort = curr_effort
        prev_head_ts = curr_head_ts
        prev_left_ts = curr_left_ts
        prev_right_ts = curr_right_ts
        prev_left_arm_ts = curr_left_arm_ts
        prev_right_arm_ts = curr_right_arm_ts
        prev_image_head = obs_dict["images"]["head"]
        prev_image_left = obs_dict["images"]["left_wrist"]
        prev_image_right = obs_dict["images"]["right_wrist"]
        # use depth image
        if args.use_depth_image:
            pass

        print(f"Frame data: {count}")
        loop_t1 = time.perf_counter()
        # print(f"===========[loop total] {(loop_t1 - loop_t0)*1000:.1f} ms")
        count += 1

        if not rclpy.ok():
            exit(-1)

        rate.sleep()
    
    def save_episode_time_records(save_dir, episode_num, time_records):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"episode_{episode_num:04d}_time.jsonl"
        with open(save_path, "w", encoding="utf-8") as f:
            for item in time_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Saved time records to: {save_path}")
    
    if joy_key_3:
        dataset._wait_image_writer()
        dataset.clear_episode_buffer()
        print(f"\033[34mdelete episode {episode_num}\033[0m")
    else:
        # save
        save_episode_time_records("./time_logs", episode_num, time_records)
        dataset.save_episode()
        voice_process(voice_engine, f"Save {episode_num}")
        print(f"\033[32m\nSaved {episode_num} !!!!!! \033[0m\n")


def main(args):
    global joy_key_3
    setup_loader(ROOT)

    rclpy.init()

    config = load_yaml(args.config)

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
    # 查找最大episode序号
    def get_resume_episode_nums(path):
        json_path = path
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("total_episodes")

    num_episodes = 1000 if args.episode_nums == -1 else args.episode_nums
    
    try:
        episode_num = 0
        if args.resume:
            path_ = args.root_path / "meta" / "info.json"
            
            episode_num = get_resume_episode_nums(path=path_)
            num_episodes = episode_num + num_episodes

        while episode_num < num_episodes and rclpy.ok():
            print(f"------------->>> Collect data from Current episode index: {episode_num}")
            collect_detect(args, episode_num, voice_engine, ros_operator)
            # ---------- 采集数据 ----------
            collect_and_save(args, dataset, ros_operator, voice_engine, episode_num)
            # 不直接保存，放入队列
            if joy_key_3:
                episode_num = episode_num
                joy_key_3 = False
            else:
                episode_num = episode_num + 1

        time.sleep(0.5)
        voice_process(voice_engine, "Over")
    except KeyboardInterrupt:
        print("KeyboardInterrupt, stopping...")

    finally:
        dataset.shutdown_pool()
        spin_running = False
        spin_thread.join(timeout=1.0)
        
        if hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()   # 关键

        voice_engine.stop()
        
        ros_operator.destroy_node()
        rclpy.shutdown()



def parse_arguments(known=False):
    parser = argparse.ArgumentParser()

    # 数据集配置
    parser.add_argument('--root_path', type=str, default=Path.joinpath(ROOT, './All_datas/test'), help='dataset dir')
    parser.add_argument('--repo_id', type=str, default="", help='datasets dir name')
    parser.add_argument('--datasets', type=str, default=Path.joinpath(ROOT, 'datasets'),
                        help='dataset dir')
    parser.add_argument('--episode_nums', type=int, default=100, help='episode nums')
    parser.add_argument('--max_timesteps', type=int, default=1500, help='max timesteps')   #800
    parser.add_argument('--frame_rate', type=int, default=30, help='frame rate')
    parser.add_argument('--use_vel', type=bool, default=False, help='use joint velocity')
    parser.add_argument('--use_effort', type=bool, default=False, help='use joint effort')
    parser.add_argument('--use_eef', type=bool, default=False, help='use gripper eef')

    # 配置文件
    parser.add_argument('--config', type=str,
                        default=Path.joinpath(ROOT, 'act/data/config.yaml'),
                        help='config file')

    # 图像处理选项
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist', ],
                        default=['head', 'left_wrist', 'right_wrist'], help='camera names')
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')

    # 机器人选项
    parser.add_argument('--use_base', action='store_true', help='use robot base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')
    parser.add_argument('--robot', choices=['Single', 'Lift', 'X7', 'Acone'], default='Acone',  help='choice robot')


    # 数据采集选项
    parser.add_argument('--resume', type=bool, default=False, help='resume data collect')
    parser.add_argument('--key_collect', action='store_true', help='use key collect')

    parser.add_argument('--task', type=str, default='Pick up the teapot and pour one-third of the tea into the glass.', help='task name')
    parser.add_argument('--is_compress', type=bool, default=True, help='is use compress image')    ##

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)


'''

python ./src/edlsrobot/datasets/collect_ledatav21.py

'''

''' ziqi 3.19 add:
python ./src/edlsrobot/datasets/collect_ledatav21.py --root_path ./All_datas/test323 --resume ''
'''
