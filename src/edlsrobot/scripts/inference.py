# -- coding: UTF-8
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

os.environ["DISPLAY"] = ""
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(PROJECT_ROOT)

import collections
import draccus
import abc
import yaml
import numpy as np
import torch
import argparse
import collections
import pickle
import yaml
from einops import rearrange
import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
import math
import logging
import time
import cv2
from act.utils.ros_operator import RosOperator, Rate
from functools import partial
import signal
from src.edlsrobot.common.configs import parser
from act.utils.setup_loader import setup_loader
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from pprint import pformat
from dataclasses import asdict, dataclass, field
import numpy as np
from src.edlsrobot.common.robot_devices.control_configs import (
    ControlConfig,
    InferenceControlConfig,
)
# from src.edlsrobot.datasets.lerobot_v21.lerobot_dataset import LeRobotDataset
# from src.edlsrobot.common.utils.utils import  init_logging
from src.edlsrobot.common.robot_devices.control_utils import sanity_check_dataset_name

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats

from lerobot.utils.visualization_utils import init_rerun
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    ACone
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
)
from lerobot.processor import (
    make_default_processors
)
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor.factory import make_default_robot_observation_processor, make_default_robot_action_processor
from lerobot.utils.control_utils import (
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from openpi_client_adapter import build_openpi_arx_observation
from openpi_client_adapter import check_arx_action_safety
from openpi_client_adapter import describe_debug_payload
from openpi_client_adapter import save_debug_observation_images
from openpi_client_adapter import ActionSafetyError
from openpi_client_adapter import select_first_openpi_action

logger = logging.getLogger(__name__)
obs_dict = collections.OrderedDict()


# 设置打印输出行宽
np.set_printoptions(linewidth=200)

# 禁用科学计数法
np.set_printoptions(suppress=True)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = False
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 1
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")

@dataclass
class InferenceConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    
    data: Union[Path, str] = Path.joinpath(Path(ROOT), './act/data/config.yaml')
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = False
    # Resume recording on an existing dataset.
    resume: bool = False
    # Query an OpenPI websocket policy server instead of loading a local LeRobot policy.
    use_openpi_server: bool = False
    # Hostname or IP address of the OpenPI websocket policy server.
    openpi_server_host: str = "localhost"
    # Port of the OpenPI websocket policy server.
    openpi_server_port: int = 8000

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None and not self.use_openpi_server:
            raise ValueError("Choose a policy, a teleoperator, or an OpenPI server to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


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

def make_shm_name_dict(args, shapes):
    shm_name_dict = {}
    for cam in args.camera_names:
        shm_name_dict[cam] = f"shm_img_{cam}"
    for state_key in shapes["states"]:
        shm_name_dict[state_key] = f"shm_state_{state_key}"
    shm_name_dict["action"] = "shm_action"

    return shm_name_dict


def create_shm_dict(args, shm_name_dict, shapes, dtypes):
    shm_dict = {}
    for cam, shape in shapes["images"].items():
        size = np.prod(shape) * np.dtype(dtypes[cam]).itemsize
        shm = SharedMemory(name=shm_name_dict[cam], create=True, size=size)
        shm_dict[cam] = (shm, shape, dtypes[cam])
    for state_key, shape in shapes["states"].items():
        size = np.prod(shape) * np.dtype(np.float32).itemsize
        shm = SharedMemory(name=shm_name_dict[state_key], create=True, size=size)
        shm_dict[state_key] = (shm, shape, np.float32)

    # action_shape = config['policy_config']['action_dim']
    action_shape = args.action_dim
    size = np.prod(action_shape) * np.dtype(np.float32).itemsize
    shm = SharedMemory(name=shm_name_dict["action"], create=True, size=size)
    shm_dict["action"] = (shm, action_shape, np.float32)

    return shm_dict


def connect_shm_dict(shm_name_dict, shapes, dtypes, args):
    shm_dict = {}
    for cam, shape in shapes["images"].items():
        shm = SharedMemory(name=shm_name_dict[cam], create=False)
        shm_dict[cam] = (shm, shape, dtypes[cam])
    for state_key, shape in shapes["states"].items():
        shm = SharedMemory(name=shm_name_dict[state_key], create=False)
        shm_dict[state_key] = (shm, shape, np.float32)

    action_shape = args.action_dim
    shm = SharedMemory(name=shm_name_dict["action"], create=False)
    shm_dict["action"] = (shm, action_shape, np.float32)

    return shm_dict


def robot_action(action, action_queue):
    action_queue.put(action.copy())

def process_obs(obs_dict):
    head_cam = obs_dict['images']['head']
    head_cam = cv2.resize(head_cam, (320, 240))
    head_cam = (np.moveaxis(head_cam, -1, 0) / 255)
    # left_cam = (np.moveaxis(obs_dict['images']['left_wrist'], -1, 0) / 255)
    # right_cam = (np.moveaxis(obs_dict['images']['right_wrist'], -1, 0) / 255)
    print("head_cam shape:", head_cam.shape, "dtype:", head_cam.dtype)

    obs = dict(
        head_cam=head_cam,
        # left_cam=left_cam,
        # right_cam=right_cam,
    )

    obs['agent_pos'] = obs_dict['qpos']
    return obs


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        # print(f'{cam_name=}')
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    return curr_image


def get_depth_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_images.append(observation['images_depth'][cam_name])
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    return curr_image


def apply_gripper_gate(action_value, gate):
    min_gripper = 0
    max_gripper = 5

    return min_gripper if action_value < gate else max_gripper


def get_obervations(args, timestep, ros_operator):
    global obs_dict

    rate = Rate(args.frame_rate)
    while True and rclpy.ok():
        obs_dict = ros_operator.get_observation(ts=timestep)
        if not obs_dict:
            print("syn fail")
            rate.sleep()

            continue

        return obs_dict

def move_to_target(ros_operator, target_pos, steps=50):
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    end = np.array(target_pos)

    for i in range(steps):
        t = i / (steps - 1)
        s = (1 - math.cos(math.pi * t)) / 2
        pos = start + (end - start) * s

        ros_operator.follow_arm_publish_continuous(pos.tolist(), pos.tolist())

def init_robot(ros_operator, use_base, connected_event, start_event):
    # init0 = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, -2.8]
    # init1 = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.0]
    ## second pos
    init0 = [0.0, 0.650, 0.700, -0.900, 0.0, 0.0, 0.0]    #go_home_position
    init1 = [0.0, 0.650, 0.700, -0.900, 0.0, 0.0, 0.0]

    # 发布初始位置（关节空间姿态）
    move_to_target(ros_operator, init0, steps=100)
    # ros_operator.robot_base_shutdown()

    connected_event.set()
    start_event.wait()

    ros_operator.follow_arm_publish_continuous(init0, init1)
    if use_base:
        ros_operator.start_base_control_thread()

def signal_handler(signal, frame, ros_operator):
    print('Caught Ctrl+C / SIGINT signal')

    # 底盘给零
    ros_operator.base_enable = False
    ros_operator.robot_base_shutdown()
    ros_operator.base_control_thread.join()

    sys.exit(0)

def cleanup_shm(names):
    for name in names:
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

def ros_process(args, meta_queue, connected_event, start_event, shm_ready_event, action_queue):
    setup_loader(ROOT)

    rclpy.init()

    data = load_yaml(args.data)
    ros_operator = RosOperator(args, data, in_collect=False)
    max_safe_joint_step = float(os.environ.get("ARX_MAX_SAFE_JOINT_STEP", "0.08"))
    max_safe_gripper_step = float(os.environ.get("ARX_MAX_SAFE_GRIPPER_STEP", "1.25"))

    # def _spin_loop(node):
    #     while rclpy.ok():
    #         rclpy.spin_once(node, timeout_sec=0.001)
    def _spin_loop(node):
        print("[DEBUG] _spin_loop started")
        try:
            spin_count = 0
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.001)
                spin_count += 1
                if spin_count <= 5 or spin_count % 500 == 0:
                    print(f"[DEBUG] spin_once count={spin_count}, left_deque={len(node.feedback_left_arm_deque)}")
            print(f"[DEBUG] _spin_loop exited, rclpy.ok()={rclpy.ok()}, spin_count={spin_count}")
        except ExternalShutdownException:
            print("[DEBUG] _spin_loop ExternalShutdownException")
        except Exception as e:
            print(f"_spin_loop exception: {e}")

    spin_thread = threading.Thread(target=_spin_loop, args=(ros_operator,), daemon=True)
    spin_thread.start()

    if args.use_base:
        signal.signal(signal.SIGINT, partial(signal_handler, ros_operator=ros_operator))

    init_robot(ros_operator, args.use_base, connected_event, start_event)

    rate = Rate(args.frame_rate)
    while rclpy.ok():
        obs = ros_operator.get_observation()
        if obs:
            shapes = {"images": {}, "states": {}, "dtypes": {}}

            for cam in args.camera_names:
                img = obs["images"][cam]
                shapes["images"][cam] = img.shape
                shapes["dtypes"][cam] = img.dtype
            shapes["states"]["qpos"] = obs["qpos"].shape
            shapes["states"]["qvel"] = obs["qvel"].shape
            shapes["states"]["effort"] = obs["effort"].shape
            shapes["states"]["robot_base"] = obs["robot_base"].shape
            shapes["states"]["base_velocity"] = obs["base_velocity"].shape

            meta_queue.put(shapes)

            break

        rate.sleep()

    # 创建共享内存
    shm_name_dict = meta_queue.get()

    cleanup_shm(shm_name_dict.values())
    shm_dict = create_shm_dict(args, shm_name_dict, shapes, shapes["dtypes"])
    shm_ready_event.set()

    rate = Rate(args.frame_rate)
    while rclpy.ok():
        obs = ros_operator.get_observation()
        if not obs:
            rate.sleep()
            continue

        # 写入共享内存
        for cam in args.camera_names:
            shm, shape, dtype = shm_dict[cam]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs["images"][cam]
        for state_key in shapes["states"]:
            shm, shape, dtype = shm_dict[state_key]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs[state_key]

        # 读取动作并执行
        action = action_queue.get()
        if  np.any(action):  # 确保动作不全是 0
            gripper_gate = args.gripper_gate
            gripper_idx = [6, 13]
            action = np.asarray(action, dtype=np.float32).copy()

            left_action = action[:gripper_idx[0] + 1]  # 取8维度
            if gripper_gate != -1:
                left_action[gripper_idx[0]] = apply_gripper_gate(left_action[gripper_idx[0]], gripper_gate)

            right_action = action[gripper_idx[0] + 1:gripper_idx[1] + 1]
            if gripper_gate != -1:
                right_action[gripper_idx[0]] = apply_gripper_gate(right_action[gripper_idx[0]], gripper_gate)

            try:
                check_arx_action_safety(
                    action,
                    obs["qpos"],
                    max_joint_step=max_safe_joint_step,
                    max_gripper_step=max_safe_gripper_step,
                )
            except ActionSafetyError as exc:
                print(f"[SAFETY STOP] {exc}")
                hold_qpos = np.asarray(obs["qpos"], dtype=np.float32)
                print(f"[SAFETY STOP] current qpos: {describe_debug_payload(hold_qpos)}")
                print(f"[SAFETY STOP] rejected action: {describe_debug_payload(action[:14])}")
                ros_operator.follow_arm_publish(hold_qpos[:7], hold_qpos[7:14])
                break

            ros_operator.follow_arm_publish(left_action, right_action)

            if args.use_base:
                action_base = action[gripper_idx[1] + 1:gripper_idx[1] + 1 + 10]
                ros_operator.set_robot_base_target(action_base)

        rate.sleep()

    # executor.shutdown()
    rclpy.shutdown()
    for shm, _, _ in shm_dict.values():
        shm.close()
        shm.unlink()

def init_infer_engine(cfg: InferenceConfig):
    init_logging()
    logging.info(pformat(asdict(cfg.robot)))
    # if cfg.display_data:
    #     init_rerun(session_name="recording")

    if cfg.use_openpi_server:
        from openpi_client import websocket_client_policy

        return {
            "openpi_client": websocket_client_policy.WebsocketClientPolicy(
                host=cfg.openpi_server_host,
                port=cfg.openpi_server_port,
            ),
        }

    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    # dataset_features = cfg.robot.features

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        # Load pretrained policy
        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor = None
        postprocessor = None
        if cfg.policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )

        # Reset policy and processor if they are provided
        if policy is not None and preprocessor is not None and postprocessor is not None:
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
        
        robot_observation_processor = make_default_robot_observation_processor()
        device = get_safe_torch_device(policy.config.device)

        return {
        "robot": robot,
        "policy": policy,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "robot_observation_processor": robot_observation_processor,
        "device": device,
        "openpi_client": None,
        }

def infer_step(engine, cfg: InferenceConfig, obs: dict):
    obs_processed = engine["robot_observation_processor"](obs)

    action_values = predict_action(
        observation=obs_processed,
        policy=engine["policy"],
        device=engine["device"],
        preprocessor=engine["preprocessor"],
        postprocessor=engine["postprocessor"],
        use_amp=engine["policy"].config.use_amp,
        task=cfg.dataset.single_task,
        robot_type=engine["robot"].robot_type,
    )

    return action_values

def plot_action_curves(action_history, save_path="action_curves.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    actions = np.asarray(action_history, dtype=np.float32)  # [T, 14]
    if actions.ndim != 2 or actions.shape[1] != 14:
        print(f"Invalid action_history shape: {actions.shape}")
        return

    left_names = ["L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_J6", "L_Gripper"]
    right_names = ["R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_J6", "R_Gripper"]

    fig, axes = plt.subplots(7, 2, figsize=(14, 20), sharex=True)

    for i in range(7):
        axes[i, 0].plot(actions[:, i])
        axes[i, 0].set_title(left_names[i])
        axes[i, 0].grid(True)

        axes[i, 1].plot(actions[:, i + 7])
        axes[i, 1].set_title(right_names[i])
        axes[i, 1].grid(True)

    axes[-1, 0].set_xlabel("Timestep")
    axes[-1, 1].set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved action curve figure to: {save_path}")

def plot_action_diff_curves(action_history, save_path="action_diff_curves.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    actions = np.asarray(action_history, dtype=np.float32)   # [T, 14]
    diffs = np.diff(actions, axis=0)                         # [T-1, 14]

    joint_names = [
        "L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_J6", "L_Gripper",
        "R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_J6", "R_Gripper",
    ]

    fig, axes = plt.subplots(14, 1, figsize=(12, 28), sharex=True)

    for i in range(14):
        axes[i].plot(diffs[:, i])
        axes[i].set_ylabel(joint_names[i], rotation=0, labelpad=35)
        axes[i].grid(True)

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved action diff figure to: {save_path}")

def inference_process(cfg, shm_dict, shapes, ros_proc, action_queue):
    engine = init_infer_engine(cfg)
    
    action_history = []   # 新增：保存每一步 action
    rate = Rate(cfg.robot.frame_rate)
    timestep = 0
    debug_io = env_flag("OPENPI_DEBUG_IO", default=True)
    debug_dir = os.environ.get("OPENPI_DEBUG_DIR", "openpi-arx-debug")
    # while timestep < cfg.robot.max_publish_step and ros_proc.is_alive():
    while timestep <3000 and ros_proc.is_alive():
        obs_dict = {"images": {}, "qpos": None, "qvel": None, "effort": None,
                            "robot_base": None, "base_velocity": None}

        # 从共享内存读取
        for cam in cfg.robot.camera_names:
            shm, shape, dtype = shm_dict[cam]
            obs_dict["images"][cam] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        for state_key in shapes["states"]:
            shm, shape, dtype = shm_dict[state_key]
            obs_dict[state_key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        observation = {
            "observation.state": (obs_dict["qpos"]),
            "observation.velocity": (obs_dict["qvel"]),
            "observation.effort": (obs_dict["effort"]),
            "observation.images.head": (obs_dict["images"]["head"]),
            "observation.images.left_wrist": (obs_dict["images"]["left_wrist"]),
            "observation.images.right_wrist": (obs_dict["images"]["right_wrist"]),
        }


        # 远程推理
        inference_start = time.time()
        if engine["openpi_client"] is not None:
            openpi_observation = build_openpi_arx_observation(obs_dict, cfg.dataset.single_task)
            if debug_io:
                print(f"[DEBUG] client observation: {describe_debug_payload(openpi_observation)}")
                saved_paths = save_debug_observation_images(
                    openpi_observation,
                    debug_dir,
                    prefix="client_obs",
                    step=timestep,
                )
                if saved_paths:
                    print(f"[DEBUG] client observation images saved: {[str(path) for path in saved_paths]}")
            results = engine["openpi_client"].infer(openpi_observation)
            if debug_io:
                print(f"[DEBUG] client received result: {describe_debug_payload(results)}")
            action = select_first_openpi_action(results)
            if debug_io:
                print(f"[DEBUG] client selected action: {describe_debug_payload(action)}")
        else:
            results = infer_step(engine, cfg, observation)
            action = results.squeeze().numpy()
        client_infer_ms = 1000 * (time.time() - inference_start)
        print("client_infer_ms: ", client_infer_ms)

        # 执行动作
        # print("+++++++++++++++++++++  ", action)
        # time.sleep(1)
        robot_action(action, action_queue)
        
        action_history.append(action.copy())   # 新增：记录

        timestep += 1
        rate.sleep()
    # 循环结束后画图
    # if len(action_history) > 0:
    #     plot_action_curves(action_history, save_path="action_curves.png")
    #     plot_action_diff_curves(action_history, save_path="action_diff_curves.png")

@parser.wrap()
def main(cfg: InferenceConfig):
    # args = cfg.robot
    
    meta_queue = mp.Queue()
    action_queue = mp.Queue(maxsize=100)

    connected_event = mp.Event()
    start_event = mp.Event()
    shm_ready_event = mp.Event()

    # 启动ROS进程
    ros_proc = mp.Process(target=ros_process, args=(cfg.robot, meta_queue, connected_event, start_event, shm_ready_event, action_queue))
    ros_proc.start()

    connected_event.wait()
    print("Robot connected, starting inference...")
    start_event.set()

    # 等待meta信息
    shapes = meta_queue.get()

    shm_name_dict = make_shm_name_dict(cfg.robot, shapes)

    meta_queue.put(shm_name_dict)

    shm_ready_event.wait()

    shm_dict = connect_shm_dict(shm_name_dict, shapes, shapes["dtypes"], cfg.robot)

    # 推理
    try:
        inference_process(cfg, shm_dict, shapes, ros_proc, action_queue)
    except KeyboardInterrupt:
        pass
    finally:
        for shm, _, _ in shm_dict.values():
            shm.close()

        ros_proc.join(timeout=1.0)
        if ros_proc.is_alive():
            print("ros_proc still alive, terminating...")
            ros_proc.terminate()
            ros_proc.join(timeout=2.0)   # <<< 改这里，别无限等

        if ros_proc.is_alive():
            print("ros_proc still alive after terminate")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    main()




"""
bash infer.sh

LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
--robots.type=ACOne
"""
