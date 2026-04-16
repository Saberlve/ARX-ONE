# -- coding: UTF-8
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    arxx7,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)

from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
##

# import os
# import sys
import socket
#
# from pathlib import Path
import websocket_server_policy
#
# import logging
# from dataclasses import dataclass, field
# from typing import List, Union, Dict, Any
# import draccus
# import abc
#
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
#     os.chdir(str(ROOT))
#
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PROJECT_ROOT)
#
# from src.lerobot.datasets.lerobot_dataset import LeRobotDataset
# from src.lerobot.policies.factory import make_policy
# from src.
# from src.common.robot_devices.control_configs import (
#     ControlConfig,
#     RecordControlConfig,
# )
#
# from src.common.robot_devices.control_utils import (
#     sanity_check_dataset_name,
# )
# from src.common.utils.utils import init_logging
# from src.common.configs import parser


logger = logging.getLogger(__name__)



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
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
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

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

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
        
        
    return policy, preprocessor, postprocessor, cfg, dataset, robot



def main():
    register_third_party_devices()
    policy, preprocessor, postprocessor, cfg, dataset, robot = record()

    hostname = socket.gethostname()
    # local_ip = socket.gethostbyname(hostname)
    local_ip = "0.0.0.0"
    port = 8080
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # policy, _ = get_policy(cfg)
    server = websocket_server_policy.WebsocketPolicyServer(
        config=cfg,
        dataset=dataset,
        robot=robot,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        host=local_ip,
        port=port,
        metadata={},
    )
    server.serve_forever()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("something")
    main()





"""
LD_LIBRARY_PATH=/opt/ros/noetic/lib:$LD_LIBRARY_PATH

python ./client/inference_pi_server_webs.py \
--robots.type=arx \
--control.type=record \
--control.fps=30 \
--control.single_task='' \
--control.repo_id=outputs/infer/pi0_grasp_bott100 \
--control.root=/home/hkk/Desktop/mobile_aloha2/run/outputs/infer/pi0_grasp_bott100 \
--control.policy.path=/home/hkk/Desktop/mobile_aloha2/run/outputs/train/pi0_grasp_bott100/checkpoints/last/pretrained_model


lerobot-record  \
  --robot.type=so100_follower \
  --robot.cameras="{ head: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, 
                    left_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30},
                    right_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}}" \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
  --dataset.repo_id=${HF_USER}/eval_so100 \
  --dataset.single_task="Put lego brick into the transparent box" \
  --policy.path=/home/hkk/Desktop/data/diffusion_grasp_bott300v3/checkpoints/100000/pretrained_model



pi05
--robot.type=arxx7
--robot.id=my_arxx7_arm
--robot.cameras="{ head: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}}"
--display_data=false
--dataset.repo_id=${HF_USER}/eval_arxx7_pi05
--dataset.single_task="Place the bottle into the basket."
# --dataset.single_task="Place the brown glass bottle in the middle of the table with your left hand, and then put it into the basket with your right hand."
--policy.path=/media/hkk/PSSD/Arx/train/pi05_grasp_bott100v3_all/checkpoints/100000/pretrained_model



dp
--robot.type=arxx7
--robot.id=my_arxx7_arm
--robot.cameras="{ head: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}}"
--display_data=false
--dataset.repo_id=${HF_USER}/eval_arxx7
--dataset.single_task="Place the bottle into the basket."
--policy.path=/home/hkk/Desktop/data/diffusion_grasp_bott300v3/checkpoints/100000/pretrained_model

--policy.path=/media/hkk/PSSD/Arx/train/diffusion_grasp_brown_glass400v3/checkpoints/100000/pretrained_model


smolvla
--robot.type=arxx7
--robot.id=my_arxx7_arm
--robot.cameras="{ camera1: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, camera3: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}}"
--display_data=false
--dataset.repo_id=${HF_USER}/eval_arxx7_smolvla
--dataset.single_task="Place the brown glass bottle in the middle of the table with your left hand, and then put it into the basket with your right hand."
--policy.path=/media/hkk/PSSD/Arx/train/smol_grasp_brown_glass200v3/checkpoints/080000/pretrained_model

--policy.path=/home/hkk/nasPublic/hkk/outputs/train/smol_grasp_brown_glass200v3/checkpoints/060000/pretrained_model


"""