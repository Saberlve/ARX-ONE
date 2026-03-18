# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import draccus

# from src.common.robot_devices.robots.configs import RobotConfig
from src.edlsrobot.common.configs import parser
from src.edlsrobot.common.configs.policies import PreTrainedConfig
# from lerobot.policies.pretrained import PreTrainedConfig

@dataclass
class ControlConfig(draccus.ChoiceRegistry):
    pass


@ControlConfig.register_subclass("inference")
@dataclass
class InferenceControlConfig(ControlConfig):
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: Optional[Union[str, Path]] = None
    policy: Optional[PreTrainedConfig] = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: Optional[int] = None
    # Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.
    warmup_time_s: Union[int, float] = 10
    # Number of seconds for data recording for each episode.
    episode_time_s: Union[int, float] = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: Union[int, float] = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = False
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: Optional[List[str]] = None
    # Number of subprocesses handling the saving of frames as PNG.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    num_image_writer_threads_per_camera: int = 4
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("control.policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("control.policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

