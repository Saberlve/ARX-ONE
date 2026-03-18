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
import abc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from src.edlsrobot.common.optim.optimizers import OptimizerConfig
from src.edlsrobot.common.optim.schedulers import LRSchedulerConfig
from src.edlsrobot.common.utils.hub import HubMixin
from src.edlsrobot.common.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from src.edlsrobot.common.configs.types import FeatureType, NormalizationMode, PolicyFeature

from typing import Dict, Optional, List, Union

import builtins
import json
import tempfile
from logging import getLogger
from typing import Any, TypeVar
from src.edlsrobot.common.utils.constants import ACTION, OBS_STATE

# Generic variable that is either PreTrainedConfig or a subclass thereof
T = TypeVar("T", bound="PreTrainedConfig")


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """

    n_obs_steps: int = 1
    normalization_mapping: Dict[str, NormalizationMode] = field(default_factory=dict)

    input_features: Dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: Dict[str, PolicyFeature] = field(default_factory=dict)

    device: Optional[str] = None    # cuda | cpu | mp
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    def __post_init__(self):
        self.pretrained_path = None
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logging.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logging.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def observation_delta_indices(self) -> Optional[List]:
        raise NotImplementedError

    @abc.abstractproperty
    def action_delta_indices(self) -> Optional[List]:
        raise NotImplementedError

    @abc.abstractproperty
    def reward_delta_indices(self) -> Optional[List]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> Optional[LRSchedulerConfig]:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> Optional[PolicyFeature]:
        for _, ft in self.input_features.items():
            if ft.type == FeatureType.STATE:    #
                return ft
        return None

    @property
    def env_state_feature(self) -> Optional[PolicyFeature]:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:  #
                return ft
        return None

    @property
    def image_features(self) -> Dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type == FeatureType.VISUAL}    ## is -> ==

    @property
    def action_feature(self) -> Optional[PolicyFeature]:
        for _, ft in self.output_features.items():
            if ft.type == FeatureType.ACTION:   #
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
            cls: Type[T],
            pretrained_name_or_path: Union[str, Path],
            *,
            force_download: bool = False,
            resume_download: Optional[bool] = None,
            proxies: Optional[Dict] = None,
            token: Optional[Union[str, bool]] = None,
            cache_dir: Optional[Union[str, Path]] = None,
            local_files_only: bool = False,
            revision: Optional[str] = None,
            **policy_kwargs,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: Optional[str] = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                print(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # HACK: this is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        return draccus.parse(cls, config_file, args=cli_overrides)


T_ = TypeVar("T_", bound="PreTrainedConfig_pi05")
logger = getLogger(__name__)

@dataclass
class PreTrainedConfig_pi05(draccus.ChoiceRegistry, HubMixin, abc.ABC):  # type: ignore[misc,name-defined] #TODO: draccus issue
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """

    n_obs_steps: int = 1

    input_features: Dict[str, "PolicyFeature"] = field(default_factory=dict)
    output_features: Dict[str, "PolicyFeature"] = field(default_factory=dict)

    device: Optional[str] = None  # e.g. "cuda", "cuda:0", "cpu", or "mps"

    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation.
    # With AMP, automatic gradient scaling is used.
    use_amp: bool = False

    push_to_hub: bool = True  # type: ignore[assignment] # TODO: use a different name to avoid override
    repo_id: Optional[str] = None

    # Upload on private repository on the Hugging Face hub.
    private: Optional[bool] = None
    # Add tags to your policy on the hub.
    tags: Optional[List[str]] = None
    # Add license to your policy on the hub.
    license: Optional[str] = None
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch.
    pretrained_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> Optional[List]:  # type: ignore[type-arg] #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> Optional[List]:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> Optional[List]:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> Optional[LRSchedulerConfig]:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> Optional[PolicyFeature]:
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> Optional[PolicyFeature]:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> Dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> Optional[PolicyFeature]:
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
            cls: Type[T_],
            pretrained_name_or_path: Union[str, Path],
            *,
            force_download: bool = False,
            resume_download: Optional[bool] = None,
            proxies: Optional[Dict[Any, Any]] = None,
            token: Optional[Union[str, bool]] = None,
            cache_dir: Optional[Union[str, Path]] = None,
            local_files_only: bool = False,
            revision: Optional[str] = None,
            **policy_kwargs: Any,
    ) -> T_:
        model_id = str(pretrained_name_or_path)
        config_file: Optional[str] = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        # HACK: Parse the original config to get the config subclass, so that we can
        # apply cli overrides.
        # This is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        with draccus.config_type("json"):
            orig_config = draccus.parse(cls, config_file, args=[])

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        with open(config_file) as f:
            config = json.load(f)

        config.pop("type")
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        with draccus.config_type("json"):
            return draccus.parse(orig_config.__class__, config_file, args=cli_overrides)