import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "src" / "edlsrobot" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from openpi_client_adapter import build_openpi_arx_observation
from openpi_client_adapter import check_arx_action_safety
from openpi_client_adapter import describe_debug_payload
from openpi_client_adapter import save_debug_observation_images
from openpi_client_adapter import ActionSafetyError
from openpi_client_adapter import select_first_openpi_action


def test_build_openpi_arx_observation_uses_policy_keys_and_prompt():
    head = np.zeros((480, 640, 3), dtype=np.uint8)
    left = np.ones((480, 640, 3), dtype=np.uint8)
    right = np.full((480, 640, 3), 2, dtype=np.uint8)
    qpos = np.arange(14, dtype=np.float32)

    obs = build_openpi_arx_observation(
        {
            "images": {
                "head": head,
                "left_wrist": left,
                "right_wrist": right,
            },
            "qpos": qpos,
        },
        prompt="place the bottle into the basket",
    )

    assert obs["observation/images/head"] is head
    assert obs["observation/images/left_wrist"] is left
    assert obs["observation/images/right_wrist"] is right
    assert obs["observation/state"] is qpos
    assert obs["prompt"] == "place the bottle into the basket"


def test_select_first_openpi_action_returns_first_chunk_action():
    action_chunk = np.arange(140, dtype=np.float32).reshape(10, 14)

    action = select_first_openpi_action({"actions": action_chunk})

    np.testing.assert_array_equal(action, action_chunk[0])


def test_select_first_openpi_action_rejects_missing_actions():
    with pytest.raises(KeyError, match="actions"):
        select_first_openpi_action({})


def test_select_first_openpi_action_rejects_empty_chunk():
    with pytest.raises(ValueError, match="empty"):
        select_first_openpi_action({"actions": np.empty((0, 14), dtype=np.float32)})


def test_check_arx_action_safety_allows_small_joint_steps():
    qpos = np.zeros(14, dtype=np.float32)
    action = np.zeros(14, dtype=np.float32)
    action[1] = 0.02
    action[7] = -0.03
    action[13] = -0.5

    checked = check_arx_action_safety(action, qpos, max_joint_step=0.05, max_gripper_step=1.0)

    np.testing.assert_array_equal(checked, action)


def test_check_arx_action_safety_rejects_large_joint_step():
    qpos = np.zeros(14, dtype=np.float32)
    action = np.zeros(14, dtype=np.float32)
    action[8] = 0.2

    with pytest.raises(ActionSafetyError, match="joint step"):
        check_arx_action_safety(action, qpos, max_joint_step=0.05, max_gripper_step=1.0)


def test_check_arx_action_safety_rejects_nonfinite_action():
    qpos = np.zeros(14, dtype=np.float32)
    action = np.zeros(14, dtype=np.float32)
    action[0] = np.nan

    with pytest.raises(ActionSafetyError, match="finite"):
        check_arx_action_safety(action, qpos, max_joint_step=0.05, max_gripper_step=1.0)


def test_describe_debug_payload_summarizes_images_and_vectors():
    payload = {
        "observation/images/head": np.zeros((8, 10, 3), dtype=np.uint8),
        "observation/state": np.arange(14, dtype=np.float32),
    }

    summary = describe_debug_payload(payload)

    assert summary["observation/images/head"] == {
        "dtype": "uint8",
        "shape": [8, 10, 3],
        "min": 0.0,
        "max": 0.0,
    }
    assert summary["observation/state"]["values"] == list(np.arange(14, dtype=np.float32))


def test_save_debug_observation_images_writes_camera_files(tmp_path):
    payload = {
        "observation/images/head": np.zeros((8, 10, 3), dtype=np.uint8),
        "observation/images/right_wrist": np.ones((8, 10, 3), dtype=np.uint8) * 255,
    }

    paths = save_debug_observation_images(payload, str(tmp_path), prefix="client_obs", step=2)

    assert [path.name for path in paths] == [
        "client_obs_step_000002_head.jpg",
        "client_obs_step_000002_right_wrist.jpg",
    ]
    assert all(path.is_file() for path in paths)
