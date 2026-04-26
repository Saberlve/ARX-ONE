import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "src" / "edlsrobot" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from openpi_client_adapter import build_openpi_arx_observation
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
