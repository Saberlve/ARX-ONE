import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

sys.modules.setdefault("rclpy", types.SimpleNamespace(ok=lambda: False, init=lambda: None, shutdown=lambda: None))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault(
    "edlsrobot.datasets.lerobot_v21.lerobot_dataset",
    types.SimpleNamespace(LeRobotDataset=object),
)
sys.modules.setdefault(
    "act.utils.ros_operator",
    types.SimpleNamespace(Rate=object, RosOperator=object),
)
sys.modules.setdefault(
    "act.utils.setup_loader",
    types.SimpleNamespace(setup_loader=lambda _root: None),
)

from edlsrobot.datasets.collect_ledatav21 import drop_last_episode_buffer_frame, normalize_gripper_state


class CollectStopConditionTest(unittest.TestCase):
    def test_normalize_gripper_state_zeroes_only_open_grippers(self) -> None:
        state = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.2, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, -0.3],
            dtype=np.float32,
        )

        normalized = normalize_gripper_state(state, gripper_close=-0.5)

        self.assertAlmostEqual(float(normalized[6]), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[13]), 0.0, places=6)
        self.assertEqual(normalized[5], state[5])
        self.assertEqual(normalized[12], state[12])

    def test_normalize_gripper_state_preserves_closed_grippers(self) -> None:
        state = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, -0.9],
            dtype=np.float32,
        )

        normalized = normalize_gripper_state(state, gripper_close=-0.5)

        self.assertAlmostEqual(float(normalized[6]), -0.8, places=6)
        self.assertAlmostEqual(float(normalized[13]), -0.9, places=6)

    def test_drop_last_episode_buffer_frame_removes_only_per_frame_values(self) -> None:
        buffer = {
            "size": 3,
            "episode_index": 7,
            "task": ["pick", "pick", "pick"],
            "observation.state": ["s0", "s1", "s2"],
            "observation.images.head": ["h0", "h1", "h2"],
            "timestamp": [0.0, 1.0 / 30.0, 2.0 / 30.0],
            "frame_index": [0, 1, 2],
        }

        removed = drop_last_episode_buffer_frame(buffer)

        self.assertTrue(removed)
        self.assertEqual(buffer["size"], 2)
        self.assertEqual(buffer["episode_index"], 7)
        self.assertEqual(buffer["task"], ["pick", "pick"])
        self.assertEqual(buffer["observation.state"], ["s0", "s1"])
        self.assertEqual(buffer["observation.images.head"], ["h0", "h1"])
        self.assertEqual(buffer["timestamp"], [0.0, 1.0 / 30.0])
        self.assertEqual(buffer["frame_index"], [0, 1])


if __name__ == "__main__":
    unittest.main()
