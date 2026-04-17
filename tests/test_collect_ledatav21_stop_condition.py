import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edlsrobot.datasets.collect_ledatav21 import normalize_gripper_state


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


if __name__ == "__main__":
    unittest.main()
