import importlib.util
import pathlib
import unittest

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "analyze_sync.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_sync", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnalyzeSyncTest(unittest.TestCase):
    def test_analyze_episode_flags_duplicate_timestamps(self) -> None:
        analyze_sync = load_module()
        df = pd.DataFrame(
            {
                "sync_timestamp.head_ns": [0, 33_000_000, 66_000_000, 66_000_000],
                "sync_timestamp.left_wrist_ns": [1_000_000, 34_000_000, 67_000_000, 67_000_000],
                "sync_timestamp.right_wrist_ns": [2_000_000, 35_000_000, 68_000_000, 68_000_000],
                "sync_timestamp.left_arm_ns": [3_000_000, 36_000_000, 69_000_000, 69_000_000],
                "sync_timestamp.right_arm_ns": [4_000_000, 37_000_000, 70_000_000, 70_000_000],
            }
        )

        result = analyze_sync.analyze_episode(df, fps=30, episode_index=0)

        self.assertEqual(result["duplicate_or_reverse_frames"]["sync_timestamp.head_ns"], 1)
        self.assertEqual(result["duplicate_or_reverse_frames"]["sync_timestamp.left_arm_ns"], 1)
        self.assertGreater(result["max_diff_ms"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
