import pathlib
import sys
import unittest
from collections import deque

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act.utils.sync_utils import consume_nearest


class ConsumeNearestTest(unittest.TestCase):
    def test_consume_nearest_returns_closest_entry_and_drops_older(self) -> None:
        entries = deque(
            [
                (100, "a"),
                (200, "b"),
                (320, "c"),
                (450, "d"),
            ]
        )

        selected = consume_nearest(entries, target_ts=300, max_diff_ns=100)

        self.assertEqual(selected, (320, "c"))
        self.assertEqual(list(entries), [(450, "d")])

    def test_consume_nearest_rejects_when_all_entries_too_far(self) -> None:
        entries = deque([(100, "a"), (200, "b")])

        selected = consume_nearest(entries, target_ts=500, max_diff_ns=50)

        self.assertIsNone(selected)
        self.assertEqual(list(entries), [(100, "a"), (200, "b")])


if __name__ == "__main__":
    unittest.main()
