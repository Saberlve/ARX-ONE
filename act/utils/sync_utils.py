from collections import deque
from typing import Any, Deque, Optional, Tuple


Entry = Tuple[int, Any]


def consume_nearest(
    entries: Deque[Entry],
    target_ts: int,
    max_diff_ns: Optional[int] = None,
) -> Optional[Entry]:
    """Return the entry closest to target_ts and consume it plus older entries.

    The deque is assumed to be ordered from oldest to newest.
    If the closest entry is farther than max_diff_ns, leave the deque unchanged
    and return None.
    """
    if not entries:
        return None

    best_idx = -1
    best_diff = None
    for idx, (ts, _) in enumerate(entries):
        diff = abs(ts - target_ts)
        if best_diff is None or diff < best_diff:
            best_idx = idx
            best_diff = diff

    if best_idx < 0:
        return None
    if max_diff_ns is not None and best_diff is not None and best_diff > max_diff_ns:
        return None

    for _ in range(best_idx):
        entries.popleft()
    return entries.popleft()
