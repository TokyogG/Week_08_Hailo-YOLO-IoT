from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class GateConfig:
    min_interval_ms: int = 250          # don’t publish faster than this
    min_score: float = 0.50             # only consider “real” detections
    publish_on_change: bool = True      # only publish if counts change


class EventGate:
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self._last_pub_ms = 0.0
        self._last_counts: Dict[int, int] = {}

    def _now_ms(self) -> float:
        return time.time() * 1000.0

    def should_publish(self, dets: List[Tuple[int, float]]) -> bool:
        """
        dets: list of (class_id, score)
        """
        now = self._now_ms()
        if now - self._last_pub_ms < self.cfg.min_interval_ms:
            return False

        # count by class for “stable eventing”
        counts: Dict[int, int] = {}
        for cls, score in dets:
            if score < self.cfg.min_score:
                continue
            counts[cls] = counts.get(cls, 0) + 1

        if not counts:
            # only publish “empty” once per interval (useful for dashboards)
            changed = self._last_counts != {}
        else:
            changed = counts != self._last_counts

        if self.cfg.publish_on_change and not changed:
            return False

        self._last_counts = counts
        self._last_pub_ms = now
        return True
