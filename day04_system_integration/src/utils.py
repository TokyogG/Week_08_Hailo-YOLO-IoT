#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


def safe_json_loads(payload: bytes) -> Optional[dict[str, Any]]:
    """
    Return dict if payload is valid JSON object, else None.
    """
    try:
        obj = json.loads(payload.decode("utf-8", errors="replace"))
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def fmt_ts_ms(ts_ms: int) -> str:
    """
    Format milliseconds unix timestamp into a readable string (local time).
    """
    try:
        sec = ts_ms / 1000.0
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sec))
    except Exception:
        return str(ts_ms)


@dataclass
class RollingRate:
    """
    Simple rolling rate estimator:
    call .tick() each event, then .rate_per_sec()
    """
    window_s: float = 5.0

    def __post_init__(self) -> None:
        self._events: list[float] = []

    def tick(self) -> None:
        t = time.time()
        self._events.append(t)
        cutoff = t - self.window_s
        # prune
        i = 0
        for i, ts in enumerate(self._events):
            if ts >= cutoff:
                break
        else:
            # all old
            self._events = []
            return
        self._events = self._events[i:]

    def rate_per_sec(self) -> float:
        if len(self._events) < 2:
            return 0.0
        dt = self._events[-1] - self._events[0]
        if dt <= 0:
            return 0.0
        return (len(self._events) - 1) / dt
