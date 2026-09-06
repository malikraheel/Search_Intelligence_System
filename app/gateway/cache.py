from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any


def request_key(model: str, messages: list[dict[str, Any]], tools: Any, response_format: Any) -> str:
    rf = getattr(response_format, "__name__", None) or response_format
    blob = json.dumps({"m": model, "msgs": messages, "tools": tools, "rf": str(rf)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    """Thread-safe TTL + LRU cache. Only deterministic (temperature 0) requests are cached."""

    def __init__(self, ttl_s: int = 600, max_items: int = 512, clock=time.monotonic):
        self._ttl = ttl_s
        self._max = max_items
        self._clock = clock
        self._lock = threading.Lock()
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None or self._clock() - item[0] > self._ttl:
                if item is not None:
                    del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (self._clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
