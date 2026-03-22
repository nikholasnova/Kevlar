from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    token_hash: str
    cache_state: list  # list of KVCache objects
    num_tokens: int
    tokens: mx.array
    byte_size: int = 0

    def __post_init__(self):
        if self.byte_size == 0:
            self.byte_size = self._estimate_bytes()

    def _estimate_bytes(self) -> int:
        total = 0
        for c in self.cache_state:
            if hasattr(c, "state"):
                keys, values = c.state
                total += keys.nbytes + values.nbytes
        return total


class LRUCache:
    def __init__(self, max_entries: int = 5):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def total_bytes(self) -> int:
        return sum(e.byte_size for e in self._cache.values())

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, entry: CacheEntry):
        with self._lock:
            if entry.token_hash in self._cache:
                self._cache.move_to_end(entry.token_hash)
                self._cache[entry.token_hash] = entry
            else:
                while len(self._cache) >= self.max_entries:
                    evicted_key, evicted = self._cache.popitem(last=False)
                    logger.debug("Evicted cache entry: %s (%d tokens, %.1f MB)", evicted_key, evicted.num_tokens, evicted.byte_size / 1e6)
                self._cache[entry.token_hash] = entry

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
