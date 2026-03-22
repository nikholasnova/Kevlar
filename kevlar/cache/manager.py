from __future__ import annotations

import logging
import threading
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, make_prompt_cache

from kevlar.cache.lru import CacheEntry, LRUCache
from kevlar.cache.persistence import SSDCacheStore
from kevlar.cache.prefix_matcher import _hash_tokens, find_longest_prefix_match

logger = logging.getLogger(__name__)


def _is_kvcache(cache_list: list) -> bool:
    return len(cache_list) > 0 and isinstance(cache_list[0], KVCache)


class CacheManager:
    def __init__(
        self,
        model: nn.Module,
        max_memory_caches: int = 5,
        ssd_cache_dir: Optional[str] = None,
        ssd_max_gb: float = 10.0,
    ):
        self.model = model
        self.memory = LRUCache(max_entries=max_memory_caches)
        self.ssd = SSDCacheStore(ssd_cache_dir, ssd_max_gb) if ssd_cache_dir else None
        self._save_lock = threading.Lock()

    def get_or_create_cache(self, tokens: mx.array) -> tuple[list, int]:
        """Returns (kv_cache, num_cached_tokens).

        The caller should prefill starting from num_cached_tokens.
        """
        entry, match_len = find_longest_prefix_match(tokens, self.memory)

        if entry and match_len > 0 and match_len == entry.num_tokens:
            # exact match -- reuse directly (works for all cache types)
            logger.info("Memory cache hit (exact): %d/%d tokens", match_len, tokens.size)
            cache = self.clone_cache(entry.cache_state)
            return cache, match_len

        if entry and match_len > 0:
            # partial prefix match -- trim KVCache layers, ArraysCache layers keep accumulated state
            logger.info("Memory cache hit (prefix): %d/%d tokens", match_len, tokens.size)
            cache = self.clone_cache(entry.cache_state)
            self._trim_cache(cache, match_len)
            return cache, match_len

        if self.ssd:
            token_hash = _hash_tokens(tokens)
            ssd_cache = self.ssd.load(token_hash)
            if ssd_cache is not None:
                logger.info("SSD cache hit: %s", token_hash)
                return ssd_cache, tokens.size

        cache = make_prompt_cache(self.model)
        return cache, 0

    def checkpoint(self, tokens: mx.array, cache: list):
        token_hash = _hash_tokens(tokens)
        entry = CacheEntry(
            token_hash=token_hash,
            cache_state=cache,
            num_tokens=tokens.size,
            tokens=tokens,
        )
        self.memory.put(entry)
        logger.debug("Cached %d tokens in memory (hash=%s)", tokens.size, token_hash)

        if self.ssd and tokens.size > 1000:
            self._save_to_ssd_background(token_hash, cache, tokens.size)

    def _save_to_ssd_background(self, token_hash: str, cache: list, num_tokens: int):
        def _save():
            with self._save_lock:
                self.ssd.save(token_hash, cache, num_tokens)

        t = threading.Thread(target=_save, daemon=True)
        t.start()

    def clone_cache(self, cache_state: list) -> list:
        """Deep copy cache to avoid mutating the stored version."""
        if _is_kvcache(cache_state):
            new_cache = make_prompt_cache(self.model)
            for new_c, old_c in zip(new_cache, cache_state):
                keys, values = old_c.state
                new_c.state = (mx.array(keys), mx.array(values))
            return new_cache

        # ArraysCache or other types -- make fresh cache and copy state
        new_cache = make_prompt_cache(self.model)
        for new_c, old_c in zip(new_cache, cache_state):
            old_state = old_c.state
            if isinstance(old_state, list):
                new_c.state = [mx.array(s) if s is not None else None for s in old_state]
            elif isinstance(old_state, tuple):
                new_c.state = tuple(mx.array(s) if s is not None else None for s in old_state)
            else:
                new_c.state = old_state
        return new_cache

    def _trim_cache(self, cache: list, target_len: int):
        """Trim KVCache to exactly target_len tokens."""
        for c in cache:
            if hasattr(c, "offset") and c.offset > target_len:
                keys, values = c.state
                c.state = (keys[:, :, :target_len, :], values[:, :, :target_len, :])
