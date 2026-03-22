from __future__ import annotations

import hashlib
import logging
from typing import Optional

import mlx.core as mx

from kevlar.cache.lru import CacheEntry, LRUCache

logger = logging.getLogger(__name__)


def _hash_tokens(tokens: mx.array, length: Optional[int] = None) -> str:
    """Hash a token sequence for cache lookup."""
    if length is not None:
        tokens = tokens[:length]
    token_bytes = tokens.tolist()
    return hashlib.sha256(str(token_bytes).encode()).hexdigest()[:16]


def find_longest_prefix_match(
    tokens: mx.array,
    lru: LRUCache,
) -> tuple[Optional[CacheEntry], int]:
    """Find the cache entry with the longest matching token prefix.

    Returns (entry, match_length). If no match, returns (None, 0).
    """
    best_entry = None
    best_match_len = 0

    for key in lru.keys():
        entry = lru.get(key)
        if entry is None:
            continue

        cached_tokens = entry.tokens
        min_len = min(tokens.size, cached_tokens.size)

        if min_len == 0:
            continue

        # compare token by token to find divergence point
        match_len = 0
        t_list = tokens[:min_len].tolist()
        c_list = cached_tokens[:min_len].tolist()
        for a, b in zip(t_list, c_list):
            if a != b:
                break
            match_len += 1

        if match_len > best_match_len:
            best_match_len = match_len
            best_entry = entry

    if best_entry:
        logger.debug(
            "Prefix match: %d/%d tokens (%.1f%%)",
            best_match_len,
            tokens.size,
            100 * best_match_len / tokens.size,
        )

    return best_entry, best_match_len
