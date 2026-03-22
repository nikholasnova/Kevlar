import pytest

from kevlar.cache.lru import CacheEntry, LRUCache
from kevlar.cache.prefix_matcher import _hash_tokens, find_longest_prefix_match


class FakeKVCache:
    def __init__(self, offset=0):
        self.offset = offset
        self._keys = None
        self._values = None

    @property
    def state(self):
        return (self._keys, self._values)

    @state.setter
    def state(self, kv):
        self._keys, self._values = kv
        if self._keys is not None:
            self.offset = self._keys.shape[2] if len(self._keys.shape) > 2 else 0


class TestLRUCache:
    def test_put_and_get(self):
        lru = LRUCache(max_entries=3)
        entry = CacheEntry(token_hash="abc", cache_state=[], num_tokens=100, tokens=None, byte_size=1000)
        lru.put(entry)
        assert lru.get("abc") is entry

    def test_miss(self):
        lru = LRUCache(max_entries=3)
        assert lru.get("missing") is None

    def test_eviction(self):
        lru = LRUCache(max_entries=2)
        e1 = CacheEntry(token_hash="a", cache_state=[], num_tokens=10, tokens=None, byte_size=100)
        e2 = CacheEntry(token_hash="b", cache_state=[], num_tokens=20, tokens=None, byte_size=200)
        e3 = CacheEntry(token_hash="c", cache_state=[], num_tokens=30, tokens=None, byte_size=300)
        lru.put(e1)
        lru.put(e2)
        lru.put(e3)
        assert lru.get("a") is None
        assert lru.get("b") is not None
        assert lru.get("c") is not None

    def test_access_promotes(self):
        lru = LRUCache(max_entries=2)
        e1 = CacheEntry(token_hash="a", cache_state=[], num_tokens=10, tokens=None, byte_size=100)
        e2 = CacheEntry(token_hash="b", cache_state=[], num_tokens=20, tokens=None, byte_size=200)
        lru.put(e1)
        lru.put(e2)
        lru.get("a")  # promote a
        e3 = CacheEntry(token_hash="c", cache_state=[], num_tokens=30, tokens=None, byte_size=300)
        lru.put(e3)
        assert lru.get("a") is not None
        assert lru.get("b") is None

    def test_update_existing(self):
        lru = LRUCache(max_entries=3)
        e1 = CacheEntry(token_hash="a", cache_state=[], num_tokens=10, tokens=None, byte_size=100)
        lru.put(e1)
        e2 = CacheEntry(token_hash="a", cache_state=[], num_tokens=20, tokens=None, byte_size=200)
        lru.put(e2)
        assert len(lru) == 1
        assert lru.get("a").num_tokens == 20

    def test_clear(self):
        lru = LRUCache(max_entries=3)
        e = CacheEntry(token_hash="a", cache_state=[], num_tokens=10, tokens=None, byte_size=100)
        lru.put(e)
        lru.clear()
        assert len(lru) == 0


class TestPrefixMatcher:
    def _make_entry(self, token_list):
        import mlx.core as mx
        tokens = mx.array(token_list)
        return CacheEntry(
            token_hash=_hash_tokens(tokens),
            cache_state=[],
            num_tokens=len(token_list),
            tokens=tokens,
            byte_size=100,
        )

    def test_exact_match(self):
        import mlx.core as mx
        lru = LRUCache(max_entries=5)
        entry = self._make_entry([1, 2, 3, 4, 5])
        lru.put(entry)
        tokens = mx.array([1, 2, 3, 4, 5])
        match, length = find_longest_prefix_match(tokens, lru)
        assert match is not None
        assert length == 5

    def test_partial_match(self):
        import mlx.core as mx
        lru = LRUCache(max_entries=5)
        entry = self._make_entry([1, 2, 3, 4, 5])
        lru.put(entry)
        tokens = mx.array([1, 2, 3, 99, 100])
        match, length = find_longest_prefix_match(tokens, lru)
        assert match is not None
        assert length == 3

    def test_no_match(self):
        import mlx.core as mx
        lru = LRUCache(max_entries=5)
        entry = self._make_entry([1, 2, 3])
        lru.put(entry)
        tokens = mx.array([99, 100, 101])
        match, length = find_longest_prefix_match(tokens, lru)
        assert length == 0

    def test_best_match_selected(self):
        import mlx.core as mx
        lru = LRUCache(max_entries=5)
        lru.put(self._make_entry([1, 2, 3]))
        lru.put(self._make_entry([1, 2, 3, 4, 5]))
        tokens = mx.array([1, 2, 3, 4, 5, 6, 7])
        match, length = find_longest_prefix_match(tokens, lru)
        assert length == 5
