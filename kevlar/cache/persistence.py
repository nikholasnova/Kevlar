from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SSDCacheStore:
    def __init__(self, cache_dir: str, max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1e9)

    def _entry_path(self, token_hash: str) -> Path:
        return self.cache_dir / token_hash

    def _metadata_path(self, token_hash: str) -> Path:
        return self._entry_path(token_hash) / "metadata.json"

    def save(self, token_hash: str, cache: list, num_tokens: int):
        try:
            from mlx_lm.models.cache import save_prompt_cache

            entry_dir = self._entry_path(token_hash)
            entry_dir.mkdir(parents=True, exist_ok=True)

            cache_file = str(entry_dir / "cache.safetensors")
            save_prompt_cache(cache_file, cache)

            metadata = {
                "token_hash": token_hash,
                "num_tokens": num_tokens,
                "timestamp": time.time(),
            }
            with open(self._metadata_path(token_hash), "w") as f:
                json.dump(metadata, f)

            logger.debug("Saved cache to SSD: %s (%d tokens)", token_hash, num_tokens)
            self._evict_if_needed()

        except Exception:
            logger.exception("Failed to save cache to SSD: %s", token_hash)

    def load(self, token_hash: str) -> Optional[list]:
        try:
            from mlx_lm.models.cache import load_prompt_cache

            entry_dir = self._entry_path(token_hash)
            cache_file = str(entry_dir / "cache.safetensors")

            if not os.path.exists(cache_file):
                return None

            cache = load_prompt_cache(cache_file)

            # update access time
            meta_path = self._metadata_path(token_hash)
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
                metadata["timestamp"] = time.time()
                with open(meta_path, "w") as f:
                    json.dump(metadata, f)

            logger.debug("Loaded cache from SSD: %s", token_hash)
            return cache

        except Exception:
            logger.exception("Failed to load cache from SSD: %s", token_hash)
            return None

    def _evict_if_needed(self):
        entries = []
        for entry_dir in self.cache_dir.iterdir():
            if not entry_dir.is_dir():
                continue
            meta_path = entry_dir / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                size = sum(f.stat().st_size for f in entry_dir.rglob("*") if f.is_file())
                entries.append((meta.get("timestamp", 0), size, entry_dir))
            except Exception:
                continue

        total = sum(e[1] for e in entries)
        if total <= self.max_size_bytes:
            return

        entries.sort(key=lambda x: x[0])
        while total > self.max_size_bytes and entries:
            ts, size, path = entries.pop(0)
            shutil.rmtree(path, ignore_errors=True)
            total -= size
            logger.debug("Evicted SSD cache: %s", path.name)

    def has(self, token_hash: str) -> bool:
        return (self._entry_path(token_hash) / "cache.safetensors").exists()

    def clear(self):
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared SSD cache: %s", self.cache_dir)
