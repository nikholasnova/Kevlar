from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache

from kevlar.cache.manager import CacheManager
from kevlar.engine.sampler import sample

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    token_id: int
    text: str
    is_eos: bool
    finish_reason: Optional[str] = None
    stop_sequence: Optional[str] = None


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    cache_hit_tokens: int = 0

    @property
    def prefill_tps(self) -> float:
        if self.prefill_time_s == 0:
            return 0
        return (self.prompt_tokens - self.cache_hit_tokens) / self.prefill_time_s

    @property
    def decode_tps(self) -> float:
        if self.decode_time_s == 0:
            return 0
        return self.completion_tokens / self.decode_time_s


class InferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        cache_manager: CacheManager,
        prefill_step_size: int = 2048,
        on_complete: Optional[callable] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_manager = cache_manager
        self.prefill_step_size = prefill_step_size
        self.on_complete = on_complete
        self._eos_token_ids = self._get_eos_tokens()
        self.last_stats: Optional[GenerationStats] = None

    def _get_eos_tokens(self) -> set[int]:
        eos_ids = set()
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        if hasattr(self.model, "config") and hasattr(self.model.config, "eos_token_id"):
            eos = self.model.config.eos_token_id
            if isinstance(eos, list):
                eos_ids.update(eos)
            elif eos is not None:
                eos_ids.add(eos)
        # some models store extra stop tokens only in added_tokens
        if hasattr(self.tokenizer, "added_tokens_encoder"):
            for token_str, token_id in self.tokenizer.added_tokens_encoder.items():
                if any(marker in token_str for marker in ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|endoftext|>", "</s>"]):
                    eos_ids.add(token_id)
        return eos_ids

    async def _prefill(
        self,
        tokens: mx.array,
        cache: list,
        start_pos: int,
    ) -> mx.array:
        remaining = tokens[start_pos:]
        if remaining.size == 0:
            # exact cache hit -- still need logits from last token to start generation
            last_token = tokens[-1:]
            logits = self.model(last_token[None], cache=cache)
            mx.eval([c.state for c in cache])
            return logits

        total = remaining.size
        logger.info("Prefilling %d tokens (cached: %d)", total, start_pos)
        t0 = time.perf_counter()

        logits = None
        for i in range(0, total, self.prefill_step_size):
            chunk = remaining[i : i + self.prefill_step_size]
            logits = self.model(chunk[None], cache=cache)
            mx.eval([c.state for c in cache])
            await asyncio.sleep(0)

        elapsed = time.perf_counter() - t0
        logger.info("Prefill done: %.2fs (%.0f tok/s)", elapsed, total / elapsed if elapsed > 0 else 0)
        return logits

    async def generate(
        self,
        prompt_tokens: mx.array,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> AsyncGenerator[GenerationResult, None]:
        stats = GenerationStats(prompt_tokens=prompt_tokens.size)

        cache, cache_hit_len = self.cache_manager.get_or_create_cache(prompt_tokens)
        stats.cache_hit_tokens = cache_hit_len

        t0 = time.perf_counter()
        logits = await self._prefill(prompt_tokens, cache, cache_hit_len)
        stats.prefill_time_s = time.perf_counter() - t0

        if logits is None:
            return

        # snapshot prompt-only cache state before decode mutates it
        prompt_cache = self.cache_manager.clone_cache(cache)

        token_id = sample(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        mx.eval(token_id)

        generated_text = ""
        stop_seqs = stop_sequences or []
        decode_start = time.perf_counter()
        detokenizer = self.tokenizer.detokenizer

        for i in range(max_tokens):
            tid = token_id.item()
            stats.completion_tokens += 1

            is_eos = tid in self._eos_token_ids
            if not is_eos:
                detokenizer.add_token(tid)
            piece = detokenizer.last_segment if not is_eos else ""
            generated_text += piece

            finish_reason = None
            matched_stop = None
            if is_eos:
                finish_reason = "end_turn"
            else:
                for s in stop_seqs:
                    if generated_text.endswith(s):
                        finish_reason = "stop_sequence"
                        matched_stop = s
                        break
            if not finish_reason and i == max_tokens - 1:
                finish_reason = "max_tokens"

            yield GenerationResult(
                token_id=tid,
                text=piece,
                is_eos=is_eos,
                finish_reason=finish_reason,
                stop_sequence=matched_stop,
            )

            if finish_reason:
                break

            logits = self.model(token_id.reshape(1, 1), cache=cache)
            token_id = sample(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            mx.eval(token_id)
            await asyncio.sleep(0)

        detokenizer.finalize()
        remaining_text = detokenizer.last_segment
        if remaining_text:
            generated_text += remaining_text
            yield GenerationResult(token_id=0, text=remaining_text, is_eos=False)

        stats.decode_time_s = time.perf_counter() - decode_start

        self.cache_manager.checkpoint(prompt_tokens, prompt_cache)
        self.last_stats = stats

        if self.on_complete:
            self.on_complete(stats)
