from typing import Optional

import mlx.core as mx


def sample(
    logits: mx.array,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
) -> mx.array:
    if temperature == 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    if min_p is not None and min_p > 0:
        probs = mx.softmax(logits, axis=-1)
        max_prob = mx.max(probs, axis=-1, keepdims=True)
        mask = probs < (min_p * max_prob)
        logits = mx.where(mask, mx.array(float("-inf")), logits)

    if top_k is not None and top_k > 0:
        top_values = mx.topk(logits, k=min(top_k, logits.shape[-1]))
        threshold = top_values[:, -1:]
        logits = mx.where(logits < threshold, mx.array(float("-inf")), logits)

    if top_p is not None and 0 < top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative = mx.cumsum(sorted_probs, axis=-1)
        cutoff_mask = cumulative - sorted_probs > top_p
        sorted_probs = mx.where(cutoff_mask, mx.array(0.0), sorted_probs)
        probs = mx.zeros_like(probs)
        probs = probs.at[mx.arange(probs.shape[0])[:, None], sorted_indices].add(sorted_probs)
        return mx.random.categorical(mx.log(probs + 1e-10))

    return mx.random.categorical(logits)
