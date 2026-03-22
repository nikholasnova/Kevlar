import logging

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load

logger = logging.getLogger(__name__)


def _count_parameters(params) -> int:
    total = 0
    if isinstance(params, dict):
        for v in params.values():
            total += _count_parameters(v)
    elif isinstance(params, (list, tuple)):
        for v in params:
            total += _count_parameters(v)
    elif isinstance(params, mx.array):
        total += params.size
    return total


class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load(self) -> tuple[nn.Module, object]:
        logger.info("Loading model: %s", self.model_path)
        self.model, self.tokenizer = load(self.model_path)
        num_params = _count_parameters(self.model.parameters())
        logger.info("Model loaded. Parameters: %.1fB", num_params / 1e9)
        return self.model, self.tokenizer
