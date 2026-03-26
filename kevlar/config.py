from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KevlarConfig:
    model_path: str = "mlx-community/Qwen3.5-122B-A10B-4bit"
    host: str = "127.0.0.1"
    port: int = 8080

    max_memory_caches: int = 5
    ssd_cache_dir: str = str(Path.home() / ".kevlar" / "cache")
    ssd_cache_max_gb: float = 10.0

    prefill_step_size: int = 4096
    default_max_tokens: int = 131072
    default_temperature: float = 0.7
    default_top_p: float = 0.9

    enable_header_normalization: bool = True
