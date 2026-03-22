from __future__ import annotations

import json
from pathlib import Path

KEVLAR_DIR = Path.home() / ".kevlar"
MODELS_FILE = KEVLAR_DIR / "models.json"

DEFAULT_MODELS = [
    "mlx-community/Qwen3.5-122B-A10B-4bit",
]


def _ensure_dir():
    KEVLAR_DIR.mkdir(parents=True, exist_ok=True)


def load_models() -> list[str]:
    _ensure_dir()
    if not MODELS_FILE.exists():
        save_models(DEFAULT_MODELS)
        return list(DEFAULT_MODELS)
    try:
        with open(MODELS_FILE) as f:
            models = json.load(f)
        if not isinstance(models, list) or not models:
            return list(DEFAULT_MODELS)
        return models
    except (json.JSONDecodeError, IOError):
        return list(DEFAULT_MODELS)


def save_models(models: list[str]):
    _ensure_dir()
    with open(MODELS_FILE, "w") as f:
        json.dump(models, f, indent=2)


def add_model(model_id: str):
    models = load_models()
    if model_id not in models:
        models.append(model_id)
        save_models(models)


def remove_model(model_id: str):
    models = load_models()
    if model_id in models:
        models.remove(model_id)
        save_models(models)
