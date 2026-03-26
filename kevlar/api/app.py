from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import mlx.core as mx

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from kevlar.api.routes import router
from kevlar.cache.manager import CacheManager
from kevlar.cli.display import print_model_loading, print_ready, print_request_stats
from kevlar.config import KevlarConfig
from kevlar.engine.generator import InferenceEngine
from kevlar.engine.loader import ModelLoader

logger = logging.getLogger(__name__)


def create_app(config: KevlarConfig) -> FastAPI:
    app = FastAPI(title="Kevlar", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from fastapi.exceptions import RequestValidationError
    from kevlar.cli.display import console

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request, exc):
        console.print(f"  [red]Validation error: {exc}[/red]")
        raw = await request.body()
        console.print(f"  [dim]Body: {raw[:1000]}[/dim]")
        return JSONResponse(
            status_code=422,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}},
        )

    @app.exception_handler(Exception)
    async def general_handler(request, exc):
        console.print(f"  [red]Unhandled error: {type(exc).__name__}: {exc}[/red]")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"type": "error", "error": {"type": "server_error", "message": str(exc)}},
        )

    app.state.config = config
    app.state._inference_lock = asyncio.Lock()
    app.state._start_time = time.time()

    @app.on_event("startup")
    async def startup():
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        logging.getLogger("kevlar").setLevel(logging.INFO)

        loader = ModelLoader(config.model_path)
        with print_model_loading():
            model, tokenizer = loader.load()

        # isolate SSD cache per model to prevent cross-model shape mismatches
        model_slug = config.model_path.replace("/", "--")
        ssd_dir = str(Path(config.ssd_cache_dir) / model_slug) if config.ssd_cache_dir else None

        cache_manager = CacheManager(
            model=model,
            max_memory_caches=config.max_memory_caches,
            ssd_cache_dir=ssd_dir,
            ssd_max_gb=config.ssd_cache_max_gb,
        )

        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            cache_manager=cache_manager,
            prefill_step_size=config.prefill_step_size,
            on_complete=print_request_stats,
        )

        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.cache_manager = cache_manager
        app.state.engine = engine

        # warmup: run a dummy forward pass to JIT-compile Metal kernels
        from mlx_lm.models.cache import make_prompt_cache
        warmup_cache = make_prompt_cache(model)
        warmup_tokens = mx.array([[tokenizer.eos_token_id or 0]])
        model(warmup_tokens, cache=warmup_cache)
        mx.eval([c.state for c in warmup_cache])

        print_ready(config)

    @app.middleware("http")
    async def log_requests(request, call_next):
        from kevlar.cli.display import console
        if request.url.path == "/v1/messages":
            console.print(f"  [dim]-> POST /v1/messages[/dim]")
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            console.print(f"  [red]Request error: {e}[/red]")
            raise

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/status")
    async def status():
        uptime = time.time() - app.state._start_time
        cache_mgr: CacheManager = app.state.cache_manager

        ssd_entries = 0
        if cache_mgr.ssd:
            ssd_entries = sum(
                1 for p in cache_mgr.ssd.cache_dir.iterdir()
                if p.is_dir() and (p / "metadata.json").exists()
            )

        return {
            "status": "ok",
            "model": config.model_path,
            "uptime_s": round(uptime, 1),
            "cache": {
                "memory_entries": len(cache_mgr.memory),
                "memory_bytes": cache_mgr.memory.total_bytes,
                "ssd_dir": config.ssd_cache_dir,
                "ssd_entries": ssd_entries,
            },
        }

    app.include_router(router)

    return app
