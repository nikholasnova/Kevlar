from __future__ import annotations

import logging
import time

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    return _client


def is_haiku_request(model: str) -> bool:
    return "haiku" in model.lower()


async def proxy_messages(body_bytes: bytes, haiku_base_url: str, stream: bool = False):
    client = _get_client()
    url = f"{haiku_base_url}/v1/messages"
    headers = {"content-type": "application/json"}

    t0 = time.monotonic()
    logger.info("Proxying /v1/messages to %s (stream=%s, %d bytes)", haiku_base_url, stream, len(body_bytes))

    try:
        if stream:
            req = client.build_request("POST", url, content=body_bytes, headers=headers)
            resp = await client.send(req, stream=True)
            logger.info("Haiku stream connected: %d in %.1fms", resp.status_code, (time.monotonic() - t0) * 1000)

            async def passthrough():
                async for chunk in resp.aiter_bytes():
                    yield chunk
                await resp.aclose()

            return StreamingResponse(
                passthrough(),
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "text/event-stream"),
            )
        else:
            resp = await client.post(url, content=body_bytes, headers=headers)
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("Haiku response: %d in %.1fms", resp.status_code, elapsed)
            return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("Haiku subprocess unreachable at %s (%.1fms)", haiku_base_url, elapsed)
        return JSONResponse(status_code=503, content={
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Haiku model unavailable"},
        })
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        logger.error("Haiku proxy error: %s: %s (%.1fms)", type(e).__name__, e, elapsed)
        return JSONResponse(status_code=502, content={
            "type": "error",
            "error": {"type": "api_error", "message": f"Haiku proxy error: {e}"},
        })


async def proxy_count_tokens(body_bytes: bytes, haiku_base_url: str):
    client = _get_client()
    url = f"{haiku_base_url}/v1/messages/count_tokens"

    t0 = time.monotonic()
    logger.info("Proxying /v1/messages/count_tokens to %s (%d bytes)", haiku_base_url, len(body_bytes))

    try:
        resp = await client.post(url, content=body_bytes, headers={"content-type": "application/json"})
        elapsed = (time.monotonic() - t0) * 1000
        logger.info("Haiku count_tokens: %d in %.1fms", resp.status_code, elapsed)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except httpx.ConnectError:
        logger.warning("Haiku subprocess unreachable at %s for count_tokens", haiku_base_url)
        return JSONResponse(status_code=503, content={
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Haiku model unavailable"},
        })
    except Exception as e:
        logger.error("Haiku count_tokens proxy error: %s: %s", type(e).__name__, e)
        return JSONResponse(status_code=502, content={
            "type": "error",
            "error": {"type": "api_error", "message": f"Haiku proxy error: {e}"},
        })
