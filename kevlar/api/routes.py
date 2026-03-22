from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from kevlar.api.models import (
    ContentBlock,
    MessagesRequest,
    MessagesResponse,
    TextContent,
    ToolUseContent,
    Usage,
)
from kevlar.api.sse import (
    content_block_delta_event,
    content_block_start_event,
    content_block_stop_event,
    input_json_delta_event,
    message_delta_event,
    message_start_event,
    message_stop_event,
    ping_event,
    tool_use_block_start_event,
)
from kevlar.preprocessing.normalizer import normalize
from kevlar.utils.tokenizer import parse_tool_calls, request_to_token_ids, strip_thinking

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/messages")
async def create_message(request: Request, body: MessagesRequest = None):
    from kevlar.cli.display import console
    if body is None:
        raw = await request.body()
        console.print(f"  [red]Failed to parse request body[/red]")
        console.print(f"  [dim]{raw[:500]}[/dim]")
        return JSONResponse(status_code=400, content={"error": "bad request"})
    try:
        return await _handle_message(request, body)
    except Exception as e:
        console.print(f"  [red]ERROR: {type(e).__name__}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"type": "error", "error": {"type": "server_error", "message": str(e)}},
        )


async def _handle_message(request: Request, body: MessagesRequest):
    from kevlar.cli.display import console
    app = request.app
    engine = app.state.engine
    tokenizer = app.state.tokenizer
    config = app.state.config

    msg_count = len(body.messages)
    has_tools = bool(body.tools)
    console.print(f"  [dim]  msgs={msg_count} tools={has_tools} stream={body.stream} think={body.thinking} max_tokens={body.max_tokens}[/dim]")

    system_text = body.get_system_text()

    normalized = normalize(
        system=system_text,
        messages=[m.model_dump() for m in body.messages],
        enabled=config.enable_header_normalization,
    )

    temp_request = body.model_copy(update={
        "system": normalized.system,
    })

    # parse thinking config -- cap budget for local models
    # claude code sends budget_tokens=31999 which is way too much for local inference
    MAX_THINKING_BUDGET = 2000
    thinking_enabled = True
    thinking_budget = 0
    if body.thinking:
        thinking_type = body.thinking.get("type", "disabled")
        if thinking_type == "disabled":
            thinking_enabled = False
        elif thinking_type in ("enabled", "adaptive"):
            thinking_enabled = True
            thinking_budget = min(
                body.thinking.get("budget_tokens", 0),
                MAX_THINKING_BUDGET,
            )
    else:
        thinking_enabled = False

    prompt_tokens, thinking_enabled = request_to_token_ids(
        temp_request, tokenizer, normalized.system,
        enable_thinking=thinking_enabled,
    )

    temperature = body.temperature if body.temperature is not None else config.default_temperature
    top_p = body.top_p if body.top_p is not None else config.default_top_p
    max_tokens = min(body.max_tokens or config.default_max_tokens, config.default_max_tokens)

    if thinking_enabled and thinking_budget > 0:
        max_tokens = max_tokens + thinking_budget

    lock = app.state._inference_lock

    if body.stream:
        return EventSourceResponse(
            _stream_response(engine, prompt_tokens, body, temperature, top_p, max_tokens, thinking_enabled, lock),
            media_type="text/event-stream",
            ping=3,
        )
    else:
        async with lock:
            return await _complete_response(engine, prompt_tokens, body, temperature, top_p, max_tokens, thinking_enabled)


async def _stream_response(
    engine,
    prompt_tokens,
    body: MessagesRequest,
    temperature: float,
    top_p: float,
    max_tokens: int,
    thinking_enabled: bool = False,
    lock: asyncio.Lock = None,
) -> AsyncGenerator[dict, None]:
    # send initial events BEFORE acquiring lock so connection stays alive while waiting
    yield message_start_event(model=body.model, input_tokens=prompt_tokens.size)
    yield ping_event()

    if lock:
        await lock.acquire()
    try:
        yield content_block_start_event(index=0)

        full_text = ""
        output_tokens = 0
        finish_reason = "end_turn"
        thinking_done = not thinking_enabled
        think_buffer = ""

        async for result in engine.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=body.top_k,
            stop_sequences=body.stop_sequences,
        ):
            output_tokens += 1

            if not result.text:
                if result.finish_reason:
                    finish_reason = result.finish_reason
                continue

            full_text += result.text

            if not thinking_done:
                think_buffer += result.text
                if "</think>" in think_buffer:
                    thinking_done = True
                    after_think = think_buffer.split("</think>", 1)[1].lstrip()
                    if after_think:
                        yield content_block_delta_event(text=after_think, index=0)
                elif output_tokens % 20 == 0:
                    yield ping_event()
            else:
                yield content_block_delta_event(text=result.text, index=0)

            if result.finish_reason:
                finish_reason = result.finish_reason

        if not thinking_done and think_buffer:
            cleaned = strip_thinking(think_buffer)
            if cleaned:
                yield content_block_delta_event(text=cleaned, index=0)

        clean_text = strip_thinking(full_text) if thinking_enabled else full_text
        tool_calls = parse_tool_calls(clean_text) if body.tools else []
        if tool_calls:
            finish_reason = "tool_use"

        yield content_block_stop_event(index=0)

        for i, tc in enumerate(tool_calls, start=1):
            yield tool_use_block_start_event(index=i, tool_id=tc.id, name=tc.name)
            yield input_json_delta_event(partial_json=json.dumps(tc.input), index=i)
            yield content_block_stop_event(index=i)

        yield message_delta_event(stop_reason=finish_reason, output_tokens=output_tokens)
        yield message_stop_event()
    finally:
        if lock:
            lock.release()


async def _complete_response(
    engine,
    prompt_tokens,
    body: MessagesRequest,
    temperature: float,
    top_p: float,
    max_tokens: int,
    thinking_enabled: bool = False,
) -> JSONResponse:
    full_text = ""
    output_tokens = 0
    finish_reason = "end_turn"

    async for result in engine.generate(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=body.top_k,
        stop_sequences=body.stop_sequences,
    ):
        if result.text:
            full_text += result.text
        output_tokens += 1
        if result.finish_reason:
            finish_reason = result.finish_reason

    clean_text = strip_thinking(full_text) if thinking_enabled else full_text
    content: list[ContentBlock] = [TextContent(text=clean_text)]

    tool_calls = parse_tool_calls(clean_text) if body.tools else []
    if tool_calls:
        finish_reason = "tool_use"
        content.extend(tool_calls)

    response = MessagesResponse(
        model=body.model,
        content=content,
        stop_reason=finish_reason,
        usage=Usage(
            input_tokens=prompt_tokens.size,
            output_tokens=output_tokens,
        ),
    )

    return JSONResponse(content=response.model_dump())
