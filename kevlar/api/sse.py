from __future__ import annotations

import json
from typing import Any

from kevlar.api.models import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJsonDelta,
    MessageDeltaBody,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageStartEvent,
    MessageStopEvent,
    MessagesResponse,
    PingEvent,
    TextContent,
    TextDelta,
    SignatureDelta,
    ThinkingContent,
    ThinkingDelta,
    ToolUseContent,
    Usage,
)


def _sse(event_type: str, data: Any) -> dict:
    if isinstance(data, str):
        payload = data
    else:
        payload = data.model_dump_json()
    return {"event": event_type, "data": payload}


def message_start_event(
    model: str,
    input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> dict:
    response = MessagesResponse(
        model=model,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=0,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        ),
    )
    event = MessageStartEvent(message=response)
    return _sse("message_start", event)


def ping_event() -> dict:
    return _sse("ping", PingEvent())


def content_block_start_event(index: int = 0) -> dict:
    event = ContentBlockStartEvent(
        index=index,
        content_block=TextContent(text=""),
    )
    return _sse("content_block_start", event)


def thinking_block_start_event(index: int = 0) -> dict:
    event = ContentBlockStartEvent(
        index=index,
        content_block=ThinkingContent(thinking=""),
    )
    return _sse("content_block_start", event)


def thinking_delta_event(thinking: str, index: int = 0) -> dict:
    event = ContentBlockDeltaEvent(
        index=index,
        delta=ThinkingDelta(thinking=thinking),
    )
    return _sse("content_block_delta", event)


def signature_delta_event(signature: str, index: int = 0) -> dict:
    event = ContentBlockDeltaEvent(
        index=index,
        delta=SignatureDelta(signature=signature),
    )
    return _sse("content_block_delta", event)


def tool_use_block_start_event(index: int, tool_id: str, name: str) -> dict:
    event = ContentBlockStartEvent(
        index=index,
        content_block=ToolUseContent(id=tool_id, name=name, input={}),
    )
    return _sse("content_block_start", event)


def content_block_delta_event(text: str, index: int = 0) -> dict:
    event = ContentBlockDeltaEvent(
        index=index,
        delta=TextDelta(text=text),
    )
    return _sse("content_block_delta", event)


def input_json_delta_event(partial_json: str, index: int = 0) -> dict:
    event = ContentBlockDeltaEvent(
        index=index,
        delta=InputJsonDelta(partial_json=partial_json),
    )
    return _sse("content_block_delta", event)


def content_block_stop_event(index: int = 0) -> dict:
    event = ContentBlockStopEvent(index=index)
    return _sse("content_block_stop", event)


def message_delta_event(
    stop_reason: str,
    output_tokens: int = 0,
    stop_sequence: str = None,
) -> dict:
    event = MessageDeltaEvent(
        delta=MessageDeltaBody(stop_reason=stop_reason, stop_sequence=stop_sequence),
        usage=MessageDeltaUsage(output_tokens=output_tokens),
    )
    return _sse("message_delta", event)


def message_stop_event() -> dict:
    return _sse("message_stop", MessageStopEvent())
