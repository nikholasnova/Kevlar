from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# --- Request models ---

class TextContent(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["text"] = "text"
    text: str


class ToolUseContent(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultContent(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, list[TextContent]] = ""
    is_error: bool = False


ContentBlock = Union[TextContent, ToolUseContent, ToolResultContent]


class Message(BaseModel):
    model_config = {"extra": "allow"}
    role: Literal["user", "assistant"]
    content: Union[str, list[ContentBlock]]


class ToolInputSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    model_config = {"extra": "allow"}
    name: str
    description: str = ""
    input_schema: ToolInputSchema = Field(default_factory=ToolInputSchema)
    type: Optional[str] = None


class SystemBlock(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["text"] = "text"
    text: str


class MessagesRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str
    max_tokens: int = 4096
    messages: list[Message]
    system: Optional[Union[str, list[SystemBlock]]] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[dict[str, Any]] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    thinking: Optional[dict[str, Any]] = None

    def get_system_text(self) -> str:
        if self.system is None:
            return ""
        if isinstance(self.system, str):
            return self.system
        return "\n\n".join(block.text for block in self.system)


# --- Response models ---

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock] = Field(default_factory=list)
    model: str = ""
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)


# --- SSE event models ---

class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: MessagesResponse


class ContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJsonDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Union[TextDelta, InputJsonDelta]


class ContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaUsage(BaseModel):
    output_tokens: int = 0


class MessageDeltaBody(BaseModel):
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: MessageDeltaBody
    usage: MessageDeltaUsage


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: dict[str, str]
