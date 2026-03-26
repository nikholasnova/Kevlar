import json

import pytest

from kevlar.api.models import (
    ContentBlockDeltaEvent,
    Message,
    MessagesRequest,
    MessagesResponse,
    SignatureDelta,
    SystemBlock,
    TextContent,
    TextDelta,
    ThinkingContent,
    ThinkingDelta,
    ToolDefinition,
    ToolInputSchema,
    ToolResultContent,
    ToolUseContent,
    Usage,
)


class TestMessagesRequest:
    def test_minimal(self):
        req = MessagesRequest(
            model="test",
            messages=[Message(role="user", content="hello")],
        )
        assert req.model == "test"
        assert req.stream is False

    def test_with_tools(self):
        req = MessagesRequest(
            model="test",
            messages=[Message(role="user", content="hello")],
            tools=[ToolDefinition(
                name="read_file",
                description="Read a file",
                input_schema=ToolInputSchema(
                    properties={"path": {"type": "string"}},
                    required=["path"],
                ),
            )],
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "read_file"

    def test_system_string(self):
        req = MessagesRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
            system="You are helpful.",
        )
        assert req.get_system_text() == "You are helpful."

    def test_system_blocks(self):
        req = MessagesRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
            system=[
                SystemBlock(text="Part 1"),
                SystemBlock(text="Part 2"),
            ],
        )
        assert req.get_system_text() == "Part 1\n\nPart 2"

    def test_system_none(self):
        req = MessagesRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
        )
        assert req.get_system_text() == ""


class TestMessageContent:
    def test_text_content(self):
        msg = Message(role="user", content=[TextContent(text="hello")])
        assert msg.content[0].text == "hello"

    def test_tool_use_content(self):
        msg = Message(
            role="assistant",
            content=[ToolUseContent(id="toolu_123", name="read_file", input={"path": "/tmp"})],
        )
        assert msg.content[0].name == "read_file"

    def test_tool_result_content(self):
        msg = Message(
            role="user",
            content=[ToolResultContent(tool_use_id="toolu_123", content="file contents")],
        )
        assert msg.content[0].tool_use_id == "toolu_123"

    def test_mixed_content(self):
        msg = Message(
            role="assistant",
            content=[
                TextContent(text="Let me read that file."),
                ToolUseContent(id="toolu_abc", name="read", input={"path": "/x"}),
            ],
        )
        assert len(msg.content) == 2


class TestMessagesResponse:
    def test_serialization(self):
        resp = MessagesResponse(
            model="test-model",
            content=[TextContent(text="Hello!")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        data = resp.model_dump()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["stop_reason"] == "end_turn"
        assert data["usage"]["input_tokens"] == 10

    def test_id_generated(self):
        r1 = MessagesResponse()
        r2 = MessagesResponse()
        assert r1.id != r2.id
        assert r1.id.startswith("msg_")


class TestThinkingContent:
    def test_serialization(self):
        tc = ThinkingContent(thinking="Let me reason about this")
        data = tc.model_dump()
        assert data["type"] == "thinking"
        assert data["thinking"] == "Let me reason about this"

    def test_in_message_content(self):
        msg = Message(
            role="assistant",
            content=[
                ThinkingContent(thinking="reasoning"),
                TextContent(text="answer"),
            ],
        )
        assert len(msg.content) == 2
        assert msg.content[0].type == "thinking"

    def test_in_response(self):
        resp = MessagesResponse(
            model="test",
            content=[
                ThinkingContent(thinking="step by step"),
                TextContent(text="the answer"),
            ],
            stop_reason="end_turn",
        )
        data = resp.model_dump()
        assert data["content"][0]["type"] == "thinking"
        assert data["content"][1]["type"] == "text"


class TestSSEModels:
    def test_content_block_delta(self):
        event = ContentBlockDeltaEvent(
            index=0,
            delta=TextDelta(text="Hello"),
        )
        data = event.model_dump()
        assert data["type"] == "content_block_delta"
        assert data["delta"]["text"] == "Hello"

    def test_thinking_delta(self):
        event = ContentBlockDeltaEvent(
            index=0,
            delta=ThinkingDelta(thinking="reasoning step"),
        )
        data = event.model_dump()
        assert data["type"] == "content_block_delta"
        assert data["delta"]["type"] == "thinking_delta"
        assert data["delta"]["thinking"] == "reasoning step"

    def test_signature_delta(self):
        event = ContentBlockDeltaEvent(
            index=0,
            delta=SignatureDelta(signature="kevlar-local"),
        )
        data = event.model_dump()
        assert data["type"] == "content_block_delta"
        assert data["delta"]["type"] == "signature_delta"
        assert data["delta"]["signature"] == "kevlar-local"


class TestThinkingSignature:
    def test_signature_field_exists(self):
        tc = ThinkingContent(thinking="reasoning")
        data = tc.model_dump()
        assert "signature" in data
        assert data["signature"] == ""

    def test_signature_field_set(self):
        tc = ThinkingContent(thinking="reasoning", signature="kevlar-local")
        data = tc.model_dump()
        assert data["signature"] == "kevlar-local"

    def test_signature_in_response(self):
        resp = MessagesResponse(
            model="test",
            content=[
                ThinkingContent(thinking="step by step", signature="kevlar-local"),
                TextContent(text="answer"),
            ],
            stop_reason="end_turn",
        )
        data = resp.model_dump()
        assert data["content"][0]["signature"] == "kevlar-local"
        assert data["content"][0]["type"] == "thinking"


class TestSSEMessageStart:
    def test_output_tokens_initial(self):
        from kevlar.api.sse import message_start_event
        event = message_start_event(model="test", input_tokens=100)
        import json
        data = json.loads(event["data"])
        assert data["message"]["usage"]["output_tokens"] == 1
        assert data["message"]["usage"]["input_tokens"] == 100

    def test_cache_metrics(self):
        from kevlar.api.sse import message_start_event
        event = message_start_event(
            model="test", input_tokens=100,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=20,
        )
        import json
        data = json.loads(event["data"])
        assert data["message"]["usage"]["cache_read_input_tokens"] == 80
        assert data["message"]["usage"]["cache_creation_input_tokens"] == 20


class TestHaikuRouting:
    def test_is_haiku_request(self):
        from kevlar.api.haiku_proxy import is_haiku_request
        assert is_haiku_request("claude-haiku-4-5-20251001") is True
        assert is_haiku_request("claude-haiku-4-5") is True
        assert is_haiku_request("CLAUDE-HAIKU-4-5") is True
        assert is_haiku_request("claude-sonnet-4-6") is False
        assert is_haiku_request("mlx-community/Qwen3-8B") is False
        assert is_haiku_request("") is False
