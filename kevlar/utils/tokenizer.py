from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import mlx.core as mx

from kevlar.api.models import (
    ContentBlock,
    Message,
    MessagesRequest,
    TextContent,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)

logger = logging.getLogger(__name__)


def _format_tools_for_template(tools: list[ToolDefinition]) -> list[dict]:
    """Convert Anthropic tool definitions to the format expected by chat templates."""
    formatted = []
    for tool in tools:
        formatted.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema.model_dump(),
            },
        })
    return formatted


def _message_to_template_format(msg: Message) -> dict:
    """Convert an Anthropic message to the dict format expected by apply_chat_template."""
    if isinstance(msg.content, str):
        return {"role": msg.role, "content": msg.content}

    parts = []
    tool_calls = []

    for block in msg.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolUseContent):
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": block.input,
                },
            })
        elif isinstance(block, ToolResultContent):
            # tool results go as separate messages with role=tool
            content_text = block.content if isinstance(block.content, str) else " ".join(
                b.text for b in block.content if hasattr(b, "text")
            )
            return {
                "role": "tool",
                "tool_call_id": block.tool_use_id,
                "content": content_text,
            }

    result = {"role": msg.role, "content": "\n".join(parts) if parts else ""}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def request_to_token_ids(
    request: MessagesRequest,
    tokenizer,
    system_text: str,
    enable_thinking: bool = False,
) -> tuple[mx.array, bool]:
    """Convert an Anthropic Messages request to token IDs using the tokenizer's chat template.

    Returns (token_ids, thinking_actually_enabled).
    """
    template_messages = []

    for msg in request.messages:
        if isinstance(msg.content, list):
            non_tool_blocks = []
            for block in msg.content:
                if isinstance(block, ToolResultContent):
                    if non_tool_blocks:
                        synthetic = Message(role=msg.role, content=non_tool_blocks)
                        template_messages.append(_message_to_template_format(synthetic))
                        non_tool_blocks = []
                    template_messages.append(_message_to_template_format(
                        Message(role=msg.role, content=[block])
                    ))
                else:
                    non_tool_blocks.append(block)
            if non_tool_blocks:
                synthetic = Message(role=msg.role, content=non_tool_blocks)
                template_messages.append(_message_to_template_format(synthetic))
        else:
            template_messages.append(_message_to_template_format(msg))

    tools = _format_tools_for_template(request.tools) if request.tools else None
    thinking_active = False

    # try with enable_thinking if requested, fall back gracefully
    template_kwargs = {}
    if enable_thinking:
        template_kwargs["enable_thinking"] = True

    try:
        token_ids = tokenizer.apply_chat_template(
            template_messages,
            tools=tools,
            add_generation_prompt=True,
            **template_kwargs,
        )
        # verify the template actually added thinking markers
        if enable_thinking:
            tail = tokenizer.decode(token_ids[-10:])
            thinking_active = "<think>" in tail
    except TypeError:
        # model template doesn't support enable_thinking -- retry without it
        template_kwargs.pop("enable_thinking", None)
        try:
            token_ids = tokenizer.apply_chat_template(
                template_messages,
                tools=tools,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except Exception:
            logger.warning("apply_chat_template with tools failed, falling back without tools")
            token_ids = None
    except Exception:
        logger.warning("apply_chat_template with tools failed, falling back without tools")
        token_ids = None

    if token_ids is None:
        if tools and system_text:
            tools_text = "\n\nAvailable tools:\n" + json.dumps(tools, indent=2)
            system_text = system_text + tools_text

        if system_text and template_messages:
            first = template_messages[0]
            if first["role"] == "user":
                first["content"] = system_text + "\n\n" + first.get("content", "")

        token_ids = tokenizer.apply_chat_template(
            template_messages,
            add_generation_prompt=True,
        )

    return mx.array(token_ids), thinking_active


_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TAIL_PATTERN = re.compile(r"^.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove thinking content from model output.

    Handles two cases:
    - <think>...</think> fully present in output
    - Output starts inside a think block (prompt ended with <think>)
      so text is: [thinking content]</think>[actual answer]
    """
    result = _THINK_BLOCK_PATTERN.sub("", text)
    if "</think>" in result:
        result = _THINK_TAIL_PATTERN.sub("", result)
    return result.lstrip()


def parse_tool_calls(text: str) -> list[ToolUseContent]:
    """Parse tool call syntax from model output.

    Handles formats:
    - <tool_call>{"name": ..., "arguments": ...}</tool_call>  (Qwen JSON style)
    - <function=Name><parameter=key>value</parameter></function>  (Qwen XML style)
    - {"name": ..., "arguments": ...}  (bare JSON)
    """
    import uuid
    tool_calls = []

    # try <function=Name> XML style first (Qwen with tools via chat template)
    # check both bare text and inside <tool_call> wrappers
    func_pattern = re.compile(
        r"<function=(\w+)>\s*(.*?)\s*</function>",
        re.DOTALL,
    )
    func_matches = func_pattern.findall(text)
    for name, params_block in func_matches:
        args = {}
        param_pattern = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)
        for param_name, param_value in param_pattern.findall(params_block):
            try:
                args[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                args[param_name] = param_value
        tool_calls.append(ToolUseContent(
            id=f"toolu_{uuid.uuid4().hex[:24]}",
            name=name,
            input=args,
        ))
    if tool_calls:
        return tool_calls

    # try <tool_call> JSON tags
    tag_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    matches = tag_pattern.findall(text)

    if not matches:
        json_pattern = re.compile(
            r'\{"(?:name|function)":\s*"[^"]+"\s*,\s*"(?:arguments|parameters)":\s*\{.*?\}\s*\}',
            re.DOTALL,
        )
        matches = json_pattern.findall(text)

    for match in matches:
        try:
            data = json.loads(match)
            name = data.get("name") or data.get("function", "")
            args = data.get("arguments") or data.get("parameters", {})
            if isinstance(args, str):
                args = json.loads(args)

            tool_calls.append(ToolUseContent(
                id=f"toolu_{uuid.uuid4().hex[:24]}",
                name=name,
                input=args,
            ))
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to parse tool call: %s", match[:100])
            continue

    return tool_calls
