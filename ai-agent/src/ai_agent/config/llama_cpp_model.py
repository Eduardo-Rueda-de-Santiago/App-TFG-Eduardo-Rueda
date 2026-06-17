"""
llama_cpp_model.py - pydantic-ai Model implementation backed by llama-cpp-python.

Uses Llama.create_chat_completion() directly -- no HTTP server required.

Key insight from benchmarks (ai-models_benchmark_llamacpp.ipynb):
  chat_format MUST be passed to the Llama() constructor at load time.
  Without it, tool_calls are always empty regardless of what tools= you pass.

  Recommended formats:
    "chatml-function-calling"  -- LFM2.5, Qwen2.5, Phi-4-mini, most models
    "llama-3"                  -- Llama 3.x models
    None                       -- let llama-cpp autodetect (ok for text, unreliable for tools)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic_ai.models import (
    Model,
    ModelMessage,
    ModelRequest,
    ModelRequestParameters,
    ModelResponse,
    ModelSettings,
    RequestUsage,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.messages import (
    InstructionPart,
    RetryPromptPart,
    ToolReturnPart,
)

logger = logging.getLogger(__name__)


class LlamaCppDirectModel(Model):
    """pydantic-ai Model that calls llama-cpp-python directly.

    Parameters
    ----------
    llm:
        A pre-loaded llama_cpp.Llama instance. MUST have been constructed
        with the correct chat_format for tool calling to work.
    display_name:
        Identifier returned in ModelResponse.model_name.
    """

    # Declare _provider = None so the base class property does not raise.
    _provider = None  # type: ignore[assignment]

    def __init__(
        self,
        llm: Any,
        display_name: str = "llama-cpp-direct",
    ) -> None:
        super().__init__()
        self._llm = llm
        self._display_name = display_name

    # ------------------------------------------------------------------
    # Abstract properties required by pydantic-ai Model base class
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._display_name

    @property
    def system(self) -> str | None:
        return None

    # ------------------------------------------------------------------
    # pydantic-ai Model interface
    # ------------------------------------------------------------------

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Convert pydantic-ai messages -> llama-cpp call -> pydantic-ai response."""
        chat_messages = self._to_chat_messages(messages)
        tools = self._to_tool_definitions(model_request_parameters)

        kwargs: dict[str, Any] = {"messages": chat_messages, "stream": False}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if model_settings:
            if getattr(model_settings, "temperature", None) is not None:
                kwargs["temperature"] = model_settings.temperature
            if getattr(model_settings, "max_tokens", None) is not None:
                kwargs["max_tokens"] = model_settings.max_tokens

        logger.debug(
            "[llama] create_chat_completion: %d messages, %d tools",
            len(chat_messages),
            len(tools) if tools else 0,
        )
        response = await asyncio.to_thread(
            self._llm.create_chat_completion, **kwargs
        )
        return self._to_model_response(response)

    # ------------------------------------------------------------------
    # Message conversion: pydantic-ai -> OpenAI-compatible chat format
    # ------------------------------------------------------------------

    def _to_chat_messages(self, messages: list[ModelMessage]) -> list[dict]:
        """Convert pydantic-ai message history to OpenAI-compatible dicts."""
        result: list[dict] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        result.append({"role": "system", "content": part.content})

                    elif isinstance(part, InstructionPart):
                        result.append({"role": "system", "content": part.content})

                    elif isinstance(part, UserPromptPart):
                        content = (
                            part.content
                            if isinstance(part.content, str)
                            else str(part.content)
                        )
                        result.append({"role": "user", "content": content})

                    elif isinstance(part, ToolReturnPart):
                        result.append({
                            "role": "tool",
                            "content": str(part.content),
                            "tool_call_id": part.tool_call_id,
                        })

                    elif isinstance(part, RetryPromptPart):
                        result.append({"role": "user", "content": str(part.content)})

            else:
                # ModelResponse -- reconstruct the assistant turn.
                text_parts: list[str] = []
                tool_calls: list[dict] = []

                for part in msg.parts:
                    if isinstance(part, TextPart):
                        text_parts.append(part.content)
                    elif isinstance(part, ToolCallPart):
                        args = part.args
                        if args is None:
                            args_str = "{}"
                        elif isinstance(args, str):
                            args_str = args
                        else:
                            args_str = json.dumps(args)

                        tool_calls.append({
                            "id": part.tool_call_id or f"call_{part.tool_name}",
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": args_str,
                            },
                        })

                assistant: dict[str, Any] = {
                    "role": "assistant",
                    # content MUST always be present. The chatml-function-calling
                    # Jinja template accesses message.content unconditionally; if
                    # the key is missing (tool-call-only turn) it raises
                    # UndefinedError.  Use None when there is no prose.
                    "content": "".join(text_parts) if text_parts else None,
                }
                if tool_calls:
                    assistant["tool_calls"] = tool_calls

                if assistant["content"] is not None or tool_calls:
                    result.append(assistant)

        return result

    # ------------------------------------------------------------------
    # Tool definition conversion: pydantic-ai -> OpenAI format
    # ------------------------------------------------------------------

    def _to_tool_definitions(
        self, params: ModelRequestParameters
    ) -> list[dict] | None:
        """Convert pydantic-ai ToolDefinition list to OpenAI tool dicts."""
        if not params.function_tools:
            return None

        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.parameters_json_schema,
                },
            }
            for t in params.function_tools
        ]

    # ------------------------------------------------------------------
    # Response conversion: llama-cpp dict -> pydantic-ai ModelResponse
    # ------------------------------------------------------------------

    def _to_model_response(self, response: dict) -> ModelResponse:
        """Convert a llama-cpp create_chat_completion response to ModelResponse."""
        choice = response["choices"][0]
        message = choice.get("message", {})
        parts: list = []

        content = message.get("content") or ""
        tool_calls_raw = message.get("tool_calls") or []

        if content.strip():
            parts.append(TextPart(content=content))

        for tc in tool_calls_raw:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args: Any = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = raw_args
            else:
                args = raw_args

            parts.append(ToolCallPart(
                tool_name=fn.get("name", "unknown"),
                args=args,
                tool_call_id=tc.get("id") or f"call_{fn.get('name', 'tool')}",
            ))

        # Guard: pydantic-ai raises UnexpectedModelBehavior if the ModelResponse
        # has no parts at all.  This happens when the model returns empty content
        # AND no tool calls (e.g. after a tool error left it confused).  Fall back
        # to an empty TextPart so pydantic-ai gets a valid (if empty) string reply.
        if not parts:
            parts.append(TextPart(content=""))

        usage = response.get("usage") or {}
        return ModelResponse(
            parts=parts,
            usage=RequestUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            model_name=self._display_name,
            timestamp=datetime.now(timezone.utc),
        )
