"""
mcp_tools.py — MCP tool schemas and execution layer.

This module does two things:
1. Defines the tool schemas in OpenAI function-calling format so they can be
   passed to llama-cpp-python's `create_chat_completion(tools=...)`.
2. Provides an `execute_tool()` function that dispatches actual HTTP calls to
   the Memory Game NestJS backend (the same calls the FastMCP server makes).

Why direct HTTP instead of the MCP stdio transport?
- The MCP server you provided is a thin wrapper around HTTP endpoints on :4000.
- Calling the backend directly avoids spawning an MCP subprocess, managing stdio
  pipes, and pulling in the `mcp` SDK — all for the same result.
- If you later want to swap in real MCP transport, only this file changes.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ai_agent.config import BACKEND_URL
from ai_agent.schemas import ToolCallRecord

# =============================================================================
# TOOL SCHEMAS  (OpenAI function-calling format for llama-cpp-python)
# =============================================================================

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_game_state",
            "description": (
                "Retrieve the current memory game state including all cards, "
                "move count, match count, and win status.  Unrevealed cards "
                "have their value masked as '?' to prevent cheating."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flip_card",
            "description": (
                "Flip a card on the memory game board by its 1-based ID "
                "(typically 1 to 16).  If two mismatched cards are flipped, "
                "the backend auto-unflips them after 1.2 seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "integer",
                        "description": "The 1-based ID of the card to flip.",
                    }
                },
                "required": ["card_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_game",
            "description": (
                "Reset the memory game to a brand new shuffled state.  "
                "All progress is lost."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_again",
            "description": (
                "Start a new game session.  Useful after winning.  "
                "Returns the fresh initial game state."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Lookup: tool_name → single tool definition dict.
# Used by the orchestrator to send only ONE tool per request (FIX 1 from
# the benchmark: small models choke when given all tools at once).
TOOL_SCHEMA_MAP: dict[str, dict[str, Any]] = {
    t["function"]["name"]: t for t in TOOL_SCHEMAS
}


# =============================================================================
# HTTP DISPATCH  (mirrors what the FastMCP server does internally)
# =============================================================================

# Map tool names to (endpoint, HTTP method)
_TOOL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "get_game_state": ("/game/state", "GET"),
    "flip_card": ("/game/flip", "POST"),
    "reset_game": ("/game/reset", "POST"),
    "play_again": ("/game/play-again", "POST"),
}

# Map tool argument names to the backend's expected JSON keys
_ARG_TRANSFORMS: dict[str, dict[str, str]] = {
    "flip_card": {"card_id": "cardId"},
}


def _http_request(
    endpoint: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send an HTTP request to the NestJS backend and return parsed JSON."""
    url = f"{BACKEND_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Content-Type": "application/json"}

    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        raise RuntimeError(
            f"Backend request to {url} failed: {reason}.  "
            f"Is the NestJS server running on {BACKEND_URL}?"
        ) from exc


def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> ToolCallRecord:
    """
    Execute a single tool call against the Memory Game backend.

    Args:
        tool_name:  One of the tool names from TOOL_SCHEMAS.
        tool_args:  Arguments dict as produced by the LLM's tool_call.

    Returns:
        A validated ToolCallRecord with the result or error.
    """
    if tool_name not in _TOOL_ENDPOINTS:
        return ToolCallRecord(
            tool_name=tool_name,
            tool_args=tool_args,
            success=False,
            error=f"Unknown tool: {tool_name}",
        )

    endpoint, method = _TOOL_ENDPOINTS[tool_name]

    # Transform argument names if needed (e.g. card_id → cardId)
    payload: dict[str, Any] | None = None
    if tool_args:
        transforms = _ARG_TRANSFORMS.get(tool_name, {})
        payload = {transforms.get(k, k): v for k, v in tool_args.items()}

    try:
        result = _http_request(endpoint, method, payload)
        return ToolCallRecord(
            tool_name=tool_name,
            tool_args=tool_args,
            result=result,
            success=True,
        )
    except Exception as exc:
        return ToolCallRecord(
            tool_name=tool_name,
            tool_args=tool_args,
            success=False,
            error=str(exc),
        )
