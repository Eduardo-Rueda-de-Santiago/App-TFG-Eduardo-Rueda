"""
memory_game_agent.py — Factory for the memory-game pydantic-ai Agent.

Usage
-----
Build a fully configured Agent and hand it to PydanticAudioAgentLoop::

    from ai_agent.config.llm_config import LLMConfig
    from ai_agent.agent_loop.memory_game_agent import build_memory_game_agent
    from ai_agent.agent_loop.agent_pydantic import PydanticAudioAgentLoop

    model = LLMConfig().build_model()
    agent = build_memory_game_agent(model)
    loop  = PydanticAudioAgentLoop(agent, prompt_queue, response_queue)

Swapping the tool-set or persona only requires writing a different factory
function and passing its result to ``PydanticAudioAgentLoop``.

Tools
-----
All four tools call the NestJS memory-game backend on ``http://localhost:4000``.
They are intentionally synchronous HTTP calls wrapped with
``asyncio.to_thread`` so they never block the event loop.

System prompt
-------------
The system prompt is tuned for a voice interface:
- Short, natural sentences (easier for TTS).
- The agent reveals card values it has seen and uses memory to reason.
- It never asks the user to flip cards; it does it autonomously.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NestJS backend
# ---------------------------------------------------------------------------

_BACKEND_URL = "http://localhost:4000"

_MEMORY_GAME_SYSTEM_PROMPT = """
You are a helpful voice assistant playing a memory card game.
The game board has 16 cards arranged in a 4×4 grid.
Each card has a hidden value; cards come in matched pairs.
Your goal is to find all pairs by flipping two cards at a time.

Rules you must follow:
- Always fetch the current game state before making a move.
- Only flip two cards per turn.
- Use your memory: if you have seen a card value before, remember its position.
- After flipping two cards, wait for the result and comment briefly on what happened.
- Keep your spoken responses short and natural — one or two sentences at most.
- When the game is won, congratulate the user warmly and ask if they want to play again.

Speak in a friendly, conversational tone suited for a voice interface.
Never output bullet points, markdown, or code blocks.
""".strip()


# ---------------------------------------------------------------------------
# Backend HTTP helper (sync — called from a thread)
# ---------------------------------------------------------------------------


def _http(
    endpoint: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send an HTTP request to the NestJS game backend (blocking)."""
    url = f"{_BACKEND_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            raise RuntimeError(
                f"Cannot reach the memory-game backend at {_BACKEND_URL}. "
                "Make sure the NestJS server is running."
            ) from exc
        raise RuntimeError(f"HTTP request to {url} failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected error calling {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_memory_game_agent(model: Model) -> Agent:
    """Return a pydantic-ai Agent configured for the memory card game.

    The agent is initialised with *model* (any pydantic-ai-compatible model),
    the memory-game system prompt, and four tools that call the NestJS backend.

    Parameters
    ----------
    model:
        A pydantic-ai Model returned by ``LLMConfig.build_model()``.  The
        agent does not care which provider the model uses.

    Returns
    -------
    Agent
        Fully configured agent ready to be passed to ``PydanticAudioAgentLoop``.
    """
    agent: Agent = Agent(
        model,
        system_prompt=_MEMORY_GAME_SYSTEM_PROMPT,
    )

    # ------------------------------------------------------------------
    # Tool: get_game_state
    # ------------------------------------------------------------------

    @agent.tool_plain
    async def get_game_state(mask_unrevealed: bool = True) -> dict[str, Any]:
        """Retrieve the current memory game state from the backend.

        Args:
            mask_unrevealed: When True, cards that are neither flipped nor
                matched have their value replaced with "?" so the model
                must rely on memory rather than peeking at the board.
        """
        state = await asyncio.to_thread(_http, "/game/state", "GET")

        if mask_unrevealed and "cards" in state:
            masked = []
            for card in state["cards"]:
                c = dict(card)
                if not card.get("flipped") and not card.get("matched"):
                    c["value"] = "?"
                masked.append(c)
            state["cards"] = masked

        logger.debug("[memory-game] game state fetched")
        return state

    # ------------------------------------------------------------------
    # Tool: flip_card
    # ------------------------------------------------------------------

    @agent.tool_plain
    async def flip_card(card_id: int) -> dict[str, Any]:
        """Flip a card on the board by its 1-based ID.

        Args:
            card_id: The card to flip (1 – 16).  If two unmatched cards are
                already face-up, the backend auto-unflips them after ~1.2 s.
        """
        result = await asyncio.to_thread(
            _http, "/game/flip", "POST", {"cardId": card_id}
        )
        logger.debug("[memory-game] flipped card %d → %s", card_id, result)
        return result

    # ------------------------------------------------------------------
    # Tool: reset_game
    # ------------------------------------------------------------------

    @agent.tool_plain
    async def reset_game() -> dict[str, Any]:
        """Reset the game to a fresh shuffled board."""
        result = await asyncio.to_thread(_http, "/game/reset", "POST")
        logger.debug("[memory-game] game reset")
        return result

    # ------------------------------------------------------------------
    # Tool: play_again
    # ------------------------------------------------------------------

    @agent.tool_plain
    async def play_again() -> dict[str, Any]:
        """Start a new game after the current one has been won."""
        result = await asyncio.to_thread(_http, "/game/play-again", "POST")
        logger.debug("[memory-game] play again triggered")
        return result

    return agent
