"""
agent_pydantic.py — Pydantic-AI backed implementation of AudioAgentLoop.

The pydantic-ai Agent encapsulates the full agentic loop described in the
architecture diagram:

    User Prompt
        │
        ▼
    ┌─────────────────────────────────────────┐
    │              pydantic-ai Agent          │
    │                                         │
    │  1. Planner  — model decides whether    │
    │               tools are needed          │
    │                                         │
    │  2. Tool Caller — executes registered   │
    │                   tools if requested    │
    │                                         │
    │  3. Conversational — model produces     │
    │                      the final natural- │
    │                      language answer    │
    └─────────────────────────────────────────┘
        │
        ▼
    Natural Language Answer  ──►  TTS queue

The Agent is injected at construction time, already configured with:
  - The pydantic-ai Model (any supported provider)
  - All tool definitions and their system prompt
  - Any provider-specific settings

Swapping behaviour (different model, different tools, different persona)
only requires passing a different Agent instance — the loop itself is
provider-agnostic.

Conversation history
--------------------
Each ``PydanticAudioAgentLoop`` instance maintains a rolling message history
so the agent has multi-turn memory across prompts.  The history is capped at
``max_history_turns`` full turns to keep context within token limits.
Call ``clear_history()`` to start a new session.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from typing import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import (
    ModelRequest,
    ModelRequestParameters,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from ai_agent.agent_loop.agent_generic import AudioAgentLoop

logger = logging.getLogger(__name__)

# Matches a bare function-call echo like "functions.flip_card:" — the small
# model sometimes outputs this as its prose response instead of a real sentence.
_FUNC_ECHO_RE = re.compile(r"^\s*functions\.\w+:?\s*$")


def _is_function_echo(text: str) -> bool:
    """Return True when *text* is just a function-name echo, not real prose."""
    stripped = text.strip()
    return not stripped or bool(_FUNC_ECHO_RE.match(stripped))

# ANSI colour codes for terminal output (disabled on non-TTY streams)
_TTY = sys.stdout.isatty()
_C_USER  = "\033[96m"   # cyan
_C_AGENT = "\033[92m"   # green
_C_TOOL  = "\033[93m"   # yellow
_C_RESET = "\033[0m"


def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{_C_RESET}" if _TTY else text


class PydanticAudioAgentLoop(AudioAgentLoop):
    """Audio agent loop driven by a pydantic-ai ``Agent``.

    Parameters
    ----------
    agent:
        A fully configured pydantic-ai ``Agent`` instance — model, tools,
        and system prompt are all set before this loop is constructed.
        This is the only coupling point between the loop and the LLM
        backend; replacing the agent swaps the entire AI stack.
    input_queue:
        Async queue of transcribed user prompts (``str``).
    output_queue:
        Async queue consumed by the TTS speaker (``str | None``).
    sentence_chunk:
        Buffer streamed text into sentence-length pieces before forwarding
        to the TTS queue.  Defaults to ``True``.
    max_history_turns:
        Maximum number of complete user-agent turns to retain in the
        conversation history.  Older turns are discarded FIFO.
        Defaults to 20 (≈ 40 messages in the pydantic-ai history list).
    """

    def __init__(
        self,
        agent: Agent,
        input_queue: asyncio.Queue[str],
        output_queue: "asyncio.Queue[str | None]",
        *,
        sentence_chunk: bool = True,
        max_history_turns: int = 20,
    ) -> None:
        super().__init__(input_queue, output_queue, sentence_chunk=sentence_chunk)
        self._agent = agent
        self._history: list = []          # pydantic-ai ModelMessage list
        self._max_history_msgs = max_history_turns * 2   # each turn = 2 msgs

    # ------------------------------------------------------------------
    # AudioAgentLoop interface
    # ------------------------------------------------------------------

    async def process_prompt(self, prompt: str) -> AsyncIterator[str]:
        """Stream the agent response for *prompt*, yielding text chunks.

        The pydantic-ai ``Agent.run_stream`` context manager handles the
        full planner → tool-call → conversational response cycle internally.
        We yield only the final text chunks so the base-class sentence
        chunker can forward them to the TTS queue at natural phrase
        boundaries.

        All intermediate steps (tool calls, tool results) and the assembled
        response are also printed to the terminal for testing/debugging.

        Fallback chain
        --------------
        1. ``Agent.run_stream()``   — preferred; supports streaming deltas.
        2. ``Agent.run()``          — for models that don't implement streaming
                                      (e.g. ``LlamaCppDirectModel``).
        3. ``_call_without_tools()`` — direct model call with no tools, used
                                       when the agent loop fails with
                                       ``UnexpectedModelBehavior``.  This covers
                                       purely conversational turns where the
                                       tool-calling overhead confuses the model
                                       (e.g. it tries a tool, the backend is
                                       unreachable, and it returns an empty reply).

        Parameters
        ----------
        prompt:
            The transcribed user utterance.

        Yields
        ------
        str
            Incremental text deltas from the model's streamed response.
        """
        logger.info("[agent] processing: %r", prompt)

        # Print the user prompt to the terminal
        print(f"\n{_c(_C_USER, 'USER:')} {prompt}", flush=True)
        print(_c(_C_AGENT, "AGENT: "), end="", flush=True)

        assembled = ""

        try:
            async with self._agent.run_stream(
                prompt,
                message_history=self._history or None,
            ) as stream:
                async for delta in stream.stream_text(delta=True):
                    print(delta, end="", flush=True)
                    assembled += delta
                    yield delta

                # Print any tool calls that happened during the run
                from pydantic_ai.models import ToolCallPart as TCP
                for msg in stream.all_messages():
                    if isinstance(msg, ModelResponse):
                        for part in msg.parts:
                            if isinstance(part, TCP):
                                print(
                                    f"\n  {_c(_C_TOOL, '[tool call]')} "
                                    f"{part.tool_name}({part.args})",
                                    flush=True,
                                )

                new_messages = list(stream.all_messages())
                self._history = (self._history + new_messages)[
                    -self._max_history_msgs :
                ]
            # Streaming succeeded — skip the fallback paths below.
            print()
            logger.debug("[agent] response: %r", assembled)
            logger.debug("[agent] history length: %d messages", len(self._history))
            return

        except UnexpectedModelBehavior as exc:
            logger.warning(
                "[agent] streaming run failed (%s), skipping to direct call", exc
            )
        except NotImplementedError:
            pass  # Model doesn't support streaming — try Agent.run() next.

        # ------------------------------------------------------------------ #
        # Fallback 2: blocking Agent.run()                                   #
        # (used when the model doesn't implement streaming)                  #
        # ------------------------------------------------------------------ #
        try:
            result = await self._agent.run(
                prompt,
                message_history=self._history or None,
            )
            assembled = result.output

            # Guard: the small model sometimes echoes the function name as its
            # final prose (e.g. "functions.flip_card:") instead of a real
            # sentence.  Treat this exactly like a hard failure and fall through
            # to the no-tools path so the user gets an actual response.
            if _is_function_echo(assembled):
                logger.warning(
                    "[agent] response %r looks like a function echo; "
                    "falling back to direct no-tools call",
                    assembled,
                )
            else:
                print(assembled, end="", flush=True)
                yield assembled

                new_messages = list(result.all_messages())
                self._history = (self._history + new_messages)[
                    -self._max_history_msgs :
                ]
                print()
                logger.debug("[agent] response: %r", assembled)
                logger.debug("[agent] history length: %d messages", len(self._history))
                return

        except Exception as exc:
            logger.warning(
                "[agent] agent.run() failed (%s: %s), falling back to direct model call",
                type(exc).__name__, exc,
            )

        # ------------------------------------------------------------------ #
        # Fallback 3: direct model call — no tool-calling overhead           #
        # Used for purely conversational prompts where the agent loop failed. #
        # ------------------------------------------------------------------ #
        logger.info("[agent] using direct model call (no tools)")
        assembled, new_messages = await self._call_without_tools(prompt)
        print(assembled, end="", flush=True)
        yield assembled

        self._history = (self._history + new_messages)[-self._max_history_msgs :]
        print()
        logger.debug("[agent] response: %r", assembled)
        logger.debug("[agent] history length: %d messages", len(self._history))

    async def _call_without_tools(self, prompt: str) -> tuple[str, list]:
        """Call the underlying model directly, bypassing the tool-calling loop.

        Injects the system prompt (retrieved from the agent) and conversation
        history, then calls ``model.request()`` with no tools registered.
        This is the "conversational bypass" path — it handles greetings, small
        talk, or any turn where the full agentic loop would otherwise time-out
        or return an empty reply.

        Returns
        -------
        tuple[str, list]
            ``(response_text, new_messages)`` where *new_messages* should be
            appended to ``self._history``.
        """
        # Retrieve static system prompts from the agent (private but stable).
        sys_prompts: tuple[str, ...] = getattr(self._agent, "_system_prompts", ())

        # Build the current request.  On the very first turn the system prompt
        # is not yet in self._history, so we prepend it here.
        request_parts: list = []
        if sys_prompts and not self._history:
            request_parts.extend(SystemPromptPart(content=sp) for sp in sys_prompts)
        request_parts.append(UserPromptPart(content=prompt))

        current_request = ModelRequest(parts=request_parts)
        messages = [*self._history, current_request]

        # No tools — pure text completion.
        params = ModelRequestParameters(
            function_tools=[],
            output_tools=[],
            allow_text_output=True,
        )

        response: ModelResponse = await self._agent.model.request(
            messages, None, params
        )

        text = "".join(
            p.content for p in response.parts if isinstance(p, TextPart)
        )

        return text, [current_request, response]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """Discard conversation history and start a fresh session."""
        self._history = []
        logger.info("[agent] conversation history cleared")

    @property
    def history_length(self) -> int:
        """Number of messages currently in the conversation history."""
        return len(self._history)
