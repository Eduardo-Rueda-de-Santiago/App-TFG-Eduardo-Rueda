"""
input_generic.py — Abstract base class for user-prompt input sources.

Defines the interface that any input source must satisfy so that the main
agent loop can be written once and remain agnostic about whether prompts
come from a terminal, a microphone, a network socket, etc.

Design contract
---------------
- ``start()`` initialises any background resources (mic stream, thread pool).
  It is *not* a long-running coroutine; it returns as soon as the source
  is ready to accept ``get_next_prompt`` calls.
- ``get_next_prompt()`` blocks (awaits) until the next prompt is available,
  then returns it as a plain string.  Returns ``None`` when the input source
  signals end-of-session (user typed "quit", microphone stopped, etc.).
- ``stop()`` tears down background resources cleanly.

Typical implementations
-----------------------
- ``TerminalInput``  — reads from stdin via ``asyncio.get_event_loop().run_in_executor``
- ``VoiceInput``     — VAD + Whisper transcription pipeline producing strings
                       via an asyncio.Queue
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class InputGeneric(ABC):
    """
    Abstract base class for user-prompt input sources.

    All concrete input classes must implement ``get_next_prompt``.
    ``start`` and ``stop`` have no-op default implementations; override
    them only when background resources need to be managed.
    """

    async def start(self) -> None:
        """
        Initialise and start any background resources required by this source.

        Called once by the agent loop before the first ``get_next_prompt``
        call.  The default implementation is a no-op.
        """

    @abstractmethod
    async def get_next_prompt(self) -> str | None:
        """
        Await and return the next user prompt.

        Blocks until a prompt is available.

        Returns
        -------
        str
            The user's input as a plain string, ready to pass to the pipeline.
        None
            Signals that the input source has finished (e.g. the user typed
            "quit" or the microphone session ended).  The agent loop should
            exit cleanly on receiving ``None``.
        """

    async def stop(self) -> None:
        """
        Signal background resources to shut down and release them.

        Called once by the agent loop after the main loop exits or on an
        unhandled exception.  The default implementation is a no-op.
        """
