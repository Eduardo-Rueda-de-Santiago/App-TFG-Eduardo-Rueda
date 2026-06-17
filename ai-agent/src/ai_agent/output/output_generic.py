"""
output_generic.py — Abstract base class for agent response output channels.

Defines the interface that any output sink must satisfy so that the main
agent loop can emit responses without knowing whether they go to a terminal,
a TTS speaker, a network socket, etc.

Design contract
---------------
- ``start()`` initialises any background resources (audio stream, thread pool).
  Returns as soon as the channel is ready to accept ``emit`` calls.
- ``emit(message)`` delivers one complete response string.  For TTS this
  involves queuing audio; for terminal this involves printing.  The coroutine
  should await until the delivery is fully in-flight (not necessarily until
  the audio finishes playing).
- ``stop()`` flushes any in-flight output and releases resources cleanly.

Typical implementations
-----------------------
- ``TerminalOutput``  — prints the formatted response to stdout
- ``TTSOutput``       — pushes text into a PiperSpeaker queue for audio playback
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class OutputGeneric(ABC):
    """
    Abstract base class for response output channels.

    ``start`` and ``stop`` have no-op default implementations; override them
    only when background resources need to be managed.
    """

    async def start(self) -> None:
        """
        Initialise and start any background resources required by this channel.

        Called once by the agent loop before the first ``emit`` call.
        The default implementation is a no-op.
        """

    @abstractmethod
    async def emit(self, message: str, *, task_achieved: bool = True) -> None:
        """
        Deliver one complete response to the user.

        Parameters
        ----------
        message:
            The agent's natural-language response text.
        task_achieved:
            Whether the pipeline considers the user's original intent
            fulfilled.  Implementations may use this for formatting or
            audio cues but are not required to.
        """

    async def stop(self) -> None:
        """
        Flush in-flight output and release background resources.

        Called once by the agent loop after the main loop exits or on an
        unhandled exception.  The default implementation is a no-op.
        """
