"""
agent_generic.py — Abstract base class for the LLM agent loop.

Sits between the audio transcription pipeline and the TTS speaker:

    prompt_queue  ──►  AudioAgentLoop  ──►  response_queue
         ▲                                        │
   (FasterWhisperTranscriber)              (AudioSpeakerGeneric)

Design contract
---------------
- Reads text prompts from ``input_queue``  (produced by the transcriber).
- Passes each prompt to ``process_prompt()``, which subclasses implement.
- ``process_prompt()`` must yield text chunks as an async generator.
- Chunks are accumulated into sentence-length pieces before being pushed
  to ``output_queue`` so the TTS engine receives natural phrase boundaries.
- A ``None`` sentinel is pushed after each response to signal
  end-of-response to the speaker (see ``AudioSpeakerGeneric`` protocol).

Subclass contract
-----------------
Implement exactly one method::

    async def process_prompt(self, prompt: str) -> AsyncIterator[str]:
        # Call the LLM and yield text chunks.
        ...

The base class handles queue wiring, sentence chunking, and shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# Characters that mark a natural sentence boundary for TTS chunking.
_SENTENCE_ENDINGS = frozenset(".!?\n")


class AudioAgentLoop(ABC):
    """Abstract agent loop connecting the transcription queue to the TTS queue.

    Parameters
    ----------
    input_queue:
        Async queue of ``str`` prompts produced by the audio transcriber.
    output_queue:
        Async queue consumed by the TTS speaker.  Accepts ``str`` chunks
        and ``None`` sentinels (end-of-response markers).
    sentence_chunk:
        When ``True`` (default), text chunks yielded by ``process_prompt``
        are buffered and only forwarded to the TTS queue at sentence
        boundaries (``.``, ``!``, ``?``, ``\\n``).  This prevents the TTS
        engine from synthesising incomplete phrases.
        Set to ``False`` to forward every chunk immediately (useful for
        streaming word-by-word TTS engines).
    """

    def __init__(
        self,
        input_queue: asyncio.Queue[str],
        output_queue: "asyncio.Queue[str | None]",
        *,
        sentence_chunk: bool = True,
    ) -> None:
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._sentence_chunk = sentence_chunk
        self._running = False

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement this one method
    # ------------------------------------------------------------------

    @abstractmethod
    async def process_prompt(self, prompt: str) -> AsyncIterator[str]:
        """Process a single user prompt and yield response text chunks.

        Parameters
        ----------
        prompt:
            The transcribed user utterance.

        Yields
        ------
        str
            Incremental text chunks from the LLM response.
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Drain the input queue and process each prompt in turn.

        Runs until ``stop()`` is called.  Each prompt is processed
        sequentially — a new prompt is not pulled from the queue until the
        previous response has been fully streamed to the TTS queue.
        """
        self._running = True
        logger.info("[agent] loop started")

        while self._running:
            try:
                prompt = await asyncio.wait_for(
                    self._input_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            logger.debug("[agent] received prompt: %r", prompt)
            try:
                await self._handle_prompt(prompt)
            except Exception:
                logger.exception("[agent] error processing prompt %r", prompt)
            finally:
                self._input_queue.task_done()

        logger.info("[agent] loop stopped")

    async def stop(self) -> None:
        """Signal the drain loop to exit after the current prompt finishes."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _handle_prompt(self, prompt: str) -> None:
        """Route a prompt through the LLM and push chunks to the TTS queue."""
        buffer = ""

        async for chunk in self.process_prompt(prompt):
            if not chunk:
                continue

            if self._sentence_chunk:
                buffer += chunk
                # Flush complete sentences as soon as they arrive.
                while True:
                    boundary = self._sentence_boundary(buffer)
                    if boundary == -1:
                        break
                    sentence = buffer[: boundary + 1].strip()
                    buffer = buffer[boundary + 1 :]
                    if sentence:
                        logger.debug("[agent] → TTS: %r", sentence)
                        await self._output_queue.put(sentence)
            else:
                await self._output_queue.put(chunk)

        # Flush any remaining text that did not end with a sentence marker.
        if buffer.strip():
            logger.debug("[agent] → TTS (tail): %r", buffer.strip())
            await self._output_queue.put(buffer.strip())

        # End-of-response sentinel — speaker uses this to fire on_speaking_stopped.
        await self._output_queue.put(None)

    @staticmethod
    def _sentence_boundary(text: str) -> int:
        """Return the index of the first sentence-ending character, or -1."""
        for i, ch in enumerate(text):
            if ch in _SENTENCE_ENDINGS:
                return i
        return -1
