"""
audio_speaker_generic.py — Abstract base class for text-to-speech (TTS) playback.

Consumes text chunks from an external async queue and converts them into
audible speech.  Chunks are played in FIFO order; if a new chunk arrives
while one is still playing, it is queued and played immediately after.

Design contract
---------------
- Audio is **never** persisted to disk; every chunk is played and discarded.
- The speaker runs a continuous drain loop (``start`` / ``stop`` lifecycle).
- While the speaker is playing it notifies the VAD detector to mute itself
  so that TTS output cannot be mis-transcribed as a new user prompt.
- The mute/unmute contract is symmetric:
      ``on_speaking_started`` → fires once before the *first* chunk plays.
      ``on_speaking_stopped``  → fires once after the *last* chunk in a
                                  contiguous burst has finished.

Callback protocol
-----------------
``on_speaking_started() -> None``
    Wired to ``AudioDetectorGeneric.on_speaking_started`` by the orchestrator.
    Fired once each time the speaker transitions from idle → active.

``on_speaking_stopped() -> None``
    Wired to ``AudioDetectorGeneric.on_speaking_stopped`` by the orchestrator.
    Fired once each time the speaker transitions from active → idle (i.e.
    the input queue is empty and the last chunk has finished playing).

Queue contract
--------------
``input_queue`` must expose an async ``get() -> str | None`` coroutine and
an async ``task_done()`` coroutine (compatible with ``asyncio.Queue``).
A ``None`` sentinel signals end-of-response from the AI back-end; the
speaker treats it as a flush/boundary marker, not as a stop signal.

Typical TTS back-ends
---------------------
- ElevenLabs streaming API
- OpenAI TTS (tts-1 / tts-1-hd)
- Coqui TTS (local)
- pyttsx3 (offline, cross-platform)
- Amazon Polly / Google Cloud TTS
"""

from abc import ABC, abstractmethod
from typing import Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Minimal queue protocol — keeps the speaker decoupled from asyncio.Queue
# ---------------------------------------------------------------------------


@runtime_checkable
class InputQueue(Protocol):
    """Structural protocol for any async-capable text input queue."""

    async def get(self) -> "str | None":  # noqa: D102
        ...

    def task_done(self) -> None:  # noqa: D102
        ...


# Type aliases
SpeakingStartedCallback = Callable[[], None]
SpeakingStoppedCallback = Callable[[], None]


class AudioSpeakerGeneric(ABC):
    """
    Abstract base class for TTS audio playback with queue-drain semantics.

    Parameters
    ----------
    input_queue:
        Async queue that yields text chunks (``str``) or ``None`` sentinels.
        ``None`` marks the boundary between successive AI responses; the
        speaker uses it to decide when to fire ``on_speaking_stopped`` if
        no more chunks follow immediately.
    on_speaking_started:
        Callback fired once when the speaker transitions from idle to active.
        Wire this to ``AudioDetectorGeneric.on_speaking_started`` so the
        microphone is muted during playback.
    on_speaking_stopped:
        Callback fired once when the queue is drained and the last audio
        chunk has finished playing.  Wire this to
        ``AudioDetectorGeneric.on_speaking_stopped`` to re-enable the mic.
    """

    def __init__(
        self,
        input_queue: InputQueue,
        on_speaking_started: SpeakingStartedCallback,
        on_speaking_stopped: SpeakingStoppedCallback,
    ) -> None:
        self.input_queue = input_queue
        self._on_speaking_started = on_speaking_started
        self._on_speaking_stopped = on_speaking_stopped

        self._running: bool = False
        self._is_speaking: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the queue-drain loop.

        Continuously reads text chunks from ``input_queue`` and speaks them
        one at a time in FIFO order.  Manages the speaking state and fires
        ``on_speaking_started`` / ``on_speaking_stopped`` around each
        contiguous burst of speech.

        This coroutine runs until ``stop()`` is called.
        """
        self._running = True
        while self._running:
            chunk = await self.input_queue.get()

            if chunk is None:
                # Sentinel — end of one AI response.
                # If nothing is queued immediately after, transition to idle.
                if self._is_speaking:
                    self._is_speaking = False
                    self._on_speaking_stopped()
                self.input_queue.task_done()
                continue

            # First chunk after being idle → signal mute to the detector.
            if not self._is_speaking:
                self._is_speaking = True
                self._on_speaking_started()

            try:
                await self._speak_chunk(chunk)
            finally:
                self.input_queue.task_done()

    async def stop(self) -> None:
        """
        Signal the drain loop to exit after the current chunk finishes.

        Does not interrupt audio that is already playing — the speaker
        finishes the current chunk cleanly before shutting down.
        If the speaker is active when ``stop()`` is called, ``on_speaking_stopped``
        is fired so the detector is left in a consistent (unmuted) state.
        """
        self._running = False
        if self._is_speaking:
            self._is_speaking = False
            self._on_speaking_stopped()
        await self._teardown()

    # ------------------------------------------------------------------
    # Core abstract interface — subclasses implement exactly these two
    # ------------------------------------------------------------------

    @abstractmethod
    async def generate_audio(self, input_text: str) -> None:
        """
        Core TTS conversion and playback method.

        Convert ``input_text`` to audio and play it to completion before
        returning.  The caller (``_speak_chunk``) awaits this coroutine, so
        the next chunk will not begin until this one finishes, preserving
        FIFO order without extra locking.

        Audio must **not** be written to disk.

        Parameters
        ----------
        input_text:
            A single text chunk to synthesise and play.
        """

    @abstractmethod
    async def _teardown(self) -> None:
        """
        Release any resources held by the TTS engine.

        Called once by ``stop()`` after the drain loop exits.  Use this to
        close audio device handles, cancel pending API requests, or flush
        internal buffers.  Must be idempotent.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _speak_chunk(self, text: str) -> None:
        """
        Sanitise and play one text chunk.

        Strips whitespace before dispatching to ``generate_audio`` so that
        stray newlines or spaces from streaming LLM output do not produce
        audible artefacts (e.g. a TTS engine reading "blank line").

        Subclasses should **not** override this — customise ``generate_audio``
        instead.
        """
        clean = text.strip()
        if clean:
            await self.generate_audio(clean)

    @property
    def is_speaking(self) -> bool:
        """True while audio is being played."""
        return self._is_speaking
