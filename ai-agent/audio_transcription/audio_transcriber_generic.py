"""
audio_transcriber_generic.py — Abstract base class for speech-to-text transcription.

Sits between the VAD detector and the prompt queue.  It receives raw audio
frames from the detector (via callbacks), transcribes them asynchronously,
and pushes the resulting text prompt into an external ``OutputQueue``.

Design contract
---------------
- The transcriber is **event-driven**: it does not run its own audio loop.
  Instead it exposes two callbacks that the detector calls:

      ``on_audio_start()``   — notified that speech has begun.
      ``on_audio_end(frames)`` — receives the complete utterance frames.

- On receiving frames it transcribes them and, if the result is non-empty,
  calls ``on_transcription_ready`` (injected by the orchestrator) so that
  the orchestrator can route the prompt to the AI back-end.

- The transcriber also fires ``on_transcription_started`` the moment it
  begins processing, giving the speaker a chance to register it has work
  to do (useful for UI indicators).

Callback protocol
-----------------
``on_transcription_started() -> None``
    Fired at the start of each transcription job.  Useful for UI spinners,
    logging, and cancellation logic.

``on_transcription_ready(prompt: str) -> None``
    Fired when a non-empty transcription is available.  The orchestrator
    wires this to ``output_queue.put(prompt)`` or equivalent.

Queue contract
--------------
``output_queue`` must expose an async ``put(item: str)`` coroutine.
The ``OutputQueue`` from ``core.queues`` satisfies this contract.

Typical back-ends
-----------------
- OpenAI Whisper (local or API)
- Faster-Whisper / WhisperX
- Google Speech-to-Text
- Azure Cognitive Services Speech SDK
"""

from abc import ABC, abstractmethod
from typing import Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Minimal queue protocol — decouples the transcriber from any specific Queue
# ---------------------------------------------------------------------------

@runtime_checkable
class OutputQueue(Protocol):
    """Structural protocol for any async-capable output queue."""

    async def put(self, item: str) -> None:  # noqa: D102
        ...


# Type aliases
TranscriptionStartedCallback = Callable[[], None]
TranscriptionReadyCallback = Callable[[str], None]


class AudioTranscriberGeneric(ABC):
    """
    Abstract base class for speech-to-text (STT) transcription.

    Parameters
    ----------
    output_queue:
        Any object that exposes an async ``put(str)`` method.  Transcribed
        prompts are pushed here so that an upstream orchestrator or AI
        back-end can consume them independently of this class.
    on_transcription_started:
        Fired at the beginning of each transcription job.  Wire this to a
        UI indicator or a speaker gate if needed.
    on_transcription_ready:
        Fired with the final prompt string once transcription completes and
        the result is non-empty.  Typically implemented as
        ``lambda p: asyncio.create_task(output_queue.put(p))`` by the
        orchestrator, but kept as a plain callback here for maximum
        flexibility.
    sample_rate:
        The sample rate (Hz) of the raw PCM frames the detector will supply.
        Must match the rate the underlying STT engine expects.
    """

    def __init__(
        self,
        output_queue: OutputQueue,
        on_transcription_started: TranscriptionStartedCallback,
        on_transcription_ready: TranscriptionReadyCallback,
        *,
        sample_rate: int = 16_000,
    ) -> None:
        self.output_queue = output_queue
        self._on_transcription_started = on_transcription_started
        self._on_transcription_ready = on_transcription_ready
        self.sample_rate = sample_rate
        self._is_transcribing: bool = False

    # ------------------------------------------------------------------
    # VAD callbacks — wired to AudioDetectorGeneric by the orchestrator
    # ------------------------------------------------------------------

    def on_audio_start(self) -> None:
        """
        Notification that the user has started speaking.

        The base implementation is a no-op hook; subclasses may override
        it to pre-warm the STT engine, reset internal buffers, or update
        UI state.  Always call ``super().on_audio_start()`` if you override.
        """

    def on_audio_end(self, audio_frames: list[bytes]) -> None:
        """
        Receive the completed utterance and schedule transcription.

        Called by the detector (via ``AudioDetectorGeneric._emit_speech_end``)
        immediately after silence has persisted past the hold threshold.

        The base implementation fires ``on_transcription_started`` to signal
        work has begun, then delegates to ``transcribe`` for the actual STT
        work.  Subclasses must **not** override this method; override
        ``transcribe`` instead.

        Parameters
        ----------
        audio_frames:
            Raw PCM frames (bytes) in chronological order covering the full
            utterance, at ``self.sample_rate`` Hz.
        """
        if not audio_frames:
            return

        self._is_transcribing = True
        self._on_transcription_started()

        import asyncio  # local import — keeps the module import-light
        asyncio.create_task(self._run_transcription(audio_frames))

    # ------------------------------------------------------------------
    # Core abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def transcribe(self, audio_frames: list[bytes]) -> str:
        """
        Convert raw PCM frames into a text string.

        This is the only method subclasses **must** implement.  All STT
        engine integration (API calls, local model inference, format
        conversion) belongs here.

        Parameters
        ----------
        audio_frames:
            Raw PCM frames at ``self.sample_rate`` Hz, as supplied by the
            detector.  Implementations are responsible for any encoding or
            format conversion the underlying engine requires (e.g. writing
            to a temporary WAV file, converting to float32 numpy array).

        Returns
        -------
        str
            The transcribed text, or an empty string if recognition failed
            or the audio was too short / noisy.  The caller trims whitespace
            before checking emptiness.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_transcription(self, audio_frames: list[bytes]) -> None:
        """
        Orchestrate a single transcription job end-to-end.

        Called internally by ``on_audio_end``.  Subclasses should not
        override this — customise ``transcribe`` instead.
        """
        try:
            prompt = await self.transcribe(audio_frames)
            prompt = prompt.strip()
            if prompt:
                self._on_transcription_ready(prompt)
                await self.output_queue.put(prompt)
        finally:
            self._is_transcribing = False

    @property
    def is_transcribing(self) -> bool:
        """True while a transcription job is in-flight."""
        return self._is_transcribing
