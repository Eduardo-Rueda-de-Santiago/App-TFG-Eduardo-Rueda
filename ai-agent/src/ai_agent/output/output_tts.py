"""
output_tts.py — Text-to-speech output implementation via PiperTTS.

Wraps a ``PiperSpeaker`` and an internal ``asyncio.Queue`` so that the
agent loop can push response strings and have them spoken aloud without
blocking the main coroutine.

Usage
-----
::

    voice_input = VoiceInput(...)
    tts_output  = TTSOutput(
        on_speaking_started=voice_input.on_speaking_started,
        on_speaking_stopped=voice_input.on_speaking_stopped,
    )
    await tts_output.start()

    await tts_output.emit("Hello, how can I help?")

Mute gating
-----------
When ``on_speaking_started`` and ``on_speaking_stopped`` are wired to the
``VoiceInput``'s mute callbacks, the microphone is automatically silenced
while TTS audio is playing, preventing the agent's own voice from being
re-transcribed as a new user prompt.

If no callbacks are provided (e.g. when the input mode is ``terminal``),
the defaults are no-ops and no muting occurs.

Dependencies
------------
    pip install piper-tts sounddevice numpy
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from ai_agent.config import PIPER_CONFIG_PATH, PIPER_MODEL_PATH, PIPER_USE_CUDA
from ai_agent.output.output_generic import OutputGeneric
from ai_agent.text_to_speech.audio_speaker_piper import PiperSpeaker

logger = logging.getLogger(__name__)


class TTSOutput(OutputGeneric):
    """
    Speaks agent responses using a locally-loaded PiperTTS voice model.

    The ``PiperSpeaker`` drain loop runs as a background asyncio Task.
    ``emit`` puts the message onto the speaker's queue and returns
    immediately (it does *not* wait for playback to finish), keeping the
    agent loop responsive.

    Parameters
    ----------
    model_path:
        Path to the local Piper ``.onnx`` voice model file.
    config_path:
        Path to the companion ``.onnx.json`` config file.
        Defaults to ``<model_path>.json``.
    use_cuda:
        Use CUDA for Piper inference (requires ``onnxruntime-gpu``).
    on_speaking_started:
        Called once when TTS playback begins.  Wire to
        ``VoiceInput.on_speaking_started`` to mute the microphone.
    on_speaking_stopped:
        Called once when TTS playback ends.  Wire to
        ``VoiceInput.on_speaking_stopped`` to unmute the microphone.
    """

    def __init__(
        self,
        *,
        model_path: str = PIPER_MODEL_PATH,
        config_path: str = PIPER_CONFIG_PATH,
        use_cuda: bool = PIPER_USE_CUDA,
        on_speaking_started: Callable[[], None] = lambda: None,
        on_speaking_stopped: Callable[[], None] = lambda: None,
    ) -> None:
        # Internal queue: str chunks or None sentinels for the speaker drain loop
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()

        self._speaker = PiperSpeaker(
            input_queue=self._queue,
            on_speaking_started=on_speaking_started,
            on_speaking_stopped=on_speaking_stopped,
            model_path=model_path,
            config_path=config_path,
            use_cuda=use_cuda,
        )

        # Background task handle
        self._speaker_task: asyncio.Task | None = None

        # Timing / synchronisation
        self._last_synth_duration: float = 0.0
        self._last_play_duration: float = 0.0
        self._speech_done_event: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------
    # OutputGeneric interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Pre-warm the Piper model and launch the speaker drain loop as a
        background asyncio Task.

        Also wraps the speaker's speaking callbacks to maintain a
        ``_speech_done_event`` for timing synchronisation.
        """
        # Create the event inside the running event loop.
        self._speech_done_event = asyncio.Event()
        self._speech_done_event.set()  # idle → "done" by default

        # Capture whatever callbacks are currently set on the speaker
        # (they may already have been replaced by VoiceInput's mute hooks
        # before this coroutine runs).
        _orig_started = self._speaker._on_speaking_started
        _orig_stopped = self._speaker._on_speaking_stopped

        def _wrapped_started() -> None:
            # Speech is starting: mark the event as "not done".
            if self._speech_done_event is not None:
                self._speech_done_event.clear()
            _orig_started()

        def _wrapped_stopped() -> None:
            # Speech finished: capture per-stage durations, then signal "done".
            self._last_synth_duration = getattr(self._speaker, "_last_synth_duration", 0.0)
            self._last_play_duration = getattr(self._speaker, "_last_play_duration", 0.0)
            if self._speech_done_event is not None:
                self._speech_done_event.set()
            _orig_stopped()

        self._speaker._on_speaking_started = _wrapped_started
        self._speaker._on_speaking_stopped = _wrapped_stopped

        logger.info("TTSOutput: pre-loading Piper model …")
        self._speaker.preload()
        logger.info("TTSOutput: Piper model ready.")

        self._speaker_task = asyncio.create_task(
            self._speaker.start(), name="piper-speaker"
        )

    async def emit(self, message: str, *, task_achieved: bool = True) -> None:
        """
        Enqueue *message* for TTS synthesis and playback.

        Returns immediately after queuing; playback happens concurrently
        in the background drain task.

        Parameters
        ----------
        message:
            The agent's natural-language response text.
        task_achieved:
            Not used by TTS output (no audio cue for task status), but
            accepted to satisfy the ``OutputGeneric`` interface.
        """
        if not message.strip():
            return

        # Pre-clear the done event so that wait_for_speech_done() called
        # immediately after emit() blocks until playback actually finishes.
        if self._speech_done_event is not None:
            self._speech_done_event.clear()

        logger.debug("TTSOutput: queuing %d chars for synthesis.", len(message))
        await self._queue.put(message)
        # None sentinel tells the speaker that this response is complete;
        # it will fire on_speaking_stopped once the queue drains.
        await self._queue.put(None)

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    @property
    def last_synth_duration(self) -> float:
        """ONNX inference time for the most recent TTS burst (seconds)."""
        return self._last_synth_duration

    @property
    def last_play_duration(self) -> float:
        """sounddevice write time for the most recent TTS burst (seconds)."""
        return self._last_play_duration

    async def wait_for_speech_done(self) -> None:
        """
        Await until the current TTS playback burst has finished.

        Returns immediately if nothing is being spoken right now.
        """
        if self._speech_done_event is not None:
            await self._speech_done_event.wait()

    async def stop(self) -> None:
        """
        Signal the speaker drain loop to stop and wait for it to finish.

        Drains any remaining audio before returning so that the last
        utterance is not cut off.
        """
        logger.info("TTSOutput: stopping …")
        await self._speaker.stop()

        if self._speaker_task and not self._speaker_task.done():
            self._speaker_task.cancel()
            try:
                await self._speaker_task
            except asyncio.CancelledError:
                pass
        logger.info("TTSOutput: stopped.")
