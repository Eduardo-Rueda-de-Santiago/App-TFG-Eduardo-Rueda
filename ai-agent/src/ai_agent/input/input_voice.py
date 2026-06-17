"""
input_voice.py — Voice (microphone) input implementation.

Wires a ``WebRtcVadDetector`` and a ``FasterWhisperTranscriber`` together
into the ``InputGeneric`` interface.  Transcribed utterances are pushed into
an internal ``asyncio.Queue``; ``get_next_prompt`` awaits items from it.

Pipeline inside this class
--------------------------
::

    Microphone (sounddevice)
        │
        ▼
    WebRtcVadDetector          — frame-level speech / silence classification
        │  on_speech_end(frames)
        ▼
    FasterWhisperTranscriber   — offline CTranslate2 Whisper inference
        │  on_transcription_ready(text)
        ▼
    asyncio.Queue              — bridges the async world of the VAD/transcriber
        │                         with the synchronous pipeline
        ▼
    get_next_prompt() → str    — consumed by the agent loop

Mute gating
-----------
``on_speaking_started`` / ``on_speaking_stopped`` are exposed so that a TTS
output module can mute the microphone while the agent is speaking.  This
prevents the agent's own voice from being re-transcribed as a new prompt.
Wire them to the TTS output's callbacks in the application entry point:

    tts_output = TTSOutput(
        on_speaking_started=voice_input.on_speaking_started,
        on_speaking_stopped=voice_input.on_speaking_stopped,
    )

Dependencies
------------
    pip install webrtcvad-wheels sounddevice faster-whisper numpy
"""

from __future__ import annotations

import asyncio
import logging

from ai_agent.audio_transcription.audio_transcriber_faster_whisper import (
    FasterWhisperTranscriber,
)
from ai_agent.config import (
    VAD_AGGRESSIVENESS,
    VAD_ENERGY_THRESHOLD,
    VAD_SAMPLE_RATE,
    VAD_SILENCE_HOLD_MS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
)
from ai_agent.input.input_generic import InputGeneric
from ai_agent.voice_detection.audio_detector_webrtcvad import WebRtcVadDetector

logger = logging.getLogger(__name__)


class VoiceInput(InputGeneric):
    """
    Microphone-based input that runs VAD + Whisper transcription.

    Parameters
    ----------
    vad_aggressiveness:
        webrtcvad aggressiveness 0–3.  Lower = more sensitive.
    vad_sample_rate:
        Microphone sample rate in Hz.  Must be one of 8000, 16000, 32000, 48000.
    vad_silence_hold_ms:
        Milliseconds of silence required to finalise an utterance.
    vad_energy_threshold:
        Minimum RMS amplitude; buffers below this are discarded as noise.
    whisper_model_size:
        faster-whisper model identifier or path to a local CTranslate2 dir.
    whisper_device:
        Inference device for Whisper: ``"cuda"`` or ``"cpu"``.
    whisper_compute_type:
        Quantisation type: e.g. ``"float16"`` (GPU) or ``"int8"`` (CPU).
    whisper_language:
        BCP-47 language code (e.g. ``"en"``).  Explicit value disables
        per-utterance language auto-detection.
    mic_device:
        ``sounddevice`` input device index or name.  ``None`` selects the
        system default microphone.
    """

    def __init__(
        self,
        *,
        vad_aggressiveness: int = VAD_AGGRESSIVENESS,
        vad_sample_rate: int = VAD_SAMPLE_RATE,
        vad_silence_hold_ms: int = VAD_SILENCE_HOLD_MS,
        vad_energy_threshold: float = VAD_ENERGY_THRESHOLD,
        whisper_model_size: str = WHISPER_MODEL_SIZE,
        whisper_device: str = WHISPER_DEVICE,
        whisper_compute_type: str = WHISPER_COMPUTE_TYPE,
        whisper_language: str = WHISPER_LANGUAGE,
        mic_device: int | str | None = None,
    ) -> None:
        # Internal queue: transcribed prompts flow here
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Background task handle for the VAD detector loop
        self._detector_task: asyncio.Task | None = None

        # Transcriber — wired to put results into _queue
        self._transcriber = FasterWhisperTranscriber(
            output_queue=self._queue,
            on_transcription_started=self._on_transcription_started,
            on_transcription_ready=self._on_transcription_ready,
            model_size=whisper_model_size,
            device=whisper_device,
            compute_type=whisper_compute_type,
            sample_rate=vad_sample_rate,
            language=whisper_language,
        )

        # VAD detector — wired to the transcriber's callbacks
        self._detector = WebRtcVadDetector(
            on_speech_start=self._transcriber.on_audio_start,
            on_speech_end=self._transcriber.on_audio_end,
            sample_rate=vad_sample_rate,
            energy_threshold=vad_energy_threshold,
            silence_hold_ms=vad_silence_hold_ms,
            vad_aggressiveness=vad_aggressiveness,
            device=mic_device,
        )

    # ------------------------------------------------------------------
    # InputGeneric interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Pre-warm the Whisper model and start the VAD microphone loop.

        The VAD detector is launched as a background asyncio Task so that
        ``get_next_prompt`` can be awaited concurrently.
        """
        logger.info("VoiceInput: pre-loading Whisper model …")
        self._transcriber.preload()
        logger.info("VoiceInput: Whisper model ready.  Opening microphone …")

        self._detector_task = asyncio.create_task(
            self._detector.start(), name="vad-detector"
        )
        print("\n[Voice] Microphone open — speak now.  (Ctrl-C to exit)\n")

    async def get_next_prompt(self) -> str | None:
        """
        Await the next transcribed utterance.

        Blocks until the Whisper transcriber produces a non-empty string and
        puts it in the internal queue.

        Returns
        -------
        str
            The cleaned, lower-cased transcription ready for the pipeline.
        None
            Returned when a ``None`` sentinel is dequeued (signals shutdown).
        """
        prompt = await self._queue.get()
        self._queue.task_done()
        return prompt

    async def stop(self) -> None:
        """
        Stop the VAD detector, flush the transcriber, and clean up resources.
        """
        logger.info("VoiceInput: stopping …")

        # Signal the detector loop to exit
        await self._detector.stop()

        # Cancel the background task if it is still running
        if self._detector_task and not self._detector_task.done():
            self._detector_task.cancel()
            try:
                await self._detector_task
            except asyncio.CancelledError:
                pass

        # Flush the transcriber thread pool
        self._transcriber.shutdown()
        logger.info("VoiceInput: stopped.")

    # ------------------------------------------------------------------
    # Mute gating — wire to TTS output callbacks
    # ------------------------------------------------------------------

    def on_speaking_started(self) -> None:
        """
        Mute the microphone while TTS audio is playing.

        Wire this to the TTS output's ``on_speaking_started`` callback so
        that the agent's own voice is not re-transcribed as a new prompt.
        The VAD detector discards all frames while muted but keeps the mic
        open to prevent buffer overflows.
        """
        self._detector.on_speaking_started()

    def on_speaking_stopped(self) -> None:
        """
        Unmute the microphone after TTS playback finishes.

        Wire this to the TTS output's ``on_speaking_stopped`` callback.
        Normal VAD classification resumes immediately.
        """
        self._detector.on_speaking_stopped()

    # ------------------------------------------------------------------
    # Private transcriber hooks
    # ------------------------------------------------------------------

    def _on_transcription_started(self) -> None:
        """
        Called by the transcriber when inference begins.

        Logs progress so the user can see the pipeline is working.
        """
        logger.debug("VoiceInput: transcription started …")
        print("[Voice] Transcribing …", flush=True)

    def _on_transcription_ready(self, prompt: str) -> None:
        """
        Called by the transcriber when a clean prompt is available.

        The base-class ``_run_transcription`` also puts the prompt into the
        queue; this hook is used only for logging / UI feedback.

        Parameters
        ----------
        prompt:
            The clean, lower-cased transcription string.
        """
        logger.debug("VoiceInput: transcription ready — %r", prompt)
        print(f"[Voice] You said: {prompt!r}", flush=True)
