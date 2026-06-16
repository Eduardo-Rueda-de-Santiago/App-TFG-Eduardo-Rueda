"""
audio_speaker_piper.py — Concrete TTS speaker backed by PiperTTS (piper1-gpl).

Loads a voice model from a local ``.onnx`` + ``.onnx.json`` file pair at
construction time and streams synthesised audio directly to the system's
default output device via ``sounddevice``.  No audio is written to disk.

Dependencies
------------
    pip install piper-tts sounddevice numpy

Model files
-----------
Provide the two local files at construction:

    speaker = PiperSpeaker(
        input_queue=answer_queue,
        on_speaking_started=detector.on_speaking_started,
        on_speaking_stopped=detector.on_speaking_stopped,
        model_path="/path/to/en_US-amy-medium.onnx",
        config_path="/path/to/en_US-amy-medium.onnx.json",  # optional: auto-detected
    )

If ``config_path`` is omitted, the class looks for ``<model_path>.json``
next to the ``.onnx`` file, which is the standard Piper layout.

Playback strategy
-----------------
``PiperVoice.synthesize`` is a synchronous generator that yields
``AudioChunk`` objects containing raw int16 PCM bytes.  To avoid blocking
the asyncio event loop during inference and playback, the entire synthesis
+ playback job for one text chunk is run in a ``ThreadPoolExecutor`` via
``asyncio.get_event_loop().run_in_executor``.  This keeps ``generate_audio``
a proper awaitable while not starving other coroutines.

Within the executor thread, audio chunks are streamed to a
``sounddevice.OutputStream`` as they are produced, so the first audio
samples reach the speaker as soon as the first Piper chunk is ready —
without waiting for the full text to be synthesised first.

SynthesisConfig
---------------
Pass a ``piper.SynthesisConfig`` instance as ``syn_config`` to tune speed,
volume, noise, and normalisation.  Defaults to Piper's built-in settings.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig  # pip install piper-tts

from audio_speaker_generic import AudioSpeakerGeneric, InputQueue

logger = logging.getLogger(__name__)


class PiperSpeaker(AudioSpeakerGeneric):
    """
    TTS speaker that uses a locally-loaded PiperVoice model for synthesis
    and ``sounddevice`` for real-time audio output.

    Parameters
    ----------
    input_queue:
        Async queue of ``str | None`` chunks produced by the AI back-end.
        ``None`` is the end-of-response sentinel (see base class docs).
    on_speaking_started:
        Wired to ``AudioDetectorGeneric.on_speaking_started`` by the
        orchestrator.  Called once when TTS playback begins.
    on_speaking_stopped:
        Wired to ``AudioDetectorGeneric.on_speaking_stopped`` by the
        orchestrator.  Called once when the queue is drained and the last
        sample has been played.
    model_path:
        Absolute or relative path to the ``.onnx`` voice model file.
    config_path:
        Path to the companion ``.onnx.json`` config file.  If ``None``,
        defaults to ``<model_path>.json`` (standard Piper file layout).
    use_cuda:
        Use CUDA (GPU) for inference.  Requires ``onnxruntime-gpu``.
        Defaults to ``False`` (CPU inference).
    syn_config:
        Optional ``piper.SynthesisConfig`` to control speed, volume,
        and noise parameters.  ``None`` uses Piper's built-in defaults.
    device:
        ``sounddevice`` output device index or name.  ``None`` selects
        the system default output device.
    executor_workers:
        Maximum number of threads in the internal ``ThreadPoolExecutor``.
        Only one synthesis job runs at a time; this value rarely needs
        to be higher than ``1``.
    """

    def __init__(
        self,
        input_queue: InputQueue,
        on_speaking_started: Callable[[], None],
        on_speaking_stopped: Callable[[], None],
        *,
        model_path: str | os.PathLike,
        config_path: Optional[str | os.PathLike] = None,
        use_cuda: bool = False,
        syn_config: Optional[SynthesisConfig] = None,
        device: Optional[int | str] = None,
        executor_workers: int = 1,
    ) -> None:
        super().__init__(
            input_queue=input_queue,
            on_speaking_started=on_speaking_started,
            on_speaking_stopped=on_speaking_stopped,
        )

        self._model_path = Path(model_path)
        self._config_path = (
            Path(config_path)
            if config_path is not None
            else Path(str(model_path) + ".json")
        )
        self._use_cuda = use_cuda
        self._syn_config = syn_config
        self._device = device
        self._executor = ThreadPoolExecutor(
            max_workers=executor_workers,
            thread_name_prefix="piper_tts",
        )

        # Loaded lazily on first use — or eagerly via preload().
        self._voice: Optional[PiperVoice] = None

    # ------------------------------------------------------------------
    # Optional eager model load
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """
        Load the Piper voice model into memory immediately.

        Call this at startup if you want to pay the model-load cost up
        front rather than on the first ``generate_audio`` call.  Safe to
        call multiple times (idempotent).
        """
        if self._voice is None:
            self._voice = self._load_voice()

    # ------------------------------------------------------------------
    # AudioSpeakerGeneric implementation
    # ------------------------------------------------------------------

    async def generate_audio(self, input_text: str) -> None:
        """
        Synthesise ``input_text`` with PiperTTS and play it to completion.

        Runs the blocking Piper inference + sounddevice playback in a
        thread-pool executor so the asyncio event loop is not blocked.
        The coroutine returns only after the last audio sample has been
        written to the output device and the stream has been drained.

        Parameters
        ----------
        input_text:
            The cleaned text chunk to speak (whitespace already stripped
            by the base class before this method is called).
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._synthesize_and_play,
            input_text,
        )

    async def _teardown(self) -> None:
        """
        Shut down the thread-pool executor cleanly.

        Waits for any in-flight synthesis job to finish before returning.
        Safe to call more than once.
        """
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.debug("PiperSpeaker: executor shut down.")

    # ------------------------------------------------------------------
    # Private helpers (run in thread-pool, not on the event loop)
    # ------------------------------------------------------------------

    def _load_voice(self) -> PiperVoice:
        """
        Load and return the PiperVoice model.

        Validates that both the ``.onnx`` and ``.onnx.json`` files exist
        before attempting to load so that missing-file errors surface with
        a clear message rather than a cryptic ONNX runtime error.

        Returns
        -------
        PiperVoice
            Loaded and ready-to-use voice object.

        Raises
        ------
        FileNotFoundError
            If either the model or config file is not found on disk.
        """
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Piper model file not found: {self._model_path}\n"
                "Provide a local .onnx file via the model_path argument."
            )
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Piper config file not found: {self._config_path}\n"
                "Expected a companion .onnx.json file at the same location."
            )

        logger.info(
            "Loading PiperVoice from %s (cuda=%s) …",
            self._model_path,
            self._use_cuda,
        )
        voice = PiperVoice.load(
            str(self._model_path),
            config_path=str(self._config_path),
            use_cuda=self._use_cuda,
        )
        logger.info(
            "PiperVoice loaded. sample_rate=%d Hz",
            voice.config.sample_rate,
        )
        return voice

    def _synthesize_and_play(self, text: str) -> None:
        """
        Blocking synthesis + playback for a single text chunk.

        Intended to be run inside the ``ThreadPoolExecutor`` by
        ``generate_audio``.  Opens a ``sounddevice.OutputStream`` configured
        to match the voice's sample rate and channel count, then feeds each
        ``AudioChunk`` directly into the stream as it is produced by Piper.
        The stream is drained and closed before returning.

        Parameters
        ----------
        text:
            The text to synthesise and play.
        """
        # Lazy-load the voice on first call.
        if self._voice is None:
            self._voice = self._load_voice()

        voice = self._voice
        sample_rate: int = voice.config.sample_rate

        logger.debug("PiperSpeaker: synthesising %d chars …", len(text))

        # Open a raw int16 output stream.  Parameters are read from the
        # first AudioChunk but the stream must be opened before the loop,
        # so we rely on the config value (which always agrees with the chunks).
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=self._device,
        ) as stream:
            for chunk in voice.synthesize(text, syn_config=self._syn_config):
                # AudioChunk.audio_int16_bytes: raw little-endian int16 PCM
                pcm = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)

                # sounddevice expects shape (frames, channels) for multichannel
                # or (frames,) for mono.  Piper always produces mono.
                stream.write(pcm)

        logger.debug("PiperSpeaker: playback complete.")
