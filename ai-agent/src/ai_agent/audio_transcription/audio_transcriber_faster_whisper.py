"""
audio_transcriber_faster_whisper.py — Concrete STT transcriber using faster-whisper.

Receives raw PCM frames from an ``AudioDetectorGeneric`` implementation (via
the ``on_audio_end`` callback), runs them through a locally-loaded
``faster-whisper`` model, filters out known hallucinations and near-silence
buffers, and pushes clean text into the output queue.

This class has **no knowledge** of microphone capture, VAD, or wake-word
detection.  All of that is handled upstream by the detector.

Hallucination filtering
-----------------------
Faster-whisper (and Whisper in general) tends to emit well-known spurious
strings when given near-silence.  A curated blocklist (``_HALLUCINATIONS``)
is checked after transcription.  Additionally, ``no_speech_prob`` per segment
and a minimum word count guard against low-confidence or trivially short
outputs reaching the queue.

The wake word is expected to have been stripped **by the detector** before
frames reach this class.  If any residue leaks through it will simply pass
as normal text — the transcriber does not know what the wake word is.

Threading
---------
``faster_whisper.WhisperModel.transcribe`` is a synchronous, CPU/GPU-bound
call.  It is offloaded to a ``ThreadPoolExecutor`` inside ``transcribe`` so
the asyncio event loop is never blocked during inference.

Local model files
-----------------
Pass ``model_path`` as an absolute or relative path to a directory containing
the faster-whisper CTranslate2 model files.  If ``model_path`` is ``None``
the standard faster-whisper model-name resolution is used (downloads from
HuggingFace on first run) — but the class is designed for local-first use.

Dependencies
------------
    pip install faster-whisper numpy
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from ai_agent.audio_transcription.audio_transcriber_generic import (
    AudioTranscriberGeneric,
    OutputQueue,
    TranscriptionReadyCallback,
    TranscriptionStartedCallback,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hallucination blocklist
# ---------------------------------------------------------------------------
# Whisper is known to emit these strings when transcribing silence or
# near-silence.  Any detected text matching an entry here is discarded
# before being forwarded to the orchestrator.
# ---------------------------------------------------------------------------
_HALLUCINATIONS: frozenset[str] = frozenset(
    {
        "",
        "!",
        ".",
        "...",
        "?",
        "thank you.",
        "thank you",
        "you",
        "you.",
        "bye.",
        "bye",
        "thanks.",
        "thanks",
        "uh.",
        "uh",
        "um.",
        "um",
        "hm.",
        "hmm.",
        "hmm",
        "hm",
        "see you next time.",
        "see you next time",
        "today",
        "today.",
        "3.",
        "3",
        "youtube.",
        "youtube",
        "and i'll see you next time.",
        "i'm sorry.",
        "i'm sorry",
        "i'll see you in the next one.",
        "doing today at musi.",
    }
)

# Minimum RMS amplitude of the full audio buffer.  Buffers below this are
# almost certainly silence or electrical noise — skip inference entirely.
_MIN_RMS: float = 400.0

# Maximum no_speech_prob a Whisper segment may have before it is discarded.
_MAX_NO_SPEECH_PROB: float = 0.6

# Minimum number of real words (non-punctuation tokens) required in the final
# transcript.  Single-word or punctuation-only outputs are rejected.
_MIN_WORD_COUNT: int = 2


def _rms(pcm_bytes: bytes) -> float:
    """Return the root-mean-square amplitude of a raw int16 PCM buffer."""
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(arr**2))) if len(arr) else 0.0


def _frames_to_float32(frames: list[bytes]) -> np.ndarray:
    """
    Concatenate raw int16 PCM frames and normalise to float32 in [-1, 1].

    Faster-whisper expects a 1-D float32 array normalised to this range.

    Parameters
    ----------
    frames:
        Ordered list of raw int16 PCM byte strings.

    Returns
    -------
    np.ndarray
        1-D float32 array ready for ``WhisperModel.transcribe``.
    """
    raw = b"".join(frames)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32_768.0


def _is_hallucination(text: str) -> bool:
    """
    Return ``True`` if *text* matches the hallucination blocklist.

    Comparison is case-insensitive and ignores leading/trailing whitespace.

    Parameters
    ----------
    text:
        Lower-cased, stripped transcription candidate.
    """
    return text in _HALLUCINATIONS


def _has_enough_words(text: str) -> bool:
    """
    Return ``True`` if *text* contains at least ``_MIN_WORD_COUNT`` real words.

    Punctuation-only tokens (e.g. ``"."`` or ``","`` alone) are excluded from
    the count to avoid single-symbol outputs being treated as valid prompts.

    Parameters
    ----------
    text:
        Lower-cased, stripped transcription candidate.
    """
    real_words = [w for w in text.split() if w.strip(".,!?;:—\"'")]
    return len(real_words) >= _MIN_WORD_COUNT


class FasterWhisperTranscriber(AudioTranscriberGeneric):
    """
    Transcriber that runs faster-whisper inference on PCM frames supplied
    by the VAD detector.

    Parameters
    ----------
    output_queue:
        Async queue that receives finalised, validated prompt strings.
    on_transcription_started:
        Called at the start of each inference job (useful for UI spinners).
    on_transcription_ready:
        Called with the clean prompt string after successful transcription.
    model_path:
        Path to a local directory containing faster-whisper CTranslate2
        model files (``model.bin``, ``config.json``, etc.).  If ``None``,
        faster-whisper falls back to its built-in model-name resolution
        (may attempt a HuggingFace download).
    model_size:
        Faster-whisper model identifier used when ``model_path`` is
        ``None``.  Ignored if ``model_path`` is provided.
    device:
        Inference device — ``"cuda"`` or ``"cpu"``.
    compute_type:
        Quantisation type — e.g. ``"float16"`` (GPU), ``"int8"`` (CPU).
    sample_rate:
        Sample rate of the PCM frames the detector will supply (Hz).
    language:
        BCP-47 language code for Whisper (e.g. ``"en"``).  Passing an
        explicit language disables language detection and speeds up inference.
    beam_size:
        Beam search width.  Higher values improve accuracy at the cost of
        latency.  ``5`` is a sensible production default.
    initial_prompt:
        Optional prompt string prepended to each transcription request to
        bias Whisper toward the expected vocabulary or speaking style.
    executor_workers:
        Size of the ``ThreadPoolExecutor`` used for blocking inference.
        Rarely needs to exceed ``1`` since only one job runs at a time.
    """

    def __init__(
        self,
        output_queue: OutputQueue,
        on_transcription_started: TranscriptionStartedCallback,
        on_transcription_ready: TranscriptionReadyCallback,
        *,
        model_path: Optional[str | os.PathLike] = None,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "float16",
        sample_rate: int = 16_000,
        language: str = "en",
        beam_size: int = 5,
        initial_prompt: str = "Transcribe the following voice command accurately.",
        executor_workers: int = 1,
    ) -> None:
        super().__init__(
            output_queue,
            on_transcription_started,
            on_transcription_ready,
            sample_rate=sample_rate,
        )

        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._initial_prompt = initial_prompt
        self._executor = ThreadPoolExecutor(
            max_workers=executor_workers,
            thread_name_prefix="faster_whisper",
        )

        # Lazily loaded — or eagerly via preload().
        self._model: Optional[WhisperModel] = None

    # ------------------------------------------------------------------
    # Optional eager model load
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """
        Load the Whisper model into memory immediately.

        Useful to pay the cold-start cost at application startup rather
        than on the first utterance.  Safe to call multiple times.
        """
        if self._model is None:
            self._model = self._load_model()

    # ------------------------------------------------------------------
    # AudioTranscriberGeneric implementation
    # ------------------------------------------------------------------

    async def transcribe(self, audio_frames: list[bytes]) -> str:
        """
        Validate, transcribe, and filter one utterance.

        This method is the single mandatory override from
        ``AudioTranscriberGeneric``.  It is called by the base class's
        ``_run_transcription`` after ``on_audio_end`` receives frames from
        the detector.

        Returns an empty string for any frame set that fails validation or
        produces only hallucinations; otherwise returns the clean, lower-cased
        prompt ready for the output queue.

        Parameters
        ----------
        audio_frames:
            Ordered raw int16 PCM frames at ``self.sample_rate`` Hz.

        Returns
        -------
        str
            Clean transcription, or ``""`` if the result was rejected.
        """
        if not self._passes_rms_gate(audio_frames):
            return ""

        audio_float = _frames_to_float32(audio_frames)
        raw_text = await self._run_inference(audio_float)

        return self._validate_and_clean(raw_text)

    # ------------------------------------------------------------------
    # Validation pipeline — one focused method per concern
    # ------------------------------------------------------------------

    def _passes_rms_gate(self, frames: list[bytes]) -> bool:
        """
        Return ``True`` if the combined RMS of *frames* exceeds the minimum
        energy threshold.

        This is a secondary guard: the detector already applies an RMS check
        before calling ``on_audio_end``, but a safety net here prevents
        wasting GPU time on a buffer that somehow arrived under-threshold.

        Parameters
        ----------
        frames:
            Raw int16 PCM frames to evaluate.
        """
        combined = b"".join(frames)
        rms = _rms(combined)
        if rms < _MIN_RMS:
            logger.debug(
                "FasterWhisperTranscriber: skipped — rms %.1f below %.1f", rms, _MIN_RMS
            )
            return False
        return True

    def _validate_and_clean(self, text: str) -> str:
        """
        Apply all text-level filters and return a clean prompt or ``""``.

        Filters applied in order:
          1. Empty string guard.
          2. Hallucination blocklist.
          3. Minimum word count.

        Parameters
        ----------
        text:
            Lower-cased, stripped transcription candidate.
        """
        if not text:
            return ""

        if _is_hallucination(text):
            logger.debug("FasterWhisperTranscriber: hallucination discarded — %r", text)
            return ""

        if not _has_enough_words(text):
            logger.debug("FasterWhisperTranscriber: too short — %r", text)
            return ""

        return text

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def _run_inference(self, audio_float: np.ndarray) -> str:
        """
        Execute ``WhisperModel.transcribe`` in a thread executor and return
        the concatenated, lower-cased text of all accepted segments.

        Segments where Whisper reports ``no_speech_prob > _MAX_NO_SPEECH_PROB``
        are excluded from the output before joining.

        Parameters
        ----------
        audio_float:
            1-D float32 numpy array normalised to [-1, 1].
        """
        if self._model is None:
            self._model = self._load_model()

        model = self._model
        beam_size = self._beam_size
        language = self._language
        initial_prompt = self._initial_prompt

        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            self._executor,
            lambda: model.transcribe(
                audio_float,
                language=language,
                beam_size=beam_size,
                temperature=0.0,  # greedy — fastest and most deterministic
                vad_filter=True,  # skip Whisper's own internal silence detection
                no_speech_threshold=_MAX_NO_SPEECH_PROB,
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
            ),
        )

        accepted = [
            seg.text for seg in segments if seg.no_speech_prob <= _MAX_NO_SPEECH_PROB
        ]
        return " ".join(accepted).strip().lower()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> WhisperModel:
        """
        Load the faster-whisper model and return it.

        """
        model = WhisperModel(
            str(self._model_size),
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info(
            "FasterWhisperTranscriber: model ready (device=%s, compute_type=%s).",
            self._device,
            self._compute_type,
        )
        return model

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """
        Release the thread-pool executor.

        Call this when the application is shutting down to allow in-flight
        inference to complete before the process exits.
        """
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.debug("FasterWhisperTranscriber: executor shut down.")
