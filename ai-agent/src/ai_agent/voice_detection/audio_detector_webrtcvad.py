"""
audio_detector_webrtcvad.py — Concrete VAD implementation using webrtcvad.

Captures raw PCM frames from the microphone via ``sounddevice``, classifies
each frame with Google's WebRTC VAD engine, and fires the speech-start /
speech-end callbacks defined by ``AudioDetectorGeneric``.

Algorithm
---------
1. A ``sounddevice.RawInputStream`` produces fixed-size PCM frames at the
   configured sample rate.
2. Each frame is classified by ``webrtcvad.Vad``.
3. A debounce counter smooths over momentary VAD misclassifications (common
   for soft or accented speech at lower aggressiveness levels).
4. Once ``min_speech_frames`` consecutive debounced-speech frames arrive,
   speech onset is declared and ``_emit_speech_start`` fires.
5. Frames are accumulated in an internal buffer together with a short
   ``pre_roll`` of frames captured just before onset, so the first syllable
   is never lost.
6. When the debounced counter drops back to zero AND the RMS of the
   accumulated buffer exceeds ``min_rms``, a silence hold-timer starts.
7. After ``silence_hold_ms`` ms of sustained silence the utterance is
   declared complete:  ``_emit_speech_end`` fires with all accumulated
   frames, and the detector resets to idle.
8. While muted (TTS playing) the mic keeps reading but all frames are
   discarded immediately, preventing buffer overflows and echo capture.

webrtcvad aggressiveness
------------------------
0 — most sensitive (fewest missed frames, most noise pass-through)
3 — most aggressive (fewest false positives, may clip soft speech)

The default is 1, which works well for typical close-talking microphone
setups. Raise to 2–3 in noisy environments.

Dependencies
------------
    pip install webrtc-noise-gain webrtcvad sounddevice numpy
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad

from ai_agent.voice_detection.audio_detector_generic import (
    AudioDetectorGeneric,
    SpeechEndCallback,
    SpeechStartCallback,
)

logger = logging.getLogger(__name__)


def _rms(pcm_bytes: bytes) -> float:
    """Return the root-mean-square amplitude of a raw int16 PCM buffer."""
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(arr**2))) if len(arr) else 0.0


class WebRtcVadDetector(AudioDetectorGeneric):
    """
    Voice activity detector backed by ``webrtcvad`` and ``sounddevice``.

    Parameters
    ----------
    on_speech_start:
        Callback fired once at speech onset (no arguments).
    on_speech_end:
        Callback fired once at utterance end with all accumulated frames.
    sample_rate:
        Microphone sample rate in Hz.  webrtcvad supports only
        8 000, 16 000, 32 000, and 48 000 Hz.
    frame_duration_ms:
        VAD frame duration in milliseconds.  webrtcvad supports only
        10, 20, or 30 ms.
    energy_threshold:
        Minimum RMS amplitude required before a completed audio buffer is
        forwarded to the transcriber.  Buffers below this value are almost
        certainly silence or electrical noise and are discarded silently.
        Maps to ``AudioDetectorGeneric.energy_threshold``.
    silence_hold_ms:
        Duration of sustained silence (ms) after which the utterance is
        considered complete and ``on_speech_end`` fires.
    vad_aggressiveness:
        webrtcvad aggressiveness level 0–3.  Lower values are more
        sensitive; higher values reject more background noise.
    min_speech_frames:
        Number of consecutive debounced-positive VAD frames required to
        declare speech onset.  Guards against transient noise spikes.
    pre_roll_frames:
        Number of frames to keep in the circular pre-roll buffer.
        These are prepended to the active buffer so that audio captured
        just before the VAD onset threshold is never lost.
    device:
        ``sounddevice`` input device index or name.  ``None`` selects the
        system default input device.
    """

    def __init__(
        self,
        on_speech_start: SpeechStartCallback,
        on_speech_end: SpeechEndCallback,
        *,
        sample_rate: int = 16_000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 400.0,
        silence_hold_ms: int = 1_200,
        vad_aggressiveness: int = 1,
        min_speech_frames: int = 3,
        pre_roll_frames: int = 20,
        device: int | str | None = None,
    ) -> None:
        super().__init__(
            on_speech_start,
            on_speech_end,
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
            silence_hold_ms=silence_hold_ms,
        )

        self.frame_duration_ms = frame_duration_ms
        self.frame_size: int = int(sample_rate * frame_duration_ms / 1000)
        self.vad_aggressiveness = vad_aggressiveness
        self.min_speech_frames = min_speech_frames
        self.pre_roll_frames = pre_roll_frames
        self.device = device

        self._vad = webrtcvad.Vad(vad_aggressiveness)

        # Pre-roll: circular buffer of recent frames captured before onset.
        # Prepended to audio_buffer when speech is declared so the first
        # syllable is never lost even if the VAD fires a few frames late.
        self._pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)

        # Frames accumulated during the current speech window.
        self._audio_buffer: list[bytes] = []

        # Debounce counter: increments on speech frames, decrements on silence.
        # Speech is declared when this reaches min_speech_frames.
        # Silence is declared when it returns to zero.
        self._consecutive_speech: int = 0

        # Whether we are currently inside a speech window.
        self._in_speech: bool = False

        # Monotonic timestamp of the last frame classified as speech.
        # Used to drive the silence hold-timer.
        self._last_speech_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Open the microphone and run the VAD frame loop until ``stop()`` is called.

        The blocking ``stream.read`` call is offloaded to a thread executor
        so the asyncio event loop remains responsive between frames.
        """
        self._running = True
        self._reset()
        loop = asyncio.get_event_loop()

        logger.info(
            "WebRtcVadDetector: opening mic — %d Hz, %d ms frames, aggressiveness=%d",
            self.sample_rate,
            self.frame_duration_ms,
            self.vad_aggressiveness,
        )

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            dtype="int16",
            channels=1,
            device=self.device,
        ) as stream:
            logger.info("WebRtcVadDetector: mic open, listening …")

            while self._running:
                data, _ = await loop.run_in_executor(None, stream.read, self.frame_size)
                pcm: bytes = bytes(data)

                # While TTS is playing: drain the mic buffer and do nothing.
                if self._muted:
                    continue

                self._process_frame(pcm)

        logger.info("WebRtcVadDetector: mic closed.")

    async def stop(self) -> None:
        """Signal the VAD loop to exit on the next iteration."""
        self._running = False

    # ------------------------------------------------------------------
    # Frame processing — single-responsibility helpers
    # ------------------------------------------------------------------

    def _process_frame(self, pcm: bytes) -> None:
        """
        Classify one PCM frame and advance the VAD state machine.

        This is the core of the detector.  It is kept as a synchronous
        method (not a coroutine) because it is called on every frame and
        must be as lightweight as possible.

        Parameters
        ----------
        pcm:
            Raw int16 PCM bytes for a single VAD frame.
        """
        is_speech = self._vad.is_speech(pcm, self.sample_rate)
        self._update_debounce(is_speech)

        if self._in_speech:
            self._accumulate(pcm)
            self._check_silence_timeout()
        else:
            self._pre_roll.append(pcm)
            if self._consecutive_speech >= self.min_speech_frames:
                self._enter_speech()

    def _update_debounce(self, is_speech: bool) -> None:
        """
        Smooth the raw VAD classification to absorb single-frame glitches.

        Increments on speech frames; decrements (floored at zero) on silence.
        A gradual decay means one non-speech frame mid-word does not snap the
        counter to zero and incorrectly end the speech window early.

        Parameters
        ----------
        is_speech:
            Raw frame classification from ``webrtcvad.Vad.is_speech``.
        """
        if is_speech:
            self._consecutive_speech += 1
            self._last_speech_time = time.monotonic()
        else:
            self._consecutive_speech = max(0, self._consecutive_speech - 1)

    def _enter_speech(self) -> None:
        """
        Transition from idle to speech-active state.

        Prepends the pre-roll frames so audio captured just before the VAD
        onset threshold is included in the final buffer.
        """
        logger.debug("WebRtcVadDetector: speech onset detected.")
        self._in_speech = True
        self._audio_buffer = list(self._pre_roll)
        self._pre_roll.clear()
        self._emit_speech_start()

    def _accumulate(self, pcm: bytes) -> None:
        """
        Append one frame to the active audio buffer.

        Also updates ``_last_speech_time`` when the frame's RMS exceeds the
        energy threshold, using RMS as a secondary voice-presence signal for
        frames that the VAD may have misclassified.  This prevents the
        silence timer from firing prematurely during soft or accented speech.

        Parameters
        ----------
        pcm:
            Raw int16 PCM bytes to append.
        """
        self._audio_buffer.append(pcm)
        if _rms(pcm) > self.energy_threshold:
            self._last_speech_time = time.monotonic()

    def _check_silence_timeout(self) -> None:
        """
        Finalise the utterance if silence has persisted past the hold window.

        Called on every frame while in speech-active state.  When the
        condition is met, emits the buffered frames and resets to idle.
        """
        silence_ms = (time.monotonic() - self._last_speech_time) * 1_000
        if self._consecutive_speech == 0 and silence_ms >= self.silence_hold_ms:
            self._finalise_utterance()

    def _finalise_utterance(self) -> None:
        """
        Complete the current utterance: validate the buffer and emit frames.

        A final RMS gate is applied to the entire buffer before forwarding it.
        Buffers that are below the energy threshold are almost certainly
        caused by the silence timer firing on an empty or near-empty window
        (e.g. the VAD fired on a transient noise that then immediately stopped).
        """
        frames = self._audio_buffer

        if frames:
            combined = b"".join(frames)
            if _rms(combined) >= self.energy_threshold:
                logger.debug(
                    "WebRtcVadDetector: utterance complete — %d frames, rms=%.1f",
                    len(frames),
                    _rms(combined),
                )
                self._emit_speech_end(frames)
            else:
                logger.debug(
                    "WebRtcVadDetector: utterance discarded — rms %.1f below threshold %.1f",
                    _rms(combined),
                    self.energy_threshold,
                )

        self._reset()

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Return the detector to its idle (pre-speech) state."""
        self._in_speech = False
        self._audio_buffer = []
        self._pre_roll.clear()
        self._consecutive_speech = 0
        self._last_speech_time = 0.0
