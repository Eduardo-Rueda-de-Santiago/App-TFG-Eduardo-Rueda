"""
audio_detector_generic.py — Abstract base class for voice activity detection (VAD).

Defines the interface for any component whose sole responsibility is to decide
whether audio energy captured from a microphone constitutes "speech" or "silence".

Design contract
---------------
- The detector runs in its own continuous loop (``start`` / ``stop`` lifecycle).
- When speech *begins* it fires ``on_speech_start``.
- When speech *ends* (energy drops below a threshold for a configurable hold
  time) it fires ``on_speech_end``.
- It has **no knowledge** of transcription, queues, or TTS — it is a pure
  signal classifier.

Typical VAD back-ends
---------------------
- webrtcvad / silero-vad  (energy + ML frame classification)
- Fixed RMS energy threshold
- Push-to-talk (external GPIO / keypress signal)

Callback protocol
-----------------
``on_speech_start() -> None``
    Called once when the first speech frame is detected after a silence
    period. The transcriber uses this to open an audio buffer.

``on_speech_end(audio_frames: list[bytes]) -> None``
    Called once when silence has persisted long enough to consider the
    utterance complete. Receives the raw audio frames accumulated during
    the active speech window so the transcriber can work without needing
    its own buffering logic.

``on_speaking_started() / on_speaking_stopped()``
    Injected by the speaker to gate the detector while TTS is playing,
    preventing the AI's own voice from being mis-classified as user speech.
"""

from abc import ABC, abstractmethod
from typing import Callable


# Type aliases for readability
SpeechStartCallback = Callable[[], None]
SpeechEndCallback = Callable[[list[bytes]], None]


class AudioDetectorGeneric(ABC):
    """
    Abstract base class for voice activity detection (VAD).

    Parameters
    ----------
    on_speech_start:
        Called with no arguments the moment speech onset is detected.
    on_speech_end:
        Called with the accumulated audio frames when the utterance ends.
    sample_rate:
        Microphone sample rate in Hz (e.g. 16000). Passed down to
        concrete implementations so they can configure their VAD engine.
    energy_threshold:
        RMS / energy value below which audio is considered silence.
        Concrete implementations may interpret this differently (e.g. as
        a webrtcvad aggressiveness level or a raw amplitude floor).
    silence_hold_ms:
        How long (in milliseconds) energy must remain below
        ``energy_threshold`` before the utterance is declared over and
        ``on_speech_end`` fires.  Defaults to 800 ms — long enough to
        handle natural pauses without cutting off the speaker.
    """

    def __init__(
        self,
        on_speech_start: SpeechStartCallback,
        on_speech_end: SpeechEndCallback,
        *,
        sample_rate: int = 16_000,
        energy_threshold: float = 300.0,
        silence_hold_ms: int = 800,
    ) -> None:
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_hold_ms = silence_hold_ms

        self._running: bool = False
        # Gate flag: set to True by the speaker while TTS is playing so the
        # detector can suppress its own loop without being stopped entirely.
        self._muted: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """
        Open the microphone stream and begin the VAD loop.

        This is a long-running coroutine.  It should continuously read
        audio frames from the microphone, classify them, and fire the
        appropriate callback when speech starts or ends.

        Implementations must honour ``self._running`` and exit cleanly
        when it is set to ``False`` via ``stop()``.

        Implementations must also honour ``self._muted``:  when True,
        frames should be read and discarded without classification so
        that the mic buffer does not overflow and TTS audio is not
        mis-classified.
        """

    @abstractmethod
    async def stop(self) -> None:
        """
        Signal the VAD loop to exit and release the microphone resource.

        Must be idempotent (safe to call more than once).
        """

    # ------------------------------------------------------------------
    # Speaker gate — called by AudioSpeakerGeneric
    # ------------------------------------------------------------------

    def on_speaking_started(self) -> None:
        """
        Mute the detector while the TTS speaker is active.

        Called by the speaker at the start of audio playback.
        While muted the detector keeps the mic open (to avoid buffer
        overflows) but discards all frames and will not fire any callbacks.
        """
        self._muted = True

    def on_speaking_stopped(self) -> None:
        """
        Unmute the detector once TTS playback has finished.

        Called by the speaker when the last audio chunk has been played.
        Normal VAD classification resumes immediately.
        """
        self._muted = False

    # ------------------------------------------------------------------
    # Internal helpers for subclasses
    # ------------------------------------------------------------------

    def _emit_speech_start(self) -> None:
        """
        Fire the speech-start callback.

        Subclasses should call this rather than invoking
        ``self._on_speech_start`` directly so that the mute gate is
        respected and the call site stays free of guard logic.
        """
        if not self._muted:
            self._on_speech_start()

    def _emit_speech_end(self, audio_frames: list[bytes]) -> None:
        """
        Fire the speech-end callback with the captured frames.

        Subclasses should call this rather than invoking
        ``self._on_speech_end`` directly.

        Parameters
        ----------
        audio_frames:
            Raw PCM frames (bytes) that make up the complete utterance,
            in chronological order.
        """
        if not self._muted and audio_frames:
            self._on_speech_end(audio_frames)
