"""
test_pipeline.py — End-to-end integration test for the VAD + STT pipeline.

Wires WebRtcVadDetector and FasterWhisperTranscriber together, opens the
microphone, and prints every transcribed prompt to the terminal.

Usage
-----
1. Edit MODEL_PATH below to point at your local faster-whisper model directory.
2. Run:
       python test_pipeline.py
3. Speak into your microphone.  Transcribed prompts appear in the terminal.
4. Press Ctrl+C to stop.

Expected terminal output
------------------------
    [pipeline] loading models ...
    [pipeline] ready — speak now, Ctrl+C to stop
    [detector] mic open
    [detector] speech start
    [detector] speech end — 42 frames
    [transcriber] transcribing ...
    >>> hello, how are you doing today

Tuning
------
If the detector is too sensitive or not sensitive enough, adjust:
    - vad_aggressiveness  (0 = most sensitive, 3 = least)
    - energy_threshold    (RMS floor; raise in noisy environments)
    - silence_hold_ms     (how long to wait after speech stops)

If transcription is slow on CPU, try a smaller model such as "small" or
"base" and set compute_type="int8".
"""

from __future__ import annotations

import asyncio
import logging
import signal

from ai_agent.audio_transcription.audio_transcriber_faster_whisper import (
    FasterWhisperTranscriber,
)
from ai_agent.voice_detection.audio_detector_webrtcvad import WebRtcVadDetector

# ---------------------------------------------------------------------------
# Logging — set to DEBUG to see per-frame VAD decisions and RMS values.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_pipeline")


# ---------------------------------------------------------------------------
# Simple asyncio.Queue that satisfies the OutputQueue protocol.
# ---------------------------------------------------------------------------
prompt_queue: asyncio.Queue[str] = asyncio.Queue()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def on_speech_start() -> None:
    print("[detector] speech start")


def on_speech_end(frames: list[bytes]) -> None:
    print(f"[detector] speech end — {len(frames)} frames")


def on_transcription_started() -> None:
    print("[transcriber] transcribing ...")


def on_transcription_ready(prompt: str) -> None:
    # The base class also puts the text in the queue; this callback is used
    # for the immediate print so the user sees feedback before the queue
    # consumer coroutine gets a chance to run.
    print(f">>> {prompt}")


# ---------------------------------------------------------------------------
# Queue consumer — drains prompts and prints them.
# Redundant with on_transcription_ready here, but mirrors real usage where
# the AI back-end would consume from this queue instead.
# ---------------------------------------------------------------------------


async def drain_queue() -> None:
    """Consume prompts from the queue and echo them to the terminal."""
    while True:
        prompt = await prompt_queue.get()
        logger.debug("queue drained: %r", prompt)
        prompt_queue.task_done()


# ---------------------------------------------------------------------------
# Graceful shutdown on Ctrl+C
# ---------------------------------------------------------------------------


async def shutdown(detector: WebRtcVadDetector) -> None:
    """Stop the detector and cancel all pending tasks cleanly."""
    print("\n[pipeline] shutting down ...")
    await detector.stop()

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    print("[pipeline] bye.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:

    transcriber = FasterWhisperTranscriber(
        output_queue=prompt_queue,
        on_transcription_started=on_transcription_started,
        on_transcription_ready=on_transcription_ready,
        model_size="base.en",
        device="cpu",
        compute_type="int8",
        language="en",
        beam_size=5,
    )

    transcriber.preload()

    detector = WebRtcVadDetector(
        on_speech_start=on_speech_start,
        on_speech_end=transcriber.on_audio_end,
        sample_rate=16_000,
        frame_duration_ms=30,
        energy_threshold=400.0,
        silence_hold_ms=200,
        vad_aggressiveness=2,
        min_speech_frames=3,
    )

    print("[pipeline] ready — speak now, Ctrl+C to stop")

    try:
        await asyncio.gather(
            detector.start(),
            drain_queue(),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await shutdown(detector)


if __name__ == "__main__":
    asyncio.run(main())
