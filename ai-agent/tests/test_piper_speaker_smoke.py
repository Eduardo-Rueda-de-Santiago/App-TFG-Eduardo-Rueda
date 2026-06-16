"""
test_piper_speaker_smoke.py — Integration smoke test for PiperSpeaker.

What this test does
-------------------
1. Builds an asyncio.Queue and pre-loads it with the test sentence and a
   ``None`` sentinel (end-of-response marker).
2. Instantiates a PiperSpeaker pointed at a local model file.
3. Runs the speaker's drain loop until the queue is fully consumed, then
   stops it.
4. Asserts that the speaking-started and speaking-stopped callbacks were
   each called exactly once, proving the mute-gate lifecycle fired correctly.

This is an *integration* test — it loads the real Piper model and writes
real audio to the system's default output device.  It is not a unit test
with mocks; the goal is to confirm the full stack works end-to-end.

Usage
-----
    # Run directly:
    python test_piper_speaker_smoke.py

    # Or via pytest (pytest-asyncio is not required; the test runner is
    # embedded below):
    pytest test_piper_speaker_smoke.py -v -s

Configuration
-------------
Set the MODEL_PATH constant below to the absolute path of your local
``.onnx`` voice model before running.  The companion ``.onnx.json`` file
is expected in the same directory (standard Piper layout).
"""

from __future__ import annotations

import asyncio
import logging
import sys

# ---------------------------------------------------------------------------
# ⚙️  CONFIGURE THIS before running
# ---------------------------------------------------------------------------
MODEL_PATH: str = "./models/en_US-amy-medium.onnx"
# config_path is auto-detected as MODEL_PATH + ".json" — override if needed:
CONFIG_PATH: str | None = None
# ---------------------------------------------------------------------------

TEST_SENTENCE: str = "Model loaded and ready."

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("smoke_test")


async def run_smoke_test() -> None:
    """
    Core test coroutine.

    Drives the PiperSpeaker through one full speak → drain → stop cycle
    and verifies the callback contract.
    """
    # Import here so a missing piper/sounddevice install fails with a clear
    # ImportError rather than a confusing AttributeError at module level.
    from ai_agent.text_to_speech.audio_speaker_piper import PiperSpeaker

    # ------------------------------------------------------------------
    # 1. Build the input queue and pre-fill it.
    # ------------------------------------------------------------------
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    await queue.put(TEST_SENTENCE)  # the sentence to speak
    await queue.put(None)  # end-of-response sentinel → triggers stop

    # ------------------------------------------------------------------
    # 2. Callback tracking
    # ------------------------------------------------------------------
    started_calls: list[None] = []
    stopped_calls: list[None] = []

    def on_started() -> None:
        logger.info(">>> on_speaking_started fired")
        started_calls.append(None)

    def on_stopped() -> None:
        logger.info(">>> on_speaking_stopped fired")
        stopped_calls.append(None)

    # ------------------------------------------------------------------
    # 3. Instantiate the speaker
    # ------------------------------------------------------------------
    logger.info("Instantiating PiperSpeaker …")
    speaker = PiperSpeaker(
        input_queue=queue,  # type: ignore[arg-type]
        on_speaking_started=on_started,
        on_speaking_stopped=on_stopped,
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        use_cuda=False,
    )

    # Preload the model now so the timing log is clear.
    logger.info("Preloading voice model …")
    speaker.preload()
    logger.info("Voice model ready.")

    # ------------------------------------------------------------------
    # 4. Run the drain loop until the queue is empty, then stop.
    #
    #    strategy: run start() as a task, wait for the queue to be fully
    #    joined (all task_done() calls matched), then cancel the task and
    #    call stop() for a clean teardown.
    # ------------------------------------------------------------------
    logger.info("Starting drain loop …")
    drain_task = asyncio.create_task(speaker.start())

    # queue.join() blocks until every put() has a matching task_done().
    # The base class calls task_done() after each chunk (including the
    # None sentinel), so this unblocks as soon as the sentence has been
    # fully spoken and the sentinel processed.
    await queue.join()

    logger.info("Queue drained — stopping speaker …")
    drain_task.cancel()
    await speaker.stop()

    # Suppress the CancelledError that cancel() raises inside the task.
    try:
        await drain_task
    except asyncio.CancelledError:
        pass

    # ------------------------------------------------------------------
    # 5. Assertions
    # ------------------------------------------------------------------
    assert len(started_calls) == 1, (
        f"on_speaking_started should fire exactly once, got {len(started_calls)}"
    )
    assert len(stopped_calls) == 1, (
        f"on_speaking_stopped should fire exactly once, got {len(stopped_calls)}"
    )

    logger.info("✓ All assertions passed.")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def test_piper_speaker_speaks_sentence() -> None:
    """pytest-compatible wrapper (no pytest-asyncio needed)."""
    asyncio.run(run_smoke_test())


if __name__ == "__main__":
    print(f"\nSpeaking: {TEST_SENTENCE!r}\n")
    asyncio.run(run_smoke_test())
    print("\nDone.\n")
