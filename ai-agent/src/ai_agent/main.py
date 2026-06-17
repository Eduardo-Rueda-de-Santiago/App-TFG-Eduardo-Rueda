"""
main.py - End-to-end voice agent pipeline.

Pipeline overview::

    Microphone
      -> WebRtcVadDetector        voice activity detection
           -> FasterWhisperTranscriber   speech to text
                 -> prompt_queue
                           -> PydanticAudioAgentLoop
                                 (Planner -> Tool Caller -> Conversational)
                                       -> response_queue
                                                 -> PiperSpeaker   text to speech
                                                           -> Speaker output

Each component is decoupled through async queues.
The agent (pydantic-ai) is pre-built with the model, tools, and system
prompt before being injected into the loop -- swapping the AI behaviour
only requires changing what is passed to PydanticAudioAgentLoop.

Configuration
-------------
All settings are read from environment variables or a .env file.
See ai_agent/config/llm_config.py and .env.example for the full list.

Minimum required for the llama-cpp provider (default)::

    AI_AGENT_PROVIDER=llama_cpp
    AI_AGENT_LLAMA_CPP__SERVER_URL=http://127.0.0.1:8080/v1
    AI_AGENT_LLAMA_CPP__MODEL_NAME=local-model

Start the llama-cpp OpenAI-compatible server separately::

    python -m llama_cpp.server \
        --model /path/to/model.gguf \
        --host 127.0.0.1 --port 8080 \
        --n_ctx 4096 --n_gpu_layers -1

Usage::

    python -m ai_agent.main

Press Ctrl+C to stop gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from ai_agent.agent_loop.agent_pydantic import PydanticAudioAgentLoop
from ai_agent.agent_loop.memory_game_agent import build_memory_game_agent
from ai_agent.audio_transcription.audio_transcriber_faster_whisper import (
    FasterWhisperTranscriber,
)
from ai_agent.config.llm_config import LLMConfig
from ai_agent.text_to_speech.audio_speaker_piper import PiperSpeaker
from ai_agent.voice_detection.audio_detector_webrtcvad import WebRtcVadDetector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
# Show DEBUG-level output for the agent loop so responses and history are visible.
logging.getLogger("ai_agent.agent_loop").setLevel(logging.DEBUG)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------

# Transcribed user prompts -> agent loop
prompt_queue: asyncio.Queue[str] = asyncio.Queue()

# Agent text chunks -> TTS speaker  (None = end-of-response sentinel)
response_queue: asyncio.Queue[str | None] = asyncio.Queue()


# ---------------------------------------------------------------------------
# Callbacks - wired between components for logging / state signalling
# ---------------------------------------------------------------------------


def on_speech_start() -> None:
    logger.info("[vad] speech start")


def on_speech_end(frames: list[bytes]) -> None:
    logger.info("[vad] speech end -- %d frames", len(frames))


def on_transcription_started() -> None:
    logger.info("[stt] transcribing ...")


def on_transcription_ready(prompt: str) -> None:
    logger.info("[stt] prompt ready: %r", prompt)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


async def _shutdown(
    detector: WebRtcVadDetector,
    agent_loop: PydanticAudioAgentLoop,
    speaker: PiperSpeaker,
) -> None:
    logger.info("[main] shutting down ...")
    await detector.stop()
    await agent_loop.stop()
    await speaker.stop()

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("[main] bye.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load LLM config and build the pydantic-ai model
    # ------------------------------------------------------------------
    llm_config = LLMConfig()
    logger.info("[main] LLM provider: %s", llm_config.provider.value)

    model = llm_config.build_model()

    # ------------------------------------------------------------------
    # 2. Build the memory-game agent (model + tools + system prompt).
    #    To swap the agent's behaviour, replace build_memory_game_agent()
    #    with any other factory that returns a pydantic-ai Agent.
    # ------------------------------------------------------------------
    agent = build_memory_game_agent(model)
    logger.info("[main] agent built")

    # ------------------------------------------------------------------
    # 3. Construct pipeline components
    # ------------------------------------------------------------------

    # TTS speaker -- consumes response_queue, plays audio
    speaker = PiperSpeaker(
        input_queue=response_queue,
        on_speaking_started=lambda: logger.info("[tts] speaking ..."),
        on_speaking_stopped=lambda: logger.info("[tts] done speaking"),
        # Point at one of the bundled voice models:
        model_path="models/en_US-amy-medium.onnx",
    )

    # Agent loop -- consumes prompt_queue, streams to response_queue
    agent_loop = PydanticAudioAgentLoop(
        agent=agent,
        input_queue=prompt_queue,
        output_queue=response_queue,
        sentence_chunk=True,
    )

    # STT transcriber -- receives audio frames from the detector
    transcriber = FasterWhisperTranscriber(
        output_queue=prompt_queue,
        on_transcription_started=on_transcription_started,
        on_transcription_ready=on_transcription_ready,
        model_size="small.en",
        device="cpu",
        compute_type="int8",
        language="en",
        beam_size=5,
    )
    transcriber.preload()

    # VAD detector -- opens the microphone, fires callbacks on speech
    detector = WebRtcVadDetector(
        on_speech_start=on_speech_start,
        on_speech_end=transcriber.on_audio_end,
        sample_rate=16_000,
        frame_duration_ms=30,
        energy_threshold=400.0,
        silence_hold_ms=800,
        vad_aggressiveness=2,
        min_speech_frames=3,
    )

    # Cross-wire VAD <-> speaker mute gate so TTS output is not re-transcribed.
    _orig_tts_start = speaker._on_speaking_started
    _orig_tts_stop = speaker._on_speaking_stopped

    def _on_tts_start() -> None:
        detector.on_speaking_started()
        _orig_tts_start()

    def _on_tts_stop() -> None:
        detector.on_speaking_stopped()
        _orig_tts_stop()

    speaker._on_speaking_started = _on_tts_start
    speaker._on_speaking_stopped = _on_tts_stop

    # ------------------------------------------------------------------
    # 4. Register graceful shutdown on Ctrl+C / SIGTERM
    # ------------------------------------------------------------------
    loop = asyncio.get_running_loop()

    def _sig_handler() -> None:
        asyncio.ensure_future(_shutdown(detector, agent_loop, speaker))

    # for sig in (signal.SIGINT, signal.SIGTERM):
    #     loop.add_signal_handler(sig, _sig_handler)

    # ------------------------------------------------------------------
    # 5. Run the pipeline -- all three coroutines run concurrently
    # ------------------------------------------------------------------
    logger.info("[main] pipeline ready -- speak now (Ctrl+C to stop)")

    try:
        await asyncio.gather(
            detector.start(),
            agent_loop.start(),
            speaker.start(),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    asyncio.run(main())
