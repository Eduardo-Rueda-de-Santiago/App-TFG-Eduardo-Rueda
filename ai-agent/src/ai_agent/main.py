"""
main.py — Entry point for the AI agent.

Supports two input modes and two output modes, selectable via CLI flags:

Input modes
-----------
  ``terminal`` (default)
      Read prompts from stdin.  Supports both an interactive REPL and a
      single-prompt ``--prompt`` run.

  ``voice``
      Capture audio from the microphone, detect speech with WebRTC VAD,
      and transcribe with faster-whisper.  Runs a continuous loop:
      listen → transcribe → pipeline → respond → repeat.

Output modes
------------
  ``terminal`` (default)
      Print the formatted response to stdout.

  ``tts``
      Speak the response aloud using a local Piper voice model.

Usage examples
--------------
::

    # Interactive terminal REPL (default)
    python -m ai_agent

    # Single-shot terminal run
    python -m ai_agent --prompt "flip card 7"

    # Voice input, spoken output (full hands-free loop)
    python -m ai_agent --input voice --output tts

    # Voice input, terminal output  (useful for debugging transcription)
    python -m ai_agent --input voice --output terminal

    # Debug mode — also dumps full JSON pipeline output after every run
    python -m ai_agent --debug
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from ai_agent.input.input_generic import InputGeneric
from ai_agent.input.input_terminal import TerminalInput
from ai_agent.input.input_voice import VoiceInput
from ai_agent.model_manager import ModelManager
from ai_agent.orchestrator import run_pipeline
from ai_agent.output.output_generic import OutputGeneric
from ai_agent.output.output_terminal import TerminalOutput
from ai_agent.output.output_tts import TTSOutput
from ai_agent.schemas import PipelineOutput, PipelineTiming


# =============================================================================
# Timing table
# =============================================================================


def _print_timing_table(timing: PipelineTiming) -> None:
    """
    Print a formatted table of per-step timing to stdout.

    Always printed to stdout regardless of the active output mode so that
    the values are never read aloud by the TTS engine.
    """
    # Each row: (label, elapsed_seconds, tokens_per_second | None)
    rows: list[tuple[str, float | None, float | None]] = [
        ("1. Prompt processing",           timing.prompt_processing,  None),
        ("2. Brain (LLM)",                 timing.brain,              timing.brain_tps),
        ("3. Tool calling",                timing.tool_calling,       timing.tool_calling_tps),
        ("4. Communicator (LLM)",          timing.communicator,       timing.communicator_tps),
        ("5a. TTS synthesis (ONNX)",       timing.tts_synthesis,      None),
        ("5b. TTS playback (audio out)",   timing.tts_playback,       None),
    ]

    label_w = 32
    time_w  = 18  # wide enough for "14 tok/s 7.895 s"

    def fmt_time(t: float | None, tps: float | None) -> str:
        if t is None:
            return "—  skipped  —"
        time_str = f"{t:.3f} s"
        if tps is not None:
            return f"{tps:.0f} tok/s {time_str}"
        return time_str

    h_sep = "╠" + "═" * (label_w + 2) + "╪" + "═" * (time_w + 2) + "╣"
    top   = "╔" + "═" * (label_w + 2) + "╤" + "═" * (time_w + 2) + "╗"
    bot   = "╚" + "═" * (label_w + 2) + "╧" + "═" * (time_w + 2) + "╝"
    mid   = "╟" + "─" * (label_w + 2) + "┼" + "─" * (time_w + 2) + "╢"

    print("\n" + top)
    print(f"║ {'Stage':<{label_w}} │ {'Time':>{time_w}} ║")
    print(h_sep)
    for i, (label, t, tps) in enumerate(rows):
        print(f"║ {label:<{label_w}} │ {fmt_time(t, tps):>{time_w}} ║")
        if i < len(rows) - 1:
            print(mid)
    print(h_sep)
    print(f"║ {'Pipeline total (no TTS)':<{label_w}} │ {f'{timing.pipeline_total:.3f} s':>{time_w}} ║")
    tts_used = timing.tts_synthesis is not None or timing.tts_playback is not None
    if tts_used:
        print(mid)
        print(f"║ {'Grand total':<{label_w}} │ {f'{timing.total:.3f} s':>{time_w}} ║")
    print(bot + "\n")


# =============================================================================
# Agent loop
# =============================================================================


async def run_agent_loop(
    manager: ModelManager,
    input_: InputGeneric,
    output: OutputGeneric,
    *,
    debug: bool = False,
) -> None:
    """
    Main agent loop: repeatedly get a prompt, run the pipeline, emit a response.

    The pipeline (``run_pipeline``) is synchronous (blocking Llama inference).
    It is offloaded to a thread executor so that the asyncio event loop stays
    responsive during inference — particularly important in voice mode where
    the VAD detector and TTS drain loop must keep running concurrently.

    Parameters
    ----------
    manager:
        A ``ModelManager`` with all three models already loaded.
    input_:
        The input source to read prompts from.
    output:
        The output channel to send responses to.
    debug:
        If ``True``, also print the full JSON pipeline output after each run.
    """
    loop = asyncio.get_event_loop()

    await input_.start()
    await output.start()

    try:
        while True:
            prompt = await input_.get_next_prompt()
            if prompt is None:
                break

            try:
                # Run blocking LLM inference in a thread so the event loop
                # is free for the VAD detector / TTS drain task.
                pipeline_out: PipelineOutput = await loop.run_in_executor(
                    None, run_pipeline, manager, prompt
                )
            except Exception as exc:
                print(f"\n✗ Pipeline error: {exc}", file=sys.stderr)
                print(
                    "  (Check that the Memory Game backend is running on :4000)\n",
                    file=sys.stderr,
                )
                continue

            await output.emit(
                pipeline_out.final_response.message,
                task_achieved=pipeline_out.final_response.task_achieved,
            )

            # Collect TTS timing (blocks until playback finishes) then print.
            timing = pipeline_out.timing
            if isinstance(output, TTSOutput):
                await output.wait_for_speech_done()
                timing = timing.model_copy(update={
                    "tts_synthesis": output.last_synth_duration,
                    "tts_playback":  output.last_play_duration,
                })

            _print_timing_table(timing)

            if debug:
                print("\n── Full pipeline output (JSON) ──")
                print(pipeline_out.model_dump_json(indent=2))

    finally:
        await output.stop()
        await input_.stop()


# =============================================================================
# Single-prompt run (terminal mode only)
# =============================================================================


async def run_single(
    manager: ModelManager,
    prompt: str,
    output: OutputGeneric,
    *,
    debug: bool = False,
) -> None:
    """
    Run the pipeline once for *prompt*, emit the response, then exit.

    Parameters
    ----------
    manager:
        A ``ModelManager`` with all three models already loaded.
    prompt:
        The user's request string.
    output:
        The output channel to send the response to.
    debug:
        If ``True``, also dump the full JSON pipeline output.
    """
    await output.start()

    try:
        loop = asyncio.get_event_loop()
        pipeline_out: PipelineOutput = await loop.run_in_executor(
            None, run_pipeline, manager, prompt
        )
        await output.emit(
            pipeline_out.final_response.message,
            task_achieved=pipeline_out.final_response.task_achieved,
        )

        # Collect TTS timing (blocks until playback finishes) then print.
        timing = pipeline_out.timing
        if isinstance(output, TTSOutput):
            await output.wait_for_speech_done()
            timing = timing.model_copy(update={
                    "tts_synthesis": output.last_synth_duration,
                    "tts_playback":  output.last_play_duration,
                })

        _print_timing_table(timing)

        if debug:
            print("\n── Full pipeline output (JSON) ──")
            print(pipeline_out.model_dump_json(indent=2))
    finally:
        await output.stop()


# =============================================================================
# Entry point
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai_agent",
        description="Memory Game AI agent — local multi-model LLM orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m ai_agent                           # terminal REPL\n"
            "  python -m ai_agent --prompt 'flip card 7'   # single run\n"
            "  python -m ai_agent --input voice --output tts  # hands-free\n"
        ),
    )
    parser.add_argument(
        "--input", "-i",
        choices=["terminal", "voice"],
        default="terminal",
        help="Input source: 'terminal' (stdin) or 'voice' (microphone). Default: terminal",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["terminal", "tts"],
        default="terminal",
        help="Output channel: 'terminal' (stdout) or 'tts' (Piper speaker). Default: terminal",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Run a single prompt instead of entering the interactive loop (terminal input only).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full JSON pipeline output after every run.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments, load models, build I/O components, run the loop."""
    parser = _build_parser()
    args = parser.parse_args()

    # Validate combination
    if args.prompt and args.input == "voice":
        parser.error("--prompt is only valid with --input terminal.")

    # ── Load models ──
    manager = ModelManager()
    try:
        manager.load_all()
    except Exception as exc:
        print(f"\n✗ Failed to load models: {exc}", file=sys.stderr)
        print("  Check that your GGUF paths in config.py are correct.")
        sys.exit(1)

    # ── Build input and output components ──

    # Build output first so that VoiceInput can receive its mute callbacks
    if args.output == "tts":
        # Mute callbacks will be wired after VoiceInput is created (if needed)
        tts_output: TTSOutput | None = TTSOutput()
        output: OutputGeneric = tts_output
    else:
        tts_output = None
        output = TerminalOutput()

    if args.input == "voice":
        voice_input = VoiceInput()
        input_: InputGeneric = voice_input

        # Wire TTS mute callbacks if both voice input and TTS output are active
        if tts_output is not None:
            tts_output._speaker._on_speaking_started = voice_input.on_speaking_started
            tts_output._speaker._on_speaking_stopped = voice_input.on_speaking_stopped
    else:
        voice_input = None
        input_ = TerminalInput()

    # ── Run ──
    try:
        if args.prompt:
            # Single-run mode (terminal input only, validated above)
            asyncio.run(
                run_single(manager, args.prompt, output, debug=args.debug)
            )
        else:
            # Print mode banner
            mode_str = f"[input={args.input}  output={args.output}]"
            print(f"\nMemory Game AI Agent — {mode_str}")
            if args.input == "terminal":
                print("Type 'quit' or 'exit' to stop.\n")

            asyncio.run(
                run_agent_loop(manager, input_, output, debug=args.debug)
            )
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
    finally:
        manager.unload_all()


if __name__ == "__main__":
    main()
