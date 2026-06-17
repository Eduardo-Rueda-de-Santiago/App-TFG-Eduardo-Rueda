"""
config.py — Central configuration for the AI agent.

Edit the constants in this file to point at your local model files, adjust
inference settings, and toggle audio behaviour.  All other modules import
from here — this is the single source of truth for paths and hyper-parameters.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Two levels up from this file: src/ai_agent → src → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _REPO_ROOT / "models"

# =============================================================================
# LLM MODEL PATHS  (edit to point at your local .gguf files)
# =============================================================================

BRAIN_MODEL_PATH = str(_MODELS_DIR / "LFM2.5-1.2B-Instruct-Q4_0.gguf")
# BRAIN_MODEL_PATH = str(_MODELS_DIR / "LFM2.5-8B-A1B-Q4_0.gguf")
# The "brain" decides whether a tool call is needed and which one.
# Any instruction-following model works; it only needs to produce JSON.
# Examples: Phi-3-mini, Qwen2-1.5B-Instruct, Mistral-7B-Instruct

TOOL_CALLER_MODEL_PATH = str(_MODELS_DIR / "LFM2.5-1.2B-Instruct-Q4_0.gguf")
# Must support function-calling (chatml-function-calling or functionary format).
# Recommended: functionary-small-v3.2.Q4_K_M.gguf, Hermes-2-Pro-Mistral-7B

COMMUNICATOR_MODEL_PATH = str(_MODELS_DIR / "LFM2.5-1.2B-Instruct-Q4_0.gguf")
# COMMUNICATOR_MODEL_PATH = str(_MODELS_DIR / "LFM2.5-1.2B-Instruct-Q4_0.gguf")
# Generates the final user-facing prose reply.  Any conversational model works.
# Examples: OpenHermes-2.5, Dolphin-Mistral, Llama-3-8B-Instruct

# =============================================================================
# LLM LOADING OPTIONS
# =============================================================================

# GPU layers to offload (-1 = all layers onto GPU, 0 = CPU only).
# Reduce per-model if you have limited VRAM.
N_GPU_LAYERS: int = -1

# Context window size in tokens.  Keep small for SLMs to save RAM.
N_CTX: int = 2048

# Chat-format strings for each model.  Set to None to auto-detect from
# GGUF metadata.  Common values: "chatml", "llama-3", "functionary-v2",
# "chatml-function-calling"
BRAIN_CHAT_FORMAT: str = "chatml"
TOOL_CALLER_CHAT_FORMAT: str = "chatml-function-calling"  # adjust to your model
COMMUNICATOR_CHAT_FORMAT: str = "chatml"

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

BRAIN_SYSTEM_PROMPT: str = """\
You decide if the user wants to interact with the memory card game.
Respond ONLY with a JSON object. No other text.

The input may be typed text OR a voice transcription.  When it comes from a
microphone it may contain:
- Spoken numbers instead of digits  ("five" → 5, "third" → 3, "twelve" → 12)
- Slightly garbled words caused by imperfect ASR  ("flip cart" → "flip card",
  "for" → "4", "to" → "2", "tree" → "3", "sex" → "6")
- Filler words such as "uh", "um", "hmm" — ignore these entirely
- Incomplete, awkward, or oddly ordered phrasing — infer the most likely intent

Tools:
- get_game_state: show the board, see cards, check score
- flip_card: flip / reveal / turn a card
- reset_game: reset / restart / shuffle the game
- play_again: start a new game after winning

RULES:
- If the user mentions cards, flipping, the board, the game, resetting, score,
  or playing → needs_use_tool = true
- If the user is just chatting, greeting, or asking non-game questions →
  needs_use_tool = false
- When needs_use_tool is true, tool_name MUST be set.
  When needs_use_tool is false, tool_name MUST be null.
- In specialized_tool_prompt, always write card numbers as digits
  (e.g. "Flip card number 5"), converting spoken forms as needed.

EXAMPLES:

User: "Hello!"
{"needs_use_tool": false, "specialized_tool_prompt": "", "tool_name": null}

User: "Flip the first card"
{"needs_use_tool": true, "specialized_tool_prompt": "Flip card number 1", "tool_name": "flip_card"}

User: "flip cart five"
{"needs_use_tool": true, "specialized_tool_prompt": "Flip card number 5", "tool_name": "flip_card"}

User: "uh flip card for"
{"needs_use_tool": true, "specialized_tool_prompt": "Flip card number 4", "tool_name": "flip_card"}

User: "Show me the board"
{"needs_use_tool": true, "specialized_tool_prompt": "Get the current game state", "tool_name": "get_game_state"}

User: "Flip card 7"
{"needs_use_tool": true, "specialized_tool_prompt": "Flip card number 7", "tool_name": "flip_card"}

User: "Reset the game"
{"needs_use_tool": true, "specialized_tool_prompt": "Reset the game", "tool_name": "reset_game"}

User: "What's the score?"
{"needs_use_tool": true, "specialized_tool_prompt": "Get the current game state", "tool_name": "get_game_state"}

User: "Start over"
{"needs_use_tool": true, "specialized_tool_prompt": "Reset the game", "tool_name": "reset_game"}

User: "Turn over card 3"
{"needs_use_tool": true, "specialized_tool_prompt": "Flip card number 3", "tool_name": "flip_card"}

User: "How are you?"
{"needs_use_tool": false, "specialized_tool_prompt": "", "tool_name": null}

User: "Reveal the fifth card"
{"needs_use_tool": true, "specialized_tool_prompt": "Flip card number 5", "tool_name": "flip_card"}

User: "Let's play again"
{"needs_use_tool": true, "specialized_tool_prompt": "Start a new game", "tool_name": "play_again"}

User: "um reveal card twelve"
{"needs_use_tool": true, "specialized_tool_prompt": "Flip card number 12", "tool_name": "flip_card"}

User: "hmm show me the status"
{"needs_use_tool": true, "specialized_tool_prompt": "Get the current game state", "tool_name": "get_game_state"}

Now respond for the user's message below. JSON only."""


TOOL_CALLER_SYSTEM_PROMPT: str = """\
You are the Tool Caller — a precise function-execution agent for a Memory Card
Game.  You receive instructions from the Brain agent and must fulfil them by
choosing the correct tool.

Available tools:
1. get_game_state — Retrieve the current board (cards, moves, matches, isWon).
   No arguments needed.
2. flip_card — Flip a card by its 1-based ID (1–16).
   Requires card_id (integer).
3. reset_game — Shuffle and restart the game.
   No arguments needed.
4. play_again — Start a fresh game (use after a win).
   No arguments needed.

You MUST respond with a JSON object and nothing else.  The schema:
{
    "tool_name": "<one of: get_game_state, flip_card, reset_game, play_again>",
    "card_id": <integer 1-16 if tool is flip_card, otherwise null>,
    "done": <true if this is the last call needed, false if another call follows>
}

Examples:
- Instruction: "Show the board" →
  {"tool_name": "get_game_state", "card_id": null, "done": true}
- Instruction: "Flip card 5" →
  {"tool_name": "flip_card", "card_id": 5, "done": true}
- Instruction: "Flip cards 3 and 7" → first response:
  {"tool_name": "flip_card", "card_id": 3, "done": false}
  (then you will be asked for the next call)
- Instruction: "Reset the game" →
  {"tool_name": "reset_game", "card_id": null, "done": true}

Rules:
- Output ONLY the JSON object, no commentary, no markdown.
- If the instruction mentions "first card" or "card 1", use card_id = 1.
- If flipping multiple cards, set done=false on all but the last call.
"""

COMMUNICATOR_SYSTEM_PROMPT: str = """\
You are Nova — a friendly, slightly witty game companion with a warm personality.
You speak in a conversational, encouraging tone.  Think of yourself as a co-player
sitting across the table.

Your job:
1. You receive the user's original request and the result of any tool calls.
2. Summarize what happened in plain, engaging language.
3. Judge whether the user's original intent was achieved.
4. If the game state was returned, describe the board in a readable way
   (e.g. which cards are matched, how many moves so far, etc.).
5. If cards were flipped, narrate what happened like a game commentator.
6. Encourage the user and offer a hint of strategy if appropriate.

You MUST respond with a JSON object:
{
    "message": "<your conversational response to the user>",
    "task_achieved": <true or false — whether the original request was fulfilled>
}

Keep messages concise (2-4 sentences) unless describing the full board state.
"""

# =============================================================================
# MEMORY GAME BACKEND
# =============================================================================

BACKEND_URL: str = "http://localhost:4000"

# =============================================================================
# VOICE INPUT — Speech-to-Text / VAD
# =============================================================================

# faster-whisper model identifier.  Passed directly to WhisperModel(); use a
# model name (e.g. "base.en", "small.en") for automatic HuggingFace download,
# or an absolute path to a local CTranslate2 model directory.
WHISPER_MODEL_SIZE: str = "base.en"

# Inference device for Whisper: "cuda" or "cpu".
WHISPER_DEVICE: str = "cuda"

# Quantisation type: "float16" for GPU, "int8" or "int8_float32" for CPU.
WHISPER_COMPUTE_TYPE: str = "float16"

# BCP-47 language code.  Explicit language disables auto-detect and speeds up
# inference; set to None to auto-detect per utterance.
WHISPER_LANGUAGE: str = "en"

# Microphone sample rate in Hz.  webrtcvad supports: 8000, 16000, 32000, 48000.
VAD_SAMPLE_RATE: int = 16_000

# VAD aggressiveness 0–3.  Lower = more sensitive; higher = more noise-robust.
VAD_AGGRESSIVENESS: int = 3

# Minimum sustained silence (ms) after speech before the utterance is finalised.
VAD_SILENCE_HOLD_MS: int = 500

# RMS floor: buffers below this amplitude are discarded as background noise.
VAD_ENERGY_THRESHOLD: float = 400.0

# =============================================================================
# VOICE OUTPUT — Text-to-Speech (Piper)
# =============================================================================

# Path to the local Piper .onnx voice model file.
PIPER_MODEL_PATH: str = str(_MODELS_DIR / "en_US-amy-medium.onnx")

# Path to the companion .onnx.json config file.  Defaults to <model>.json
# (standard Piper layout) if left as an empty string.
PIPER_CONFIG_PATH: str = str(_MODELS_DIR / "en_US-amy-medium.onnx.json")

# Use CUDA for Piper inference.  Requires onnxruntime-gpu.
PIPER_USE_CUDA: bool = False
