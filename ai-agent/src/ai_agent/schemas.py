"""
schemas.py — Pydantic models that validate every input and output in the pipeline.

Each model in the chain has a strongly-typed input/output contract.
llama-cpp-python's grammar-constrained generation ensures LLM output conforms
to these schemas at the token level, making Pydantic validation virtually
guaranteed to pass.

Pipeline flow:
  PipelineInput
    └─► BrainInput → BrainDecision
          └─► ToolCallerInput → ToolCallRecord(s) → ToolCallerOutput
                └─► CommunicatorInput → CommunicatorResponse
                      └─► PipelineOutput
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, computed_field

# The four valid MCP tool names.  Used as a Literal constraint in BrainDecision
# so that grammar-constrained generation can only emit one of these exact strings
# — the model literally cannot hallucinate a tool name.
VALID_TOOL_NAMES = Literal[
    "get_game_state", "flip_card", "reset_game", "play_again"
]


# =============================================================================
# MODEL 1: BRAIN
# =============================================================================


class BrainInput(BaseModel):
    """What the Brain model receives."""

    user_prompt: str = Field(
        ..., description="The raw user request to analyse."
    )


class BrainDecision(BaseModel):
    """
    What the Brain model must produce (grammar-constrained to this schema).

    The Brain reads the user prompt and decides:
    1. Whether a tool call is required.
    2. If so, which tool and what instruction to pass to the Tool Caller.
    """

    specialized_tool_prompt: str = Field(
        default="",
        description=(
            "A precise instruction for the tool-calling agent.  "
            "Empty string when no tool is needed."
        ),
    )
    needs_use_tool: bool = Field(
        ...,
        description="Whether an MCP tool call is required to fulfil the request.",
    )
    tool_name: Optional[VALID_TOOL_NAMES] = Field(
        default=None,
        description=(
            "Which specific tool the tool-caller should invoke.  "
            "Must be one of: get_game_state, flip_card, reset_game, play_again.  "
            "null when needs_use_tool is false."
        ),
    )


# =============================================================================
# MODEL 2: TOOL CALLER
# =============================================================================


class ToolCallerInput(BaseModel):
    """What the Tool Caller model receives."""

    brain_instruction: str = Field(
        ..., description="The specialized prompt crafted by the Brain."
    )


class ToolCallRequest(BaseModel):
    """
    What the Tool Caller model must produce per call (grammar-constrained).

    Uses Literal so the JSON schema grammar restricts tool_name to exactly
    the four valid values at the token level — the model literally cannot
    hallucinate a tool name.
    """

    tool_name: Literal[
        "get_game_state", "flip_card", "reset_game", "play_again"
    ] = Field(..., description="Which tool to call.")
    card_id: Optional[int] = Field(
        default=None,
        description=(
            "The 1-based card ID (1–16).  Required for flip_card, "
            "omit or null for all other tools."
        ),
    )
    done: bool = Field(
        ...,
        description=(
            "Set to true if this is the last tool call needed to fulfil "
            "the instruction.  Set to false if you need to call another "
            "tool after this one (e.g. flipping two cards in sequence)."
        ),
    )


class ToolCallRecord(BaseModel):
    """A single tool invocation and its result."""

    tool_name: str = Field(..., description="Name of the tool that was called.")
    tool_args: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool."
    )
    result: Any = Field(
        default=None, description="The raw result returned by the tool."
    )
    success: bool = Field(
        default=True, description="Whether the tool call succeeded."
    )
    error: Optional[str] = Field(
        default=None, description="Error message if the call failed."
    )


class ToolCallerOutput(BaseModel):
    """Aggregated result from the Tool Caller stage (may include multiple calls)."""

    calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="Ordered list of tool calls made and their results.",
    )


# =============================================================================
# MODEL 3: COMMUNICATOR
# =============================================================================


class CommunicatorInput(BaseModel):
    """What the Communicator model receives."""

    original_prompt: str = Field(
        ..., description="The user's original request (for context)."
    )
    brain_analysis: BrainDecision = Field(
        ..., description="The Brain's routing decision."
    )
    tool_result: Optional[ToolCallerOutput] = Field(
        default=None,
        description="Tool call results, if any tools were invoked.",
    )


class CommunicatorResponse(BaseModel):
    """What the Communicator model must produce (grammar-constrained)."""

    message: str = Field(
        ...,
        description="A conversational response to present to the user.",
    )
    task_achieved: bool = Field(
        ...,
        description="Whether the user's original intent was successfully fulfilled.",
    )


# =============================================================================
# PIPELINE TIMING
# =============================================================================


class PipelineTiming(BaseModel):
    """Timing breakdown for each pipeline step (in seconds)."""

    prompt_processing: float = Field(0.0, description="Time to validate and set up the pipeline input.")
    brain: float = Field(0.0, description="Time spent in the Brain LLM step.")
    tool_calling: Optional[float] = Field(None, description="Time spent in Tool Caller step (None if skipped).")
    communicator: float = Field(0.0, description="Time spent in the Communicator LLM step.")
    tts_synthesis: Optional[float] = Field(None, description="Time for TTS ONNX inference (text → PCM).")
    tts_playback: Optional[float] = Field(None, description="Time for TTS audio playback to the output device.")

    @computed_field  # type: ignore[misc]
    @property
    def pipeline_total(self) -> float:
        """Total LLM/tool time, excluding TTS."""
        total = self.prompt_processing + self.brain + self.communicator
        if self.tool_calling is not None:
            total += self.tool_calling
        return total

    @computed_field  # type: ignore[misc]
    @property
    def total(self) -> float:
        """Grand total including TTS synthesis and playback."""
        t = self.pipeline_total
        if self.tts_synthesis is not None:
            t += self.tts_synthesis
        if self.tts_playback is not None:
            t += self.tts_playback
        return t


# =============================================================================
# TOP-LEVEL PIPELINE I/O
# =============================================================================


class PipelineInput(BaseModel):
    """Top-level input to the orchestrator."""

    user_prompt: str = Field(..., description="What the user wants.")


class PipelineOutput(BaseModel):
    """Top-level output from the orchestrator."""

    brain_decision: BrainDecision
    tool_result: Optional[ToolCallerOutput] = None
    final_response: CommunicatorResponse
    timing: PipelineTiming = Field(default_factory=PipelineTiming)
