"""
orchestrator.py — The Brain → Tool Caller → Communicator pipeline.

Each step:
  1. Builds a prompt from validated Pydantic input.
  2. Calls the appropriate Llama model (with grammar constraints for JSON output
     or function-calling tools).
  3. Parses and validates the output into a Pydantic model.
  4. Passes validated data to the next step.

Pipeline overview
-----------------
::

    User prompt
      │
      ▼
    run_brain()          — decides if a tool is needed; returns BrainDecision
      │
      ├─ needs_use_tool == False ──► straight to run_communicator()
      │
      ▼  (True)
    run_tool_caller()    — resolves the target tool, calls the backend
      │
      ▼
    run_communicator()   — generates the final user-facing reply (Nova)
      │
      ▼
    PipelineOutput

Notes on small-model robustness
---------------------------------
Three empirically derived fixes make sub-2B parameter tool-caller models
reliable:

FIX 1 — Send only ONE tool per request.
    Passing all four tool definitions overwhelms small models; they see too
    many choices and fall back to plain text instead of a tool_call block.

FIX 2 — Force tool_choice to the specific function name.
    With ``"auto"`` the model may answer in plain text.  Pinning
    ``tool_choice`` to the exact function name forces the chatml-function-calling
    handler to emit a tool_call block unconditionally.

FIX 3 — max_tokens=512, temperature=0.0.
    The chatml prompt wrapper adds ~50–80 tokens of overhead.  Combined with
    the JSON arguments the model must emit, 256 tokens was too tight, causing
    truncated or missing tool_call blocks.  ``temperature=0.0`` makes the call
    deterministic.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import llama_cpp
from llama_cpp import Llama
from pydantic import ValidationError

from ai_agent.config import (
    BRAIN_SYSTEM_PROMPT,
    COMMUNICATOR_SYSTEM_PROMPT,
    TOOL_CALLER_SYSTEM_PROMPT,
)
from ai_agent.mcp.memory_game_mcp import TOOL_SCHEMA_MAP, TOOL_SCHEMAS, execute_tool
from ai_agent.model_manager import ModelManager
from ai_agent.schemas import (
    BrainDecision,
    BrainInput,
    CommunicatorInput,
    CommunicatorResponse,
    PipelineInput,
    PipelineOutput,
    PipelineTiming,
    ToolCallerInput,
    ToolCallerOutput,
    ToolCallRecord,
)

# =============================================================================
# HELPERS
# =============================================================================


def _extract_json(raw: str) -> dict[str, Any]:
    """
    Extract the first JSON object from a possibly-noisy LLM response.

    Handles common annoyances: markdown code fences, leading/trailing prose,
    and stray control characters that small models occasionally emit.

    Parameters
    ----------
    raw:
        Raw string returned by the LLM.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If no valid JSON object can be found in *raw*.
    """
    text = raw.strip()

    # Strip markdown fences (``` or ```json … ```)
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Fast path: the whole string is valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Slow path: find the first { … } pair and try that substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{raw[:300]}")


def _chat(
    llm: Llama,
    system_prompt: str,
    user_message: str,
    *,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """
    Send a chat completion request to a Llama model.

    Parameters
    ----------
    llm:
        The loaded Llama instance to query.
    system_prompt:
        The system-role message prepended to every request.
    user_message:
        The user-role message for this specific request.
    response_format:
        Optional ``{"type": "json_object", "schema": ...}`` dict to enable
        grammar-constrained JSON output.
    tools:
        Optional list of OpenAI-format tool definitions for function-calling.
    tool_choice:
        Optional tool selection strategy (``"auto"``, ``"none"``, or a
        ``{"type": "function", "function": {"name": ...}}`` dict to force a
        specific tool call).
    max_tokens:
        Maximum number of tokens in the completion.
    temperature:
        Sampling temperature.  Use ``0.0`` for deterministic / greedy output.

    Returns
    -------
    dict
        The full response dict from ``create_chat_completion``.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    return llm.create_chat_completion(**kwargs)


# =============================================================================
# STEP 1: BRAIN
# =============================================================================


def run_brain(
    llm: Llama, inp: BrainInput
) -> tuple[BrainDecision, float | None, float, float | None]:
    """
    Ask the Brain model to analyse the user prompt and decide on routing.

    The response is grammar-constrained to the BrainDecision JSON schema so
    that Pydantic validation is virtually guaranteed to pass.

    Parameters
    ----------
    llm:
        The loaded Brain Llama instance.
    inp:
        Validated input containing the raw user prompt.

    Returns
    -------
    tuple[BrainDecision, float | None, float, float | None]
        Routing decision, generation tokens-per-second, prompt evaluation
        time in seconds, and prompt evaluation tokens-per-second.
    """
    print("\n── Step 1: Brain ──")
    print(f"   User prompt: {inp.user_prompt!r}")

    response_format = {
        "type": "json_object",
        "schema": BrainDecision.model_json_schema(),
    }

    llama_cpp.llama_perf_context_reset(llm._ctx.ctx)

    t0 = time.perf_counter()
    resp = _chat(
        llm,
        BRAIN_SYSTEM_PROMPT,
        inp.user_prompt,
        response_format=response_format,
    )
    elapsed = time.perf_counter() - t0

    perf = llama_cpp.llama_perf_context(llm._ctx.ctx)
    prompt_eval_s = perf.t_p_eval_ms / 1000.0
    generation_s = perf.t_eval_ms / 1000.0

    raw_text = resp["choices"][0]["message"]["content"]
    print(f"   Raw response: {raw_text[:200]}")

    parsed = _extract_json(raw_text)
    decision = BrainDecision.model_validate(parsed)

    print(f"   Decision: needs_tool={decision.needs_use_tool}")
    if decision.needs_use_tool:
        print(f"   Tool prompt: {decision.specialized_tool_prompt!r}")

    usage = resp.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    tps = (
        (completion_tokens / generation_s)
        if (completion_tokens and generation_s > 0)
        else None
    )
    pp_tps = (
        (prompt_tokens / prompt_eval_s)
        if (prompt_tokens and prompt_eval_s > 0)
        else None
    )

    return decision, tps, prompt_eval_s, pp_tps


# =============================================================================
# STEP 2: TOOL CALLER
# =============================================================================


def _resolve_tool_name(brain_decision: BrainDecision) -> str:
    """
    Determine which single tool to invoke.

    Primary:  use the Brain's grammar-constrained ``tool_name`` field.
    Fallback: keyword-match on ``specialized_tool_prompt`` as a defence-in-depth
              measure in case an older Brain prompt omits ``tool_name``.

    Parameters
    ----------
    brain_decision:
        The validated BrainDecision produced by ``run_brain``.

    Returns
    -------
    str
        One of: ``"get_game_state"``, ``"flip_card"``, ``"reset_game"``,
        ``"play_again"``.
    """
    if brain_decision.tool_name:
        return brain_decision.tool_name

    # Fallback keyword router
    prompt_lower = brain_decision.specialized_tool_prompt.lower()
    if any(kw in prompt_lower for kw in ("reset", "restart", "shuffle")):
        return "reset_game"
    if any(kw in prompt_lower for kw in ("play again", "new game", "fresh")):
        return "play_again"
    # Check "show/state/board" BEFORE "flip/card" — "show me the cards" means
    # get_game_state, not flip_card.
    if any(
        kw in prompt_lower for kw in ("state", "show", "board", "status", "all cards")
    ):
        return "get_game_state"
    if any(kw in prompt_lower for kw in ("flip", "turn over", "reveal")):
        return "flip_card"
    # Default: get_game_state is the safest read-only fallback
    return "get_game_state"


def run_tool_caller(
    llm: Llama,
    inp: ToolCallerInput,
    brain_decision: BrainDecision,
) -> tuple[ToolCallerOutput, float | None, float, float | None]:
    """
    Ask the Tool Caller model to invoke a single MCP tool and execute it.

    The Brain already identified which tool is needed.  We:
    1. Send only that one tool definition            (FIX 1)
    2. Force ``tool_choice`` to that function name   (FIX 2)
    3. Use ``max_tokens=512``, ``temperature=0.0``   (FIX 3)

    If the model still fails to produce a formal ``tool_calls`` block (rare
    with a forced tool_choice), a content-based extraction fallback is attempted
    before giving up.

    Parameters
    ----------
    llm:
        The loaded Tool Caller Llama instance.
    inp:
        Validated input containing the Brain's instruction string.
    brain_decision:
        The Brain's routing decision, used to resolve the target tool name.

    Returns
    -------
    tuple[ToolCallerOutput, float | None, float, float | None]
        Aggregated tool call records, generation tokens-per-second, prompt
        evaluation time in seconds, and prompt evaluation tokens-per-second.
    """
    print("\n── Step 2: Tool Caller ──")
    print(f"   Instruction: {inp.brain_instruction!r}")

    # Resolve which single tool to call
    target_tool = _resolve_tool_name(brain_decision)
    print(f"   Resolved tool: {target_tool}")

    # FIX 1: expose only the one target tool
    single_tool_def = [TOOL_SCHEMA_MAP[target_tool]]

    # FIX 2: pin tool_choice to force a tool_call block
    forced_tool_choice = {
        "type": "function",
        "function": {"name": target_tool},
    }

    llama_cpp.llama_perf_context_reset(llm._ctx.ctx)

    t0 = time.perf_counter()
    resp = _chat(
        llm,
        TOOL_CALLER_SYSTEM_PROMPT,
        inp.brain_instruction,
        tools=single_tool_def,  # FIX 1: one tool only
        tool_choice=forced_tool_choice,  # FIX 2: force the call
        max_tokens=512,  # FIX 3: more headroom
        temperature=0.0,  # FIX 3: deterministic
    )
    elapsed = time.perf_counter() - t0

    perf = llama_cpp.llama_perf_context(llm._ctx.ctx)
    prompt_eval_s = perf.t_p_eval_ms / 1000.0
    generation_s = perf.t_eval_ms / 1000.0

    message = resp["choices"][0]["message"]
    tool_calls_raw = message.get("tool_calls") or []

    # ------------------------------------------------------------------
    # Content-based fallback: if the model still didn't produce formal
    # tool_calls (rare with forced tool_choice, but possible with some
    # chat formats), attempt to extract a JSON call from the content field.
    # ------------------------------------------------------------------
    if not tool_calls_raw and message.get("content"):
        print("   (No formal tool_calls; attempting content-based extraction)")
        try:
            content_json = _extract_json(message["content"])
            if "name" in content_json:
                tool_calls_raw = [
                    {
                        "function": {
                            "name": content_json["name"],
                            "arguments": json.dumps(content_json.get("arguments", {})),
                        }
                    }
                ]
            elif target_tool == "flip_card":
                # Last-ditch: if model just output {"card_id": 5}, wrap it
                if "card_id" in content_json or "cardId" in content_json:
                    tool_calls_raw = [
                        {
                            "function": {
                                "name": "flip_card",
                                "arguments": json.dumps(content_json),
                            }
                        }
                    ]
        except (ValueError, KeyError):
            pass

    usage = resp.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    tps = (
        (completion_tokens / generation_s)
        if (completion_tokens and generation_s > 0)
        else None
    )
    pp_tps = (
        (prompt_tokens / prompt_eval_s)
        if (prompt_tokens and prompt_eval_s > 0)
        else None
    )

    if not tool_calls_raw:
        print("   ⚠ Model produced no tool calls even with forced choice.")
        print(f"   Raw content: {message.get('content', '')[:200]}")
        return ToolCallerOutput(calls=[]), tps, prompt_eval_s, pp_tps

    # Execute each tool call against the backend
    records: list[ToolCallRecord] = []
    for tc in tool_calls_raw:
        fn = tc.get("function", tc)
        name = fn.get("name", "unknown")
        args_raw = fn.get("arguments", "{}")

        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}
        else:
            args = args_raw

        print(f"   Calling tool: {name}({args})")
        record = execute_tool(name, args)
        records.append(record)
        status = "✓" if record.success else "✗"
        print(f"   {status} Result: {json.dumps(record.result)[:150]}")

    return ToolCallerOutput(calls=records), tps, prompt_eval_s, pp_tps


# =============================================================================
# STEP 3: COMMUNICATOR
# =============================================================================


def run_communicator(
    llm: Llama, inp: CommunicatorInput
) -> tuple[CommunicatorResponse, float | None, float, float | None]:
    """
    Ask the Communicator model (Nova) to summarise results for the user.

    Nova receives the original prompt plus full context of what happened and
    must produce a CommunicatorResponse JSON with a conversational message and
    a ``task_achieved`` flag.

    Parameters
    ----------
    llm:
        The loaded Communicator Llama instance.
    inp:
        Validated input containing the original prompt, Brain decision, and
        optional tool results.

    Returns
    -------
    tuple[CommunicatorResponse, float | None, float, float | None]
        The final message and task-achieved flag, generation tokens-per-second,
        prompt evaluation time in seconds, and prompt evaluation
        tokens-per-second.
    """
    print("\n── Step 3: Communicator (Nova) ──")

    # Build a context message with everything Nova needs to know
    context_parts = [
        f"Original user request: {inp.original_prompt!r}",
        f"Brain decided tool was needed: {inp.brain_analysis.needs_use_tool}",
    ]

    if inp.tool_result and inp.tool_result.calls:
        context_parts.append("Tool call results:")
        for call in inp.tool_result.calls:
            status = "succeeded" if call.success else f"failed ({call.error})"
            result_preview = json.dumps(call.result)[:300] if call.result else "null"
            context_parts.append(
                f"  - {call.tool_name}({call.tool_args}) → {status}: {result_preview}"
            )
    else:
        context_parts.append(
            "No tools were called (either not needed or the user's request "
            "was conversational)."
        )

    user_message = "\n".join(context_parts)
    print(f"   Context length: {len(user_message)} chars")

    response_format = {
        "type": "json_object",
        "schema": CommunicatorResponse.model_json_schema(),
    }

    llama_cpp.llama_perf_context_reset(llm._ctx.ctx)

    t0 = time.perf_counter()
    resp = _chat(
        llm,
        COMMUNICATOR_SYSTEM_PROMPT,
        user_message,
        response_format=response_format,
    )
    elapsed = time.perf_counter() - t0

    perf = llama_cpp.llama_perf_context(llm._ctx.ctx)
    prompt_eval_s = perf.t_p_eval_ms / 1000.0
    generation_s = perf.t_eval_ms / 1000.0

    raw_text = resp["choices"][0]["message"]["content"]
    print(f"   Raw response: {raw_text[:200]}")

    parsed = _extract_json(raw_text)
    result = CommunicatorResponse.model_validate(parsed)

    print(f"   Task achieved: {result.task_achieved}")

    usage = resp.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    tps = (
        (completion_tokens / generation_s)
        if (completion_tokens and generation_s > 0)
        else None
    )
    pp_tps = (
        (prompt_tokens / prompt_eval_s)
        if (prompt_tokens and prompt_eval_s > 0)
        else None
    )

    return result, tps, prompt_eval_s, pp_tps


# =============================================================================
# FULL PIPELINE
# =============================================================================


def run_pipeline(manager: ModelManager, user_prompt: str) -> PipelineOutput:
    """
    Run the complete Brain → Tool Caller → Communicator pipeline.

    This is the main entry point for the orchestration logic.  It is a
    synchronous function; callers in an async context (e.g. the voice-input
    loop) should run it via ``asyncio.get_event_loop().run_in_executor``.

    Parameters
    ----------
    manager:
        A ``ModelManager`` with all three models already loaded.
    user_prompt:
        The user's natural-language request (typed or voice-transcribed).

    Returns
    -------
    PipelineOutput
        Fully validated output including the Brain decision, any tool results,
        and the final Communicator response.
    """
    # ── Step 0: Prompt processing ──
    t0 = time.perf_counter()
    pipeline_input = PipelineInput(user_prompt=user_prompt)
    brain_input = BrainInput(user_prompt=pipeline_input.user_prompt)
    t_prompt_done = time.perf_counter()

    # ── Step 1: Brain ──
    t_brain_start = time.perf_counter()
    brain_decision, brain_tps, brain_pp_s, brain_pp_tps = run_brain(
        manager.brain, brain_input
    )
    t_brain_done = time.perf_counter()

    # ── Step 2: Tool Caller (conditional) ──
    tool_result: ToolCallerOutput | None = None
    tool_tps: float | None = None
    tool_pp_s: float = 0.0
    tool_pp_tps: float | None = None
    t_tool_start: float | None = None
    t_tool_done: float | None = None
    if brain_decision.needs_use_tool:
        tc_input = ToolCallerInput(
            brain_instruction=brain_decision.specialized_tool_prompt
            or pipeline_input.user_prompt
        )
        t_tool_start = time.perf_counter()
        tool_result, tool_tps, tool_pp_s, tool_pp_tps = run_tool_caller(
            manager.tool_caller, tc_input, brain_decision
        )
        t_tool_done = time.perf_counter()
    else:
        print("\n── Step 2: Tool Caller ── SKIPPED (no tool needed)")

    # ── Step 3: Communicator ──
    comm_input = CommunicatorInput(
        original_prompt=pipeline_input.user_prompt,
        brain_analysis=brain_decision,
        tool_result=tool_result,
    )
    t_comm_start = time.perf_counter()
    final_response, communicator_tps, comm_pp_s, comm_pp_tps = run_communicator(
        manager.communicator, comm_input
    )
    t_comm_done = time.perf_counter()

    # Aggregate prompt processing (prefill) time across all LLM steps
    total_prompt_eval_s = brain_pp_s + tool_pp_s + comm_pp_s

    # Compute a weighted-average prompt processing tok/s
    all_pp_tps = [t for t in (brain_pp_tps, tool_pp_tps, comm_pp_tps) if t is not None]
    avg_pp_tps = (sum(all_pp_tps) / len(all_pp_tps)) if all_pp_tps else None

    timing = PipelineTiming(
        prompt_processing=total_prompt_eval_s,
        prompt_processing_tps=avg_pp_tps,
        brain=(t_brain_done - t_brain_start) - brain_pp_s,
        brain_tps=brain_tps,
        tool_calling=(
            (t_tool_done - t_tool_start) - tool_pp_s
            if t_tool_start is not None
            else None
        ),
        tool_calling_tps=tool_tps,
        communicator=(t_comm_done - t_comm_start) - comm_pp_s,
        communicator_tps=communicator_tps,
    )

    return PipelineOutput(
        brain_decision=brain_decision,
        tool_result=tool_result,
        final_response=final_response,
        timing=timing,
    )
