"""
Subprocess worker: benchmarks ONE model in complete process isolation.

Called by the notebook orchestrator (cell 10) via subprocess.  When this
process exits, the OS reclaims ALL CUDA VRAM — no leaks, no fragmentation,
no stale driver pools.  This is the only reliable workaround for the
llama-cpp-python VRAM leak on Windows/CUDA (GitHub issue #1442).

Usage:
    python _benchmark_worker.py <input.json> <output.json>

Input JSON schema:
    { model_name, model_path, chat_format, n_gpu_layers, n_ctx,
      system_prompt, tasks: [{task_id, prompt, expected_answer}],
      tool_definitions: [...], tool_prompts: {name: prompt} }

Output JSON schema:
    { task_results: [...], tool_results: [...], tool_score: int }
"""

import gc, json, sys, time
from datetime import date

import llama_cpp
from llama_cpp import Llama


# ── Context cleanup (identical to notebook cell 7) ───────────────────────────

def _clear_context(llm):
    """Nuke every piece of inference state between calls."""
    cleared = False
    try:
        llama_cpp.llama_kv_cache_clear(llm._ctx.ctx)
        cleared = True
    except Exception:
        pass
    if not cleared:
        try:
            llm._ctx.kv_cache_clear()
        except Exception:
            pass
    for attr in ("n_tokens", "_n_tokens"):
        if hasattr(llm, attr):
            try:
                setattr(llm, attr, 0)
            except Exception:
                pass
    try:
        llm.input_ids[:] = 0
    except Exception:
        pass
    try:
        llm.scores[:] = 0
    except Exception:
        pass
    if hasattr(llm, "reset"):
        try:
            llm.reset()
        except Exception:
            pass
    if getattr(llm, "_sampler", None) is not None:
        try:
            llm._sampler.reset()
        except Exception:
            pass


# ── Static tool handlers (mocked — identical to notebook cell 5) ─────────────

def _tool_get_current_date(a):
    return {"date": date.today().isoformat()}

def _tool_get_tasks(a):
    return {"tasks": ["read book", "buy groceries", "call dentist", "review PR #42"]}

def _tool_list_files(a):
    return {"files": ["report_q1.pdf", "budget_2025.xlsx", "meeting_notes.md",
                       "logo_final.png", "deployment_script.sh"]}

def _tool_get_weather(a):
    return {"city": a.get("city", "Unknown"), "condition": "Partly cloudy",
            "temperature_c": 18.5}

def _tool_calculator(a):
    expr = a.get("expression", "")
    try:
        result = float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        result = 0.0  # JSON-safe (no NaN)
    return {"expression": expr, "result": result}

def _tool_get_user_profile(a):
    return {"username": "jdoe", "full_name": "Jane Doe",
            "email": "jdoe@example.com", "role": "admin"}

def _tool_search_knowledge_base(a):
    return {"query": a.get("query", ""),
            "title": "Getting Started with the Internal Wiki",
            "summary": "Overview of how to search, edit, and create articles "
                       "in the company knowledge base."}

def _tool_create_reminder(a):
    return {"id": "rem-0042", "text": a.get("text", ""), "due": a.get("due", "")}

def _tool_get_stock_price(a):
    ticker = a.get("ticker", "").upper()
    prices = {"AAPL": 213.49, "TSLA": 248.10, "MSFT": 431.87, "GOOG": 178.02}
    return {"ticker": ticker, "price": prices.get(ticker, 99.99), "currency": "USD"}

def _tool_translate_text(a):
    return {"source_text": a.get("text", ""),
            "target_language": a.get("target_language", ""),
            "translated_text": f"[{a.get('target_language', 'Unknown')} "
                               f"translation of: {a.get('text', '')}]"}

_TOOL_HANDLERS = {
    "get_current_date":      _tool_get_current_date,
    "get_tasks":             _tool_get_tasks,
    "list_files":            _tool_list_files,
    "get_weather":           _tool_get_weather,
    "calculator":            _tool_calculator,
    "get_user_profile":      _tool_get_user_profile,
    "search_knowledge_base": _tool_search_knowledge_base,
    "create_reminder":       _tool_create_reminder,
    "get_stock_price":       _tool_get_stock_price,
    "translate_text":        _tool_translate_text,
}

_TOOL_REQUIRED_KEYS = {
    "get_current_date":      ["date"],
    "get_tasks":             ["tasks"],
    "list_files":            ["files"],
    "get_weather":           ["city", "condition", "temperature_c"],
    "calculator":            ["expression", "result"],
    "get_user_profile":      ["username", "full_name", "email", "role"],
    "search_knowledge_base": ["query", "title", "summary"],
    "create_reminder":       ["id", "text", "due"],
    "get_stock_price":       ["ticker", "price", "currency"],
    "translate_text":        ["source_text", "target_language", "translated_text"],
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        cfg = json.load(f)

    model_name  = cfg["model_name"]
    model_path  = cfg["model_path"]
    chat_format = cfg["chat_format"]       # may be None
    n_gpu       = cfg["n_gpu_layers"]
    n_ctx       = cfg["n_ctx"]
    tasks       = cfg["tasks"]
    tool_defs   = cfg["tool_definitions"]
    tool_prompts = cfg["tool_prompts"]
    sys_prompt  = cfg["system_prompt"]

    # ── Load model ───────────────────────────────────────────────────────
    print(f"  Loading {model_path} …", end=" ", flush=True)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu,
        n_ctx=n_ctx,
        chat_format=chat_format,
        verbose=False,
        last_n_tokens_size=0,
    )
    llm.set_cache(None)
    el = time.perf_counter() - t0
    tag = f" [{chat_format}]" if chat_format else ""
    print(f"done ({el:.1f}s){tag}")

    task_results = []
    tool_results = []

    # ── Factual tasks ────────────────────────────────────────────────────
    print(f"  ── Factual tasks ({len(tasks)}) "
          "──────────"
          "──────────"
          "──────────"
          "────")
    for t in tasks:
        tid = t["task_id"]
        prompt = t["prompt"]
        expected = t["expected_answer"]
        prefix = f"  [{tid:02d}] {prompt[:52]:<52}"
        print(prefix, end=" ", flush=True)

        start = time.perf_counter()
        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            elapsed = time.perf_counter() - start
            usage = resp.get("usage", {})
            otoks = usage.get("completion_tokens", 0) or 0
            itoks = usage.get("prompt_tokens", 0) or 0
            ans   = resp["choices"][0]["message"]["content"] or ""
            tps   = otoks / elapsed if elapsed > 0 else 0.0
            tr = dict(model=model_name, task_id=tid, prompt=prompt,
                      expected_answer=expected, model_answer=ans,
                      elapsed_seconds=elapsed, input_tokens=itoks,
                      output_tokens=otoks, tokens_per_second=tps,
                      response_length=len(ans), is_correct=None, error=None)
            print(f"{elapsed:5.2f}s  {tps:6.1f} tok/s  {len(ans):4d} chars")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            tr = dict(model=model_name, task_id=tid, prompt=prompt,
                      expected_answer=expected, model_answer="",
                      elapsed_seconds=elapsed, input_tokens=0, output_tokens=0,
                      tokens_per_second=0.0, response_length=0,
                      is_correct=None, error=str(exc))
            print(f"ERROR  {str(exc)[:60]}")

        _clear_context(llm)
        task_results.append(tr)

    # ── Tool-calling tests ───────────────────────────────────────────────
    print("  ── Tool-calling tests "
          "──────────"
          "──────────"
          "──────────"
          "──────────"
          "───────")
    tool_def_map = {td["function"]["name"]: td for td in tool_defs}
    tool_sys = ("You are a helpful assistant with access to tools. "
                "When the user asks something a tool can answer, "
                "call that tool immediately without narrating the plan.")

    for tool_name, prompt in tool_prompts.items():
        print(f"    [{tool_name}]", end=" ", flush=True)
        single_def = [tool_def_map[tool_name]]
        forced = {"type": "function", "function": {"name": tool_name}}
        t0 = time.perf_counter()
        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": tool_sys},
                    {"role": "user",   "content": prompt},
                ],
                tools=single_def,
                tool_choice=forced,
                max_tokens=512,
                temperature=0.0,
            )
            elapsed = time.perf_counter() - t0
            _clear_context(llm)
            msg   = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []

            if not calls:
                r = dict(tool_name=tool_name, called=False, valid=False,
                         raw="(no tool call emitted)", elapsed_seconds=elapsed)
            else:
                fn = calls[0]["function"]["name"]
                try:
                    args = json.loads(calls[0]["function"]["arguments"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                handler = _TOOL_HANDLERS.get(fn)
                if handler is None:
                    r = dict(tool_name=tool_name, called=True, valid=False,
                             raw=f"Unknown tool: {fn}", elapsed_seconds=elapsed)
                else:
                    res = handler(args)
                    required = _TOOL_REQUIRED_KEYS.get(fn, [])
                    valid = all(k in res for k in required)
                    r = dict(tool_name=tool_name, called=True, valid=valid,
                             raw=str(res)[:200], elapsed_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            r = dict(tool_name=tool_name, called=False, valid=False,
                     raw=f"ERROR: {exc}", elapsed_seconds=elapsed)

        _clear_context(llm)
        if r["valid"]:
            status = "✓ valid"
        elif r["called"]:
            status = "~ called, bad result"
        else:
            status = "✗ not called"
        print(f"{status}  ({r['elapsed_seconds']:.2f}s)  |  {r['raw'][:60]}")
        tool_results.append(r)

    score = sum(1 for r in tool_results if r["called"] and r["valid"])
    print(f"  Tool score: {score} / {len(tool_results)}")

    # ── Cleanup (process exit frees VRAM anyway, but be tidy) ────────────
    try:
        llm.close()
    except Exception:
        pass
    del llm
    gc.collect()

    # ── Write results ────────────────────────────────────────────────────
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump({"task_results": task_results,
                    "tool_results": tool_results,
                    "tool_score": score}, f)

    print("  Subprocess done — VRAM freed on exit.\n")


if __name__ == "__main__":
    main()
