"""Unit test for the unified chat/tool renderer (fractal.chat_format) — Phase 0 deliverable.

Feeds one representative record per data source (as data_mix._render sees them) through the
renderer and checks the invariants of the ONE protocol:
  - the right markers appear, in order;
  - a glaive `<functioncall>` becomes a <|tool_call|> segment whose payload is VALID JSON;
  - a glaive `FUNCTION RESPONSE:` becomes <|tool_result|>;
  - every non-empty conversation terminates with exactly one <|end|>;
  - malformed / empty records render to '' (skipped, not a bare <|end|>);
  - the divergent 'User:/Bot:' framing never leaks into rendered output.

No GPU, no network, no tokenizer — pure string rendering (CPU). Run:
    uv run python -m fractal.tests.test_chat_format
"""

from __future__ import annotations

import json

from fractal import chat_format as cf
from fractal.data_mix import _render

# --- representative records, matching each source's real schema -------------------------------

REC_MESSAGES = {"messages": [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Hi there"},
    {"role": "assistant", "content": "Hello!"},
]}

REC_CONVERSATIONS = {"conversations": [
    {"from": "human", "value": "2+2?"},
    {"from": "gpt", "value": "4"},
]}

REC_DOLLY = {"instruction": "Name a primary color.", "context": "", "response": "Blue."}

REC_HH = {"chosen": "\n\nHuman: What is 3*7?\n\nAssistant: 21."}

# glaive: system carries the tool defs, chat is a transcript; the tool call is a single-quoted
# embedded-JSON `arguments` (not valid JSON as-is) that the renderer must repair.
REC_GLAIVE = {
    "system": ("SYSTEM: You are a helpful assistant with access to the following functions. "
               "Use them if required -\n"
               '{"name": "get_stock_price", "description": "Get the current stock price", '
               '"parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, '
               '"required": ["symbol"]}}'),
    "chat": ("USER: What is Apple trading at?\n\n\n"
             "ASSISTANT: <functioncall> {\"name\": \"get_stock_price\", \"arguments\": "
             "'{\"symbol\": \"AAPL\"}'} <|endoftext|>\n\n\n"
             "FUNCTION RESPONSE: {\"stock_price\": \"$150.75\"}\n\n\n"
             "ASSISTANT: Apple (AAPL) is trading at $150.75. <|endoftext|>"),
}

REC_EMPTY = {"messages": []}
REC_JUNK = {"messages": [{"role": "assistant", "content": "   "}]}

# opencode export: HF messages with inline <think> reasoning + OpenAI tool_calls / tool results.
REC_OPENCODE = {"id": "ses_x", "source": "opencode", "agent": "coder", "model": "p/x", "messages": [
    {"role": "system", "content": "You are the coder agent."},
    {"role": "user", "content": "list usb"},
    {"role": "assistant", "content": "<think>run lsusb</think>",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "bash", "arguments": "{\"command\": \"lsusb\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "Bus 001 ..."},
    {"role": "assistant", "content": "Found it."},
]}

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  " if cond else "  FAIL") + "  " + msg)
    if not cond:
        FAILS.append(msg)


def _end_count(s: str) -> int:
    return s.count(cf.END)


def test_messages() -> None:
    print("[messages]")
    out = _render(REC_MESSAGES, "messages")
    check(out.startswith(cf.SYSTEM), "starts with <|system|>")
    check(cf.USER in out and cf.ASSISTANT in out, "has <|user|> and <|assistant|>")
    check(out.rstrip().endswith(cf.END) and _end_count(out) == 1, "ends with exactly one <|end|>")
    check("You are concise." in out and "Hello!" in out, "content preserved")


def test_conversations_and_dolly() -> None:
    print("[conversations + dolly]  (ShareGPT aliases, single-turn instruction)")
    conv = _render(REC_CONVERSATIONS, "conversations")
    check(conv.count(cf.USER) == 1 and conv.count(cf.ASSISTANT) == 1, "human→user, gpt→assistant")
    check(conv.rstrip().endswith(cf.END), "conversation ends with <|end|>")
    dolly = _render(REC_DOLLY, "dolly")
    check(dolly.startswith(cf.USER) and "Blue." in dolly, "dolly → user/assistant turn")
    check(_end_count(dolly) == 1, "dolly ends with one <|end|>")


def test_hh() -> None:
    print("[hh-rlhf]  (chosen transcript, Human/Assistant → user/assistant)")
    out = _render(REC_HH, "hh")
    check(out.startswith(cf.USER), "first turn is <|user|>")
    check(cf.ASSISTANT in out and "21." in out, "assistant turn present")
    check("Human:" not in out and "Assistant:" not in out, "raw Human:/Assistant: prefixes gone")
    check(_end_count(out) == 1, "ends with one <|end|>")


def test_glaive_tool_call() -> None:
    print("[glaive]  (tool call → <|tool_call|>, response → <|tool_result|>)")
    out = _render(REC_GLAIVE, "glaive")
    check(out.startswith(cf.SYSTEM), "system tool-defs first")
    check(cf.TOOL_CALL in out, "has <|tool_call|>")
    check(cf.TOOL_RESULT in out, "has <|tool_result|>")
    check(out.rstrip().endswith(cf.END) and _end_count(out) == 1, "ends with one <|end|>")
    check("<functioncall>" not in out and "FUNCTION RESPONSE:" not in out, "raw glaive markers gone")

    # the tool-call payload must be VALID JSON (Phase 1 parser + validity evals depend on it)
    payload = out.split(cf.TOOL_CALL + "\n", 1)[1].split("\n" + cf.TOOL_RESULT, 1)[0].strip()
    try:
        call = json.loads(payload)
        ok = call.get("name") == "get_stock_price"
        # arguments repaired from the single-quoted embedded string into a real object
        args = call.get("arguments")
        if isinstance(args, str):
            args = json.loads(args)
        ok = ok and args.get("symbol") == "AAPL"
        check(ok, f"tool-call JSON valid and correct: {payload}")
    except Exception as e:
        check(False, f"tool-call payload is not valid JSON ({e}): {payload!r}")

    # order: system → user → assistant?/tool_call → tool_result → assistant → end
    check(out.index(cf.TOOL_CALL) < out.index(cf.TOOL_RESULT), "tool_call precedes tool_result")


def test_opencode() -> None:
    print("[opencode]  (HF messages + <think> + tool_calls → unified)")
    out = _render(REC_OPENCODE, "opencode")
    check(out.startswith(cf.SYSTEM) and "You are the coder agent." in out, "system message rendered")
    check("<think>run lsusb</think>" in out, "inline <think> reasoning preserved")
    check(cf.TOOL_CALL in out and cf.TOOL_RESULT in out, "has <|tool_call|> and <|tool_result|>")
    check(out.rstrip().endswith(cf.END) and _end_count(out) == 1, "ends with one <|end|>")
    # tool-call payload is the same {"name","arguments":{...}} shape as glaive, args inlined to an object
    payload = out.split(cf.TOOL_CALL + "\n", 1)[1].split("\n" + cf.TOOL_RESULT, 1)[0].strip()
    try:
        call = json.loads(payload)
        ok = call.get("name") == "bash" and call.get("arguments", {}).get("command") == "lsusb"
        check(ok, f"tool-call payload valid + inlined args: {payload}")
    except Exception as e:
        check(False, f"tool-call payload not valid JSON ({e}): {payload!r}")
    check(out.index(cf.TOOL_CALL) < out.index(cf.TOOL_RESULT) < out.index("Found it."),
          "order: tool_call → tool_result → final assistant text")


def test_skips_empty() -> None:
    print("[skip]  (empty / whitespace-only records render to '')")
    check(_render(REC_EMPTY, "messages") == "", "empty message list → ''")
    check(_render(REC_JUNK, "messages") == "", "whitespace-only content → ''")
    check(cf.render([]) == "", "render([]) → '' (no bare <|end|>)")


def main() -> None:
    test_messages()
    test_conversations_and_dolly()
    test_hh()
    test_glaive_tool_call()
    test_opencode()
    test_skips_empty()
    if FAILS:
        print(f"\nFAIL — {len(FAILS)} check(s) failed:")
        for m in FAILS:
            print(f"  - {m}")
        raise SystemExit(1)
    print("\nOK — unified chat/tool renderer is consistent across all sources")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
