"""The one chat/tool protocol — used everywhere (data rendering, agent loop, chat).

ONE template used by data rendering, the agent loop, and tool-result ingestion. Legacy checkpoints
encode the literal strings with their existing byte BPE; Natural Cortex trains a fresh tokenizer
where every marker is one atomic special token.

    <|system|>\n{system}
    <|user|>\n{user text}
    <|assistant|>\n{assistant text}
    <|tool_call|>\n{json}
    <|tool_result|>\n{result}
    <|assistant|>\n{final text}
    <|end|>

Roles: system, user, assistant, tool_call, tool_result. A conversation is a list of
(role, content) segments; `render()` joins them and terminates with <|end|>. The model learns to
emit <|tool_call|> and <|end|> itself — there is no FSM and no keyword routing (see docs/ROADMAP.md).
"""

from __future__ import annotations

import json
import re

# --- The markers (literal strings; byte-BPE encodes them as ordinary tokens) ------------------
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
TOOL_CALL = "<|tool_call|>"
TOOL_RESULT = "<|tool_result|>"
SKILL = "<|skill|>"
TEACH = "<|teach|>"
END = "<|end|>"

ROLES = ("system", "user", "assistant", "tool_call", "tool_result", "skill", "teach")
_MARKER = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT,
           "tool_call": TOOL_CALL, "tool_result": TOOL_RESULT,
           "skill": SKILL, "teach": TEACH}

# roles whose content is a tokens-of-interest span for a chat template consumer; kept for reuse.
ASSISTANT_ROLES = ("assistant", "tool_call")


def render(segments) -> str:
    """Serialize a conversation (list of (role, content) segments) to flat training text.

    Empty/whitespace-only segments are dropped. Returns '' if nothing is left (so a malformed
    record is skipped, not emitted as a bare <|end|>). Every non-empty conversation ends in <|end|>.
    """
    parts = []
    for role, content in segments:
        c = "" if content is None else str(content).strip()
        if not c:
            continue
        marker = _MARKER.get(role)
        if marker is None:
            continue
        parts.append(f"{marker}\n{c}")
    if not parts:
        return ""
    return "\n".join(parts) + "\n" + END


# roles whose tokens the model must LEARN to emit (loss on); the rest is context the harness supplies
_TRAIN_ROLES = {"assistant", "tool_call"}


def render_pieces(segments):
    """Like render(), but return [(text, trainable)] pieces whose concatenation EQUALS render().

    trainable=True marks the tokens the model must learn to generate — assistant content, the whole
    <|tool_call|>{json}, and the terminating <|end|>. The structural markers the harness itself
    supplies at inference (<|system|>/<|user|>/<|tool_result|> and the <|assistant|> opener) and
    their context content are False, so Phase-2 cross-entropy is applied only to what the model
    actually produces. Returns [] for an all-empty conversation (mirrors render()→'')."""
    parts = [(role, "" if c is None else str(c).strip()) for role, c in segments]
    parts = [(role, c) for role, c in parts if c and _MARKER.get(role)]
    if not parts:
        return []
    pieces = []
    for i, (role, content) in enumerate(parts):
        head = ("" if i == 0 else "\n") + f"{_MARKER[role]}\n"
        if role == "assistant":
            pieces.append((head, False))            # <|assistant|>\n opener is supplied by the harness
            pieces.append((content, True))          # the assistant's answer is learned
        elif role == "tool_call":
            pieces.append((head + content, True))   # the model emits the whole tool call
        else:                                       # system / user / tool_result / teaching metadata
            pieces.append((head + content, False))
    pieces.append(("\n" + END, True))               # the model learns to stop
    return pieces


def turns_from_messages(messages, role_map=None):
    """Normalize an OpenAI/ShareGPT-style message list to [(role, content)] segments.

    `role_map` maps source role names onto ours (defaults to identity for our own names, plus the
    common ShareGPT aliases). Unknown roles fall back to 'user'.
    """
    rm = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool_result",
          "user": "user", "assistant": "assistant", "function": "tool_result"}
    if role_map:
        rm.update(role_map)
    out = []
    for m in messages or []:
        role = rm.get(m.get("role") or m.get("from"), "user")
        out.append((role, m.get("content") or m.get("value") or ""))
    return out


# --- glaive-function-calling-v2 → unified segments -------------------------------------------
# The raw transcript uses  USER:/ASSISTANT:/FUNCTION RESPONSE:  turn prefixes; a tool call is
# `<functioncall> {json} <|endoftext|>` inside an ASSISTANT turn, where `arguments` is a
# single-quoted embedded JSON string (not valid JSON as-is). We split those out into their own
# tool_call segments and repair the JSON so downstream (Phase 1 parser, validity evals) can load it.
_GLAIVE_SPLIT = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
_FUNCTIONCALL = re.compile(r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)", re.DOTALL)


def repair_toolcall_json(payload: str) -> str:
    """Best-effort repair of glaive's `{"name": .., "arguments": '{...}'}` into valid JSON text.

    glaive embeds the arguments object as a single-quoted string. If the whole payload parses,
    return it re-dumped (compact); otherwise strip the surrounding single quotes around the
    arguments object and retry. Falls back to the stripped/raw string if it still won't parse."""
    s = payload.strip()
    try:
        return json.dumps(json.loads(s), ensure_ascii=False)
    except Exception:
        pass
    fixed = re.sub(r":\s*'(\{.*?\})'", lambda m: ": " + m.group(1), s, flags=re.DOTALL)
    try:
        obj = json.loads(fixed)
        if isinstance(obj.get("arguments"), str):          # arguments still a JSON string → inline it
            try:
                obj["arguments"] = json.loads(obj["arguments"])
            except Exception:
                pass
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return fixed


def _glaive_assistant_segments(text: str):
    """One ASSISTANT turn → assistant text (if any) then a tool_call segment (if any)."""
    text = text.replace("<|endoftext|>", "").strip()
    m = _FUNCTIONCALL.search(text)
    if not m:
        return [("assistant", text)]
    pre = text[:m.start()].strip()
    segs = []
    if pre:
        segs.append(("assistant", pre))
    segs.append(("tool_call", repair_toolcall_json(m.group(1))))
    return segs


def glaive_segments(system: str, chat: str):
    """Parse a glaive record (system tool-defs + chat transcript) into unified segments."""
    segs = []
    sys = (system or "").strip()
    if sys.startswith("SYSTEM:"):
        sys = sys[len("SYSTEM:"):].strip()
    if sys:
        segs.append(("system", sys))
    parts = _GLAIVE_SPLIT.split(chat or "")
    # parts = [pre, marker, body, marker, body, ...]; ignore any leading pre-text.
    i = 1
    while i < len(parts):
        marker = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2
        body = body.strip()
        if marker == "USER:":
            segs.append(("user", body))
        elif marker == "FUNCTION RESPONSE:":
            segs.append(("tool_result", body.replace("<|endoftext|>", "").strip()))
        elif marker == "ASSISTANT:":
            segs.extend(_glaive_assistant_segments(body))
    return segs


_HH_SPLIT = re.compile(r"\n\n(Human|Assistant):")


def hh_segments(chosen: str):
    """Anthropic hh-rlhf 'chosen' transcript → unified segments.

    The transcript is a flat string with `\\n\\nHuman:` / `\\n\\nAssistant:` prefixes (the very first
    turn may have no leading blank line). Split on those prefixes into role-tagged segments."""
    s = (chosen or "").strip()
    if not s:
        return []
    if s.startswith("Human:"):                    # normalize the first turn to the split shape
        s = "\n\n" + s
    parts = _HH_SPLIT.split(s)                     # [pre, role, body, role, body, ...]
    segs = []
    i = 1
    while i < len(parts):
        role = "user" if parts[i] == "Human" else "assistant"
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        segs.append((role, body))
        i += 2
    return [(r, c) for r, c in segs if c]


# --- opencode HF-messages export → unified segments ------------------------------------------
# Records from the opencode-training-exporter are in the standard HF chat format:
#   {"messages": [{"role": system|user|assistant|tool, "content", "tool_calls"?, "tool_call_id"?}]}
# Assistant reasoning is inlined as <think>...</think> in `content` (kept as-is — the model learns
# to emit it). Tool calls use the OpenAI convention (`tool_calls`, `arguments` a JSON string) and
# tool results are `role:"tool"` messages. We render tool calls to <|tool_call|> with the SAME
# {"name","arguments":{...}} payload as glaive, and tool results to <|tool_result|>.


# A single tool result (e.g. a file dump / long command output) can be tens of KB and would
# dominate a training window; cap it so the model still sees the shape without the noise.
TOOL_RESULT_CAP = 1500


def _cap(text, limit=TOOL_RESULT_CAP):
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated {len(text) - limit} chars]"


def opencode_segments(messages):
    """Normalize an opencode HF-messages list to unified [(role, content)] segments.
    Oversized tool results are truncated (see TOOL_RESULT_CAP)."""
    segs = []
    for m in messages or []:
        role = m.get("role")
        if role in ("system", "user"):
            segs.append((role, m.get("content") or ""))
        elif role == "assistant":
            content = m.get("content") or ""
            if content.strip():                       # keep inline <think>...</think>
                segs.append(("assistant", content))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):             # OpenAI arguments is a JSON string → inline it
                    try:
                        args = json.loads(args)
                    except (ValueError, TypeError):
                        pass
                segs.append(("tool_call", json.dumps({"name": fn.get("name") or "", "arguments": args},
                                                      ensure_ascii=False)))
        elif role == "tool":
            segs.append(("tool_result", _cap(m.get("content") or "")))
    return segs
