"""Emergent persistent AGENT — the real, scale-invariant agent loop (ROADMAP Phase 1).

The model generates in the ONE unified chat/tool format (fractal.chat_format), carrying its
persistent fast-weight state W across turns AND restarts (VIBE #4). The model DECIDES when to use a
tool by emitting `<|tool_call|>` {json}; the harness parses it, executes it via a clean tool
registry, and streams the `<|tool_result|>…` back THROUGH the model (so W absorbs it), then
generation continues until `<|end|>`. Conversation history and tool results live in W, not in a
growing prompt window — this is the differentiator (O(1) state, no KV cache).

No FSM, no keyword routing, no copy-constrained arguments — `grammar.py` is NOT used here. An
optional inference-time JSON-validity guard is available via `--json-guard` (off by default; not a
crutch). Output quality scales with training/size (VIBE #7): on a checkpoint not yet trained on this
format the output is weak and the model may not emit the markers reliably, but this is the real loop
that scales — Phase 2 trains exactly it.

  uv run python -m fractal.agent --ckpt <ckpt.pt> --tokenizer fractal_tokenizer_32k.json
  uv run python -m fractal.agent --ckpt ... --fresh           # reset memory (start from W0)
"""

from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import datetime
import json
import operator
import os

import torch
import torch.nn.functional as F

from fractal import chat_format as cf
from fractal import persist, tokenizer as tk

# markers that end an assistant span (the model emits these as ordinary byte-BPE tokens)
_STOP_MARKERS = (cf.TOOL_CALL, cf.END, cf.USER, cf.SYSTEM)


# ---- tool registry (real executors; args are the parsed JSON `arguments` object) -------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node):
    """Evaluate a numeric arithmetic AST — no names, calls, or attribute access."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def t_calc(args):
    """Arithmetic. args: {"expression": "2 + 3*4"} or {"a": .., "b": .., "op": "+|-|*|/"}."""
    expr = args.get("expression")
    if expr is None and "a" in args and "b" in args:
        expr = f"({args['a']}){args.get('op', '+')}({args['b']})"
    if not expr:
        return "error: calc needs 'expression' or 'a'/'b'"
    try:
        return str(_safe_eval(ast.parse(str(expr), mode="eval").body))
    except Exception as e:
        return f"error: cannot evaluate ({e})"


def t_time(args):
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def t_echo(args):
    return str(args.get("text", ""))


REGISTRY = {"calc": t_calc, "time": t_time, "echo": t_echo}


# ---- tool-call parsing (the model's emitted JSON) --------------------------------------------
def _extract_json(text):
    """Return the first balanced {...} substring of `text`, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_tool_call(payload, json_guard=False):
    """Parse an emitted tool call into (name, arguments_dict) or (None, error_string).

    Default: strict json.loads on the first balanced object. With --json-guard, fall back to the
    lenient repair in chat_format (single-quoted args, arguments-as-string) before giving up."""
    blob = _extract_json(payload)
    if not blob:
        return None, "error: no JSON object in tool call"
    try:
        call = json.loads(blob)
    except (ValueError, TypeError):
        if not json_guard:
            return None, "error: tool-call JSON is invalid (enable --json-guard to repair)"
        call = None                                   # best-effort repair: glaive-style, then python-dict
        for attempt in (lambda: json.loads(cf.repair_toolcall_json(blob)), lambda: ast.literal_eval(blob)):
            try:
                cand = attempt()
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(cand, dict):
                call = cand
                break
        if call is None:
            return None, "error: tool-call JSON is invalid even after repair"
    name = call.get("name") or ""
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            args = {"text": args}
    if not isinstance(args, dict):
        args = {}
    return name, args


def execute_tool(name, args):
    fn = REGISTRY.get(name)
    if fn is None:
        return f"error: no tool named {name!r} (have: {', '.join(sorted(REGISTRY))})"
    try:
        return str(fn(args))
    except Exception as e:
        return f"error: tool {name} failed ({e})"


# ---- generation primitives (carry / update W) -----------------------------------------------
@torch.no_grad()
def _feed(model, states, ids, dev, autonomous_evidence=False):
    """Run ids through persistent memory and optionally credit externally observed outcomes."""
    track_evidence = (
        autonomous_evidence
        and getattr(model.cfg, "event_algebra", False)
        and len(ids) > 1
    )
    before = [state.clone() for state in states] if track_evidence else None
    logits, states = model.forward_stream(torch.tensor([ids], device=dev), states)
    if track_evidence:
        from fractal import feedback
        credit = feedback.observed_surprise(logits, ids)
        evidence = feedback.recent_evidence(
            before, states, len(ids), getattr(model.cfg, "eligibility_decay", 0.95))
        norm = feedback.apply_to_state(states, evidence, credit, lr=0.05, max_fraction=0.02)
        model._last_autonomous_credit = {"credit": credit, "update_norm": norm,
                                         "tokens": len(ids)}
    return logits, states


def _sample(logits, temperature, top_k):
    lg = logits[:, -1, :] / max(temperature, 1e-6)
    if top_k:
        v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
        lg = lg.masked_fill(lg < v[:, [-1]], -float("inf"))
    return int(torch.multinomial(F.softmax(lg, dim=-1), num_samples=1))


@torch.no_grad()
def _gen_span(model, tok, states, logits, dev, stop_markers, max_new, temperature, top_k, extra_stop=()):
    """Sample until a stop marker appears in the decoded text (or max_new). Every token is fed into
    W. Returns (text_before_marker, full_text, states, logits, hit_marker_or_None)."""
    gen_ids, text, hit = [], "", None
    stops = tuple(stop_markers) + tuple(extra_stop)
    for _ in range(max_new):
        nxt = _sample(logits, temperature, top_k)
        gen_ids.append(nxt)
        logits, states = _feed(model, states, [nxt], dev)
        text = tok.decode(gen_ids)
        hit = next((s for s in stops if s in text), None)
        if hit:
            break
    before = text.split(hit)[0] if hit else text
    return before, text, states, logits, hit


@torch.no_grad()
def run_turn(model, tok, states, user_text, dev, *, max_new=200, max_tool_calls=6,
             temperature=0.8, top_k=40, json_guard=False, skill_id=None):
    """One user turn through the emergent loop. Returns (transcript, states). W carries forward."""
    prime = tok.encode(f"\n{cf.USER}\n{user_text}\n{cf.ASSISTANT}\n").ids
    skill_score = -1.0
    if skill_id is None and getattr(model, "skill_cortex", None) is not None \
            and model.skill_cortex.auto_route:
        route_ids = torch.tensor([prime], device=dev, dtype=torch.long)
        skill_id, skill_score = model.route_skill_from_ids(route_ids)
    if skill_id is not None:
        skill_score = 1.0
    model._last_skill_route = {"expert_id": skill_id, "score": skill_score}
    cortex = getattr(model, "skill_cortex", None)
    skill_context = (
        (cortex.force(skill_id) if skill_id is not None else cortex.suspend())
        if cortex is not None else nullcontext()
    )
    with skill_context:
        logits, states = _feed(model, states, prime, dev, autonomous_evidence=True)
        transcript, tool_calls = [], 0

        while True:
            span, _, states, logits, hit = _gen_span(
                model, tok, states, logits, dev, _STOP_MARKERS, max_new, temperature, top_k)
            if span.strip():
                transcript.append(("assistant", span.strip()))

            if hit != cf.TOOL_CALL:                   # role marker / token budget → done
                return transcript, states

            tool_calls += 1
            if tool_calls > max_tool_calls:
                transcript.append(("note", "tool-call budget exceeded"))
                return transcript, states

            payload, _, states, logits, _ = _gen_span(
                model, tok, states, logits, dev,
                (cf.TOOL_RESULT, cf.ASSISTANT, cf.END, cf.USER), max_new=128,
                temperature=temperature, top_k=top_k)
            name, args = parse_tool_call(payload, json_guard)
            result = execute_tool(name, args) if name else args
            transcript.append(("tool_call", {"name": name, "arguments": args if name else {}}))
            transcript.append(("tool_result", result))

            back = tok.encode(f"\n{cf.TOOL_RESULT}\n{result}\n{cf.ASSISTANT}\n").ids
            logits, states = _feed(model, states, back, dev, autonomous_evidence=True)


def _print_turn(transcript):
    for kind, val in transcript:
        if kind == "assistant":
            print(f"bot> {val}")
        elif kind == "tool_call":
            print(f"     ⮑ call {val['name']}({json.dumps(val['arguments'], ensure_ascii=False)})")
        elif kind == "tool_result":
            r = val if len(val) <= 200 else val[:200] + "…"
            print(f"     ⮐ {r}")
        elif kind == "note":
            print(f"     [{val}]")
    if not any(k == "assistant" for k, _ in transcript):
        print("bot> (no reply)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", default="fractal_agent_state.pt")
    ap.add_argument("--tokenizer", default="fractal_tokenizer.json",
                    help="MUST match the tokenizer the checkpoint was trained with (e.g. 32k → fractal_tokenizer_32k.json)")
    ap.add_argument("--fresh", action="store_true", help="reset memory (start from W0)")
    ap.add_argument("--json-guard", action="store_true",
                    help="lenient inference-time repair of malformed tool-call JSON (default off)")
    ap.add_argument("--max_new", type=int, default=200, help="max tokens per assistant span")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = persist.load_model(args.ckpt, dev)
    model.eval()
    tok = tk.load(args.tokenizer)
    states = model.init_states(1, dev) if (args.fresh or not os.path.exists(args.state)) \
        else persist.load_states(args.state, dev)

    print("emergent agent ready — unified <|user|>/<|assistant|>/<|tool_call|>/<|tool_result|> loop.")
    print(f"tools: {', '.join(sorted(REGISTRY))} | json-guard: {'on' if args.json_guard else 'off'} "
          f"| memory persists to {args.state}. (empty line = quit)\n")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            break
        transcript, states = run_turn(model, tok, states, line, dev, max_new=args.max_new,
                                      temperature=args.temperature, top_k=args.top_k,
                                      json_guard=args.json_guard)
        _print_turn(transcript)
        persist.save_states(args.state, states)      # memory survives restart (VIBE #8)


if __name__ == "__main__":
    main()
