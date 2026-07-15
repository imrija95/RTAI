"""Guided (constrained) decoding for tool calls — LEGACY scaffold (see docs/ROADMAP.md).

NOTE: this module is part of the retired prove-in-small scaffold and is no longer used by the agent
loop. `fractal/agent.py` now generates tool calls emergently in the unified format
(`fractal/chat_format.py`) and uses only an optional inline JSON-validity guard (`--json-guard`).
Kept for reference / small-scale experiments; not on the scale-invariant path.

A tiny model routes well, but (a) cannot emit the `<tool>…</tool>` scaffold, (b) cannot copy argument
content out of a natural sentence. Solution:
  - guided_call: the structure is FORCED, the name is chosen by SCORING (routing, guaranteed valid),
    arguments are generated COPY-CONSTRAINED (only tokens from the input → a copy, even unseen words).
  - wants_tool: reply-gate — should a tool be called, or is a terse reply enough?

  from fractal.grammar import guided_call, wants_tool
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from fractal.data_agent import TOOLS

_NEG = -1e9


@torch.no_grad()
def _feed(model, states, ids, device):
    """Run ids through the stream, return (logits_for_next, new_state)."""
    return model.forward_stream(torch.tensor([ids], device=device), states)


# command words → almost certainly a TOOL (reply-gate hint; the model still picks WHICH)
HINTS = {"what", "how", "set", "make", "tell", "remind", "list", "find", "search", "look",
         "define", "translate", "calculate", "compute", "forget", "remember", "note", "keep",
         "timer", "weather", "time", "delete", "drop", "explain", "show", "add", "minus", "plus",
         "times", "google", "meaning", "raining", "temperature"}
# pure small-talk → reply
CHIT = {"hello", "hi", "hey", "thanks", "thank", "morning", "night", "cool", "nice", "great", "who", "you"}
# function + command words that are NOT arguments (copy content only). Op-words (plus/minus/times) do NOT belong here.
STOP = {"my", "is", "the", "a", "an", "to", "for", "of", "in", "on", "at", "what", "whats", "that",
        "please", "do", "you", "i", "me", "can", "how", "much", "did", "say", "was", "it", "and",
        "with", "your", "remind", "remember", "note", "keep", "forget", "delete", "drop", "set",
        "make", "tell", "list", "show", "find", "search", "look", "up", "define", "explain",
        "meaning", "word", "translate", "weather", "time", "timer", "calculate", "compute", "google",
        "information", "about", "minutes", "need", "everything", "notes", "have", "mean", "french",
        "spanish", "raining", "hows", "google", "there", "are", "s", "us"}


def _intent(prompt):
    """Extract the plain user text from 'User: <text>\\nBot:'."""
    s = prompt.split("\nBot:")[0]
    return s[6:] if s.startswith("User: ") else s


def extract_args(name, intent):
    """DETERMINISTICALLY extract arguments from the user sentence (not by model generation). The model
    only routes; slots = content words (function words in STOP dropped) / numbers+op for calc. Robust even
    for unseen words (copied from the input). = the principle 'model routes, deterministic code delegates'."""
    words = re.findall(r"[a-z0-9]+", intent.lower())
    content = [w for w in words if w not in STOP]
    if name == "calc":
        nums = re.findall(r"\d+", intent)
        op = next((w for w in words if w in ("plus", "minus", "times")),
                  next((c for c in intent if c in "+-*"), "plus"))
        return [nums[0], op, nums[1]] if len(nums) >= 2 else nums
    if name == "timer":
        nums = re.findall(r"\d+", intent)
        return nums[:1]
    if name == "remember":
        return content[:2]
    if name in ("recall", "forget", "define", "weather", "translate"):
        return content[:2] if content else []
    if name in ("note", "search"):
        return content
    return []                                            # time, list — no args


@torch.no_grad()
def wants_tool(model, tok, prompt, device, n_peek: int = 5):
    """Reply-gate: TOOL vs just reply. Bias toward a tool (commands are more common). Word hint (a command
    word in the input) + fallback to free generation (fragments 'tool'/'<'). A pure greeting → reply."""
    words = re.findall(r"[a-z]+", _intent(prompt).lower())
    if any(w in HINTS for w in words):
        return True
    if words and all(w in CHIT or w in {"me", "are", "is", "there", "your"} for w in words):
        return False
    ids = tok.encode(prompt).ids
    st = model.init_states(1, device)
    lg, st = _feed(model, st, ids, device)
    out = []
    for _ in range(n_peek):
        t = int(lg[0, -1].argmax()); out.append(t)
        lg, st = _feed(model, st, [t], device)
    dec = tok.decode(out)
    return ("tool" in dec) or ("<" in dec)


@torch.no_grad()
def guided_call(model, tok, prompt, device, arg_budget: int = 12):
    model.eval()
    open_ids = tok.encode(prompt + " <tool>").ids
    st = model.init_states(1, device)
    lg, st = _feed(model, st, open_ids, device)

    # --- routing: tool with the highest logprob for ' name' (guaranteed valid name) ---
    def score(name):
        s, cur, lp = [x.clone() for x in st], lg, 0.0
        for tid in tok.encode(" " + name).ids:
            lp += float(F.log_softmax(cur[0, -1], dim=-1)[tid])
            cur, s = _feed(model, s, [tid], device)
        return lp, s, cur
    name = max(TOOLS, key=lambda n: score(n)[0])            # routing: tool with the highest logprob
    args = extract_args(name, _intent(prompt))              # args deterministically from the input (not generated)
    return f"<tool> {name}{(' ' + ' '.join(args)) if args else ''} </tool>"
