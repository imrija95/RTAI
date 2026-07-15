"""High-quality mixed corpus → a new data dir, via an EXISTING tokenizer — SAFE / non-destructive.

Streams several curated sources and weight-interleaves them at the document level, so the resulting
.bin is homogeneous nowhere and every source is represented throughout. Two recipes:

  base : pretrain-only quality mix — cosmopedia-v2 + fineweb-edu + TinyStories + Python.
  v2   : the base kept as the ~80% majority + a ~20% instruction/agentic layer (chat, tool-calls,
         assistant register) rendered into the ONE unified chat/tool format defined in
         fractal.chat_format (<|user|>/<|assistant|> turns, <|tool_call|>{json} / <|tool_result|>,
         terminated by <|end|>):
           - HuggingFaceTB/smoltalk        (SFT mix built for small models; messages)
           - glaiveai/glaive-function-calling-v2  (tool-call JSON syntax)
           - Anthropic/hh-rlhf             (helpful/harmless; 'chosen' transcript only)
           - databricks/databricks-dolly-15k (crisp human instructions)
           - allenai/tulu-3-sft-mixture    (diversity, small share)

A LOCAL opencode export (the opencode-training-exporter .jsonl, HF `messages` format with
tool_calls / <think> reasoning) can be mixed in with --opencode_file; it is rendered into the same
unified chat/tool format via fractal.chat_format (see the 'opencode' kind).

REUSES an existing tokenizer (no retrain) and writes to a SEPARATE dir — never overwrites
fractal_data*/ nor the tokenizer. Robust to flaky links: exponential-backoff reconnect + skip.

Run:  uv run python -m fractal.data_mix --recipe v2 --tokenizer fractal_tokenizer_32k.json \
          --out_dir fractal_data_v2 --max_tokens 350000000 \
          --opencode_file opencode_training_data.jsonl --opencode_weight 0.05
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from datasets import load_dataset

from fractal import chat_format as cf
from fractal import tokenizer as tk

EOT = "<|endoftext|>"

# Entry = (name, config, kind, weight). kind selects the serializer in _render().
MIX_BASE = [
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text", 0.40),
    ("HuggingFaceFW/fineweb-edu",   "sample-10BT",   "text", 0.30),
    ("roneneldan/TinyStories",       None,           "text", 0.15),
    ("codeparrot/codeparrot-clean",  None,           "text", 0.15),
]

# v2: base ~80% (majority for a small model) + instruction/agentic ~20%.
MIX_V2 = [
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text",     0.30),  # synthetic textbooks
    ("HuggingFaceFW/fineweb-edu",   "sample-10BT",   "text",     0.24),  # real-world edu web
    ("roneneldan/TinyStories",       None,           "text",     0.12),  # fluency scaffold
    ("codeparrot/codeparrot-clean",  None,           "text",     0.14),  # Python (reasoning + tools)
    ("HuggingFaceTB/smoltalk",       "all",          "messages", 0.09),  # chat SFT backbone
    ("glaiveai/glaive-function-calling-v2", None,    "glaive",   0.04),  # tool-call JSON
    ("Anthropic/hh-rlhf",            None,           "hh",       0.04),  # assistant register
    ("databricks/databricks-dolly-15k", None,        "dolly",    0.02),  # human instructions
    ("allenai/tulu-3-sft-mixture",   None,           "messages", 0.01),  # diversity
]

# instruction/agentic layer ONLY (reuse an already-tokenized base + concat) — smaller, more
# reliable streaming than re-fetching the huge base parquets over a flaky link.
MIX_INSTRUCT = [
    ("HuggingFaceTB/smoltalk",       "all",  "messages", 0.45),
    ("glaiveai/glaive-function-calling-v2", None, "glaive", 0.20),
    ("Anthropic/hh-rlhf",            None,   "hh",       0.20),
    ("databricks/databricks-dolly-15k", None,"dolly",    0.10),
    ("allenai/tulu-3-sft-mixture",   None,   "messages", 0.05),
]

# tool-heavy layer: glaive (pure tool-call transcripts) as the majority + some chat so conversation
# ability is not forgotten. Pair with --opencode_file for real agent trajectories.
MIX_TOOLS = [
    ("glaiveai/glaive-function-calling-v2", None, "glaive",   0.60),  # tool-call syntax, majority
    ("HuggingFaceTB/smoltalk",       "all",  "messages", 0.25),        # keep general chat
    ("Anthropic/hh-rlhf",            None,   "hh",       0.15),        # assistant register
]

RECIPES = {"base": MIX_BASE, "v2": MIX_V2, "instruct": MIX_INSTRUCT, "tools": MIX_TOOLS}

def _segments(ex, kind):
    """Build unified (role, content) segments for one chat/tool record (not for 'text')."""
    if kind == "messages":                                   # smoltalk / tulu-3 / ultrachat
        return cf.turns_from_messages(ex.get("messages"))
    if kind == "conversations":                              # ShareGPT-style (hermes-fc / OpenHermes)
        return cf.turns_from_messages(ex.get("conversations"))
    if kind == "hh":                                         # Anthropic hh-rlhf: use 'chosen' only
        return cf.hh_segments(ex.get("chosen"))
    if kind == "dolly":
        ins, ctx, resp = ex.get("instruction", ""), ex.get("context", ""), ex.get("response", "")
        u = ins + (("\n" + ctx) if ctx else "")
        return [("user", u), ("assistant", resp)]
    if kind == "glaive":                                     # system holds tool defs, chat is a transcript
        return cf.glaive_segments(ex.get("system", ""), ex.get("chat", ""))
    if kind == "opencode":                                   # local opencode export: HF messages + tools
        return cf.opencode_segments(ex.get("messages"))
    return []


def _render(ex, kind):
    """Serialize one record into the unified chat/tool format. '' skips a malformed/empty record."""
    if kind == "text":
        return (ex.get("text") or ex.get("content") or "").strip()
    return cf.render(_segments(ex, kind))


def _render_pieces(ex, kind):
    """[(text, trainable)] pieces for Phase-2 loss masking. Pretrain 'text' is all trainable
    (plain LM); chat/tool records mark only assistant + <|tool_call|> + <|end|> as trainable.
    Returns [] to skip an empty/malformed record."""
    if kind == "text":
        t = (ex.get("text") or ex.get("content") or "").strip()
        return [(t, True)] if t else []
    return cf.render_pieces(_segments(ex, kind))


class Source:
    """One streamed dataset that yields one non-empty rendered document per call and survives link
    drops by reconnecting and .skip()ing the docs it has already consumed (no output duplication)."""

    def __init__(self, name, config, kind, weight, retries=40, data_files=None):
        self.name, self.config, self.kind, self.weight = name, config, kind, weight
        self.data_files = data_files          # local file(s) → load via the "json" builder, not the Hub
        self.retries = retries
        self.docs_seen = 0
        self.tokens = 0
        self.exhausted = False
        self._it = None

    def _open(self):
        if self.data_files:                   # local jsonl (e.g. the opencode export) — no Hub fetch
            ds = load_dataset("json", data_files=self.data_files, split="train", streaming=True)
        elif self.config:
            ds = load_dataset(self.name, self.config, split="train", streaming=True)
        else:
            ds = load_dataset(self.name, split="train", streaming=True)
        if self.docs_seen:
            ds = ds.skip(self.docs_seen)
        self._it = iter(ds)

    def _next(self, render_fn):
        """Yield the next non-empty rendered doc (via render_fn), surviving link drops."""
        attempt = 0
        while attempt <= self.retries:
            try:
                if self._it is None:
                    self._open()
                for ex in self._it:
                    self.docs_seen += 1
                    r = render_fn(ex)
                    if r:
                        return r
                self.exhausted = True
                return None
            except Exception as e:                # link drop → exponential backoff + reconnect + skip
                attempt += 1
                wait = min(30.0, 1.5 * 2 ** min(attempt, 5))
                print(f"  [drop {self.name} {attempt}/{self.retries}: {type(e).__name__}] "
                      f"wait {wait:.0f}s → reconnect+skip {self.docs_seen}", flush=True)
                time.sleep(wait)
                self._it = None
        self.exhausted = True
        return None

    def next_text(self):
        return self._next(lambda ex: _render(ex, self.kind))

    def next_pieces(self):
        return self._next(lambda ex: _render_pieces(ex, self.kind) or None)


def _encode_pieces(tok, pieces):
    """Tokenize [(text, trainable)] → (ids, mask). The first piece carries the byte-BPE leading
    space (matching the non-masked path's `encode(' ' + doc)`). mask is 1 on trainable tokens."""
    ids, mask, first = [], [], True
    for text, trainable in pieces:
        enc = tok.encode((" " + text) if first else text).ids
        first = False
        if not enc:
            continue
        ids.extend(enc)
        mask.extend([1 if trainable else 0] * len(enc))
    return ids, mask


def prepare(out_dir, tokenizer_path, max_tokens, recipe, opencode_file=None, opencode_weight=0.05,
            emit_mask=False):
    os.makedirs(out_dir, exist_ok=True)
    tok = tk.load(tokenizer_path)
    eot = tok.token_to_id(EOT)
    assert eot is not None, f"tokenizer {tokenizer_path} must define {EOT}"

    sources = [Source(n, c, k, w) for n, c, k, w in RECIPES[recipe]]
    if opencode_file:                         # mix in a local opencode export (HF messages + tools)
        sources.append(Source("opencode", None, "opencode", opencode_weight, data_files=opencode_file))
    total_w = sum(s.weight for s in sources)
    val_n = max(50_000, max_tokens // 100)
    need = max_tokens + val_n
    targets = {s.name: s.weight / total_w * need for s in sources}
    print(f"recipe {recipe} | tokenizer {tokenizer_path} (vocab {tok.get_vocab_size()}) | "
          f"target {need:,} tok → {out_dir}" + (" | +loss mask" if emit_mask else ""), flush=True)
    for s in sources:
        print(f"  {s.name} [{s.config}] {s.kind} w{s.weight} → ~{int(targets[s.name]):,} tok", flush=True)

    # Stream tokens straight to disk in ~1M-token chunks (never hold the whole corpus → no OOM).
    # First val_n tokens go to val.bin, the rest to train.bin. With emit_mask, a parallel uint8
    # {train,val}.mask.bin is written token-aligned (1 = train loss on this token).
    vf = open(f"{out_dir}/val.bin", "wb")
    tf = open(f"{out_dir}/train.bin", "wb")
    vmf = open(f"{out_dir}/val.mask.bin", "wb") if emit_mask else None
    tmf = open(f"{out_dir}/train.mask.bin", "wb") if emit_mask else None
    buf: list[int] = []
    mbuf: list[int] = []
    total = val_written = last_print = 0

    def _flush():
        nonlocal buf, mbuf, val_written
        if not buf:
            return
        arr = np.array(buf, dtype=np.uint16)
        marr = np.array(mbuf, dtype=np.uint8) if emit_mask else None
        buf, mbuf = [], []
        if val_written < val_n:
            k = min(val_n - val_written, len(arr))
            if k:
                arr[:k].tofile(vf)
                if emit_mask:
                    marr[:k].tofile(vmf)
                val_written += k
            arr = arr[k:]
            if emit_mask:
                marr = marr[k:]
        if len(arr):
            arr.tofile(tf)
            if emit_mask:
                marr.tofile(tmf)

    while total < need and not all(s.exhausted for s in sources):
        live = [s for s in sources if not s.exhausted]
        s = max(live, key=lambda s: (targets[s.name] - s.tokens) / max(targets[s.name], 1.0))
        if emit_mask:
            pieces = s.next_pieces()
            if pieces is None:
                continue
            enc, emask = _encode_pieces(tok, pieces)
            if not enc:
                continue
            buf.extend(enc); mbuf.extend(emask)
            buf.append(eot); mbuf.append(0)           # doc separator is structural (masked)
        else:
            t = s.next_text()
            if t is None:
                continue
            enc = tok.encode(" " + t).ids
            buf.extend(enc)
            buf.append(eot)
        n = len(enc) + 1
        s.tokens += n
        total += n
        if len(buf) >= 1_000_000:
            _flush()
            if total - last_print >= 10_000_000:
                last_print = total
                mix = " | ".join(f"{s.name.split('/')[-1]} {100*s.tokens/max(total,1):.0f}%" for s in sources)
                print(f"  …{total:,} tok  [{mix}]", flush=True)

    _flush()
    vf.close()
    tf.close()
    if emit_mask:
        vmf.close(); tmf.close()
    if total < 2 * val_n:
        raise SystemExit(f"too few tokens ({total:,}) — sources exhausted early; try again")
    print(f"{out_dir}/val.bin: {val_written:,} | {out_dir}/train.bin: {total-val_written:,} tokens", flush=True)
    print("mix (final tokens per source):", flush=True)
    for s in sources:
        print(f"  {s.name}: {s.tokens:,} tok ({100*s.tokens/max(total,1):.1f}%)"
              + (" [exhausted]" if s.exhausted else ""), flush=True)
    print(f"done. Train with: --data_dir {out_dir}" + (" --task chat" if emit_mask else ""), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="fractal_data_mix")
    ap.add_argument("--tokenizer", default="fractal_tokenizer.json")
    ap.add_argument("--max_tokens", type=int, default=300_000_000)
    ap.add_argument("--recipe", default="base", choices=list(RECIPES))
    ap.add_argument("--opencode_file", default=None,
                    help="local opencode export .jsonl (HF messages) to mix in as an 'opencode' source")
    ap.add_argument("--opencode_weight", type=float, default=0.05,
                    help="document-level mix weight for --opencode_file")
    ap.add_argument("--emit_mask", action="store_true",
                    help="also write {train,val}.mask.bin (loss on assistant + <|tool_call|> + <|end|> "
                         "only) for Phase-2 masked training (--task chat)")
    a = ap.parse_args()
    prepare(a.out_dir, a.tokenizer, a.max_tokens, a.recipe,
            opencode_file=a.opencode_file, opencode_weight=a.opencode_weight, emit_mask=a.emit_mask)
    os._exit(0)      # skip datasets' streaming-thread teardown race at exit
