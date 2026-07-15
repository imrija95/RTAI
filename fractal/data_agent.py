"""Curriculum for a small AGENT — input is NATURAL ENGLISH, output is a tool / "caveman".

The user types plain English ("what is 5 times 6", "remember my dog is brown", "what's my dog");
the model must UNDERSTAND, route to a tool and extract the ARGUMENTS (copied from the input), OR reply
telegraphically in "caveman". Design principle: model = ROUTER + MEMORY in the weights; delegate the deterministic part.

Key point: the operator in calc is a WORD ("times/plus/minus") — the model just copies it, the mapping to +−*
is done later by the executor (agent.py). Args are always content words from the input → copied in the agent (copy-constrained).

Tools (rigid output grammar → grammar.py). TOOLS spec shared with grammar.py: (arg_types, NL templates).
The API keeps the shape of recall.py (batch → (x,y,w), accuracy) → train.py via --task agent.
"""

from __future__ import annotations

import random

import torch

OPS = [("+", "plus"), ("-", "minus"), ("*", "times")]      # (symbol for the executor, word in the NL and in the call)

# name → (arg_types, NATURAL English input templates). {a}{b}=num {op}=op-word {k}{v}{w}=word {t}=text
TOOLS = {
    "calc":      (["num", "op", "num"], [
        "what is {a} {op} {b}", "what's {a} {op} {b}", "how much is {a} {op} {b}",
        "can you calculate {a} {op} {b}", "{a} {op} {b} please", "compute {a} {op} {b}"]),
    "remember":  (["word", "word"], [
        "remember that my {k} is {v}", "please remember my {k} is {v}", "keep in mind my {k} is {v}",
        "my {k} is {v}", "note that my {k} is {v}", "don't forget my {k} is {v}"]),
    "recall":    (["word"], [
        "what is my {k}", "what's my {k}", "do you remember my {k}",
        "what did i say my {k} was", "tell me my {k}", "remind me of my {k}"]),
    "forget":    (["word"], [
        "forget my {k}", "please forget about my {k}", "delete my {k}", "drop my {k} from memory"]),
    "note":      (["text"], [
        "make a note to {t}", "remind me to {t}", "note that i need to {t}", "write down {t}"]),
    "list":      ([], [
        "list my notes", "what notes do i have", "show me everything", "what do you remember"]),
    "time":      ([], [
        "what time is it", "what's the time", "tell me the time", "do you know the time"]),
    "timer":     (["num"], [
        "set a timer for {a} minutes", "remind me in {a} minutes", "wake me in {a} minutes", "timer for {a}"]),
    "search":    (["text"], [
        "search for {t}", "look up {t}", "find information about {t}", "can you google {t}"]),
    "define":    (["word"], [
        "what does {k} mean", "define {k}", "what is the meaning of {k}", "explain the word {k}"]),
    "weather":   (["word"], [
        "what's the weather in {k}", "is it raining in {k}", "weather for {k}", "how's the weather in {k}"]),
    "translate": (["word"], [
        "translate {k} to french", "how do you say {k} in french", "what is {k} in spanish"]),
}

# natural English "chitchat" (→ reply, no tool)
CHITCHAT = ["hello", "hi there", "how are you", "thanks", "good morning", "what's up",
            "who are you", "you are great", "nice to meet you", "hey", "good night", "cool"]
# caveman output (procedural, telegraphic: no articles/filler, content words, "me")
CV_V = ("hear", "see", "know", "help", "think", "like", "want", "have", "do")
CV_ADJ = ("good", "big", "fast", "ok", "fine", "tired", "happy", "cold", "strong")
REPLY_PATS = ("me {V}", "me {V} {N}", "{N} {ADJ}", "me no {V}", "me want {N}", "you {ADJ}", "me here", "me {V} you")


class ToolGen:
    def __init__(self, tok, seed: int = 0):
        self.tok = tok
        pool = []
        for i in range(tok.get_vocab_size()):
            s = tok.decode([i])
            if len(s) > 3 and s[0] == " " and s[1:].isalpha() and s[1:].islower():
                pool.append(s.strip())
        random.Random(seed).shuffle(pool)
        cut = int(len(pool) * 0.85)
        self.train_words, self.test_words = pool[:cut], pool[cut:]
        print(f"[ToolGen] tools: {len(TOOLS)} | slot vocabulary: {len(pool)} "
              f"(train {len(self.train_words)} / held-out {len(self.test_words)})")

    def _caveman(self, pool):
        return random.choice(REPLY_PATS).format(V=random.choice(CV_V), ADJ=random.choice(CV_ADJ), N=random.choice(pool))

    def _episode(self, pool):
        """(intent_NL, bot, tool|None, args). ~16 % = reply. Args = content words (copyable from the input)."""
        if random.random() < 0.16:
            return random.choice(CHITCHAT), self._caveman(pool), None, []
        name = random.choice(list(TOOLS))
        spec, templates = TOOLS[name]
        tpl = random.choice(templates)
        if name == "calc":
            a, b, (_sym, opw) = random.randint(0, 9), random.randint(0, 9), random.choice(OPS)
            intent, args = tpl.format(a=a, op=opw, b=b), [str(a), opw, str(b)]
        elif name == "remember":
            k, v = random.choice(pool), random.choice(pool)
            intent, args = tpl.format(k=k, v=v), [k, v]
        elif spec == ["word"]:
            k = random.choice(pool)
            intent, args = tpl.format(k=k), [k]
        elif spec == ["num"]:
            a = random.randint(1, 30)
            intent, args = tpl.format(a=a), [str(a)]
        elif spec == ["text"]:
            args = random.sample(pool, random.randint(1, 3))
            intent = tpl.format(t=" ".join(args))
        else:                                                # time / list — no args
            intent, args = tpl, []
        bot = f"<tool> {name}{(' ' + ' '.join(args)) if args else ''} </tool>"
        return intent, bot, name, args

    def _e(self, s):
        return self.tok.encode(s).ids

    def _mem_episode(self, pool):
        """MEMORY in the weights: store 2-5 facts 'my K is V.' → query 'my Kq is' → answer Vq (single-token).
        Format aligns with agent.mem_write/mem_read. Fast-weights absorb the facts, the model reads."""
        n = random.randint(2, 5)
        keys = random.sample(pool, n)
        vals = [random.choice(pool) for _ in keys]
        facts = " ".join(f"my {k} is {v}." for k, v in zip(keys, vals))
        j = random.randrange(n)
        return f"{facts} my {keys[j]} is", vals[j]

    def batch(self, batch_size, seq_len, device, w_ans: float = 1.0, max_facts: int = 0, p_mem: float = 0.35):
        """Mix: ~p_mem memory episodes (memory in the weights) + the rest routing/reply episodes."""
        rows, wts = [], []
        for _ in range(batch_size):
            if random.random() < p_mem:
                store, ans = self._mem_episode(self.train_words)
                p, t, wv = self._e(store), self._e(" " + ans), 5.0    # heavy weight on the answer
            else:
                intent, bot, _, _ = self._episode(self.train_words)
                p, t, wv = self._e(f"User: {intent}\nBot:"), self._e(f" {bot}"), w_ans
            seq = p + t
            # w aligned to y=seq[1:]: predicting token seq[i] is weighted at position i−1 → w_ans starts at len(p)−1
            # (previously [0]*len(p) = off by one → for a 1-token answer the PAD after the answer was weighted, not the value → recall 0 %)
            w = [0.0] * (len(p) - 1) + [wv] * len(t)
            seq = (seq + [0] * (seq_len + 1))[:seq_len + 1]
            w = (w + [0.0] * (seq_len + 1))[:seq_len]
            rows.append(seq); wts.append(w)
        t_ = torch.tensor(rows, dtype=torch.long, device=device)
        wt = torch.tensor(wts, dtype=torch.float32, device=device)
        return t_[:, :-1], t_[:, 1:], wt

    @torch.no_grad()
    def mem_accuracy(self, model, device="cpu", n=128, held_out=True):
        """Recall a fact from fast-weight memory: store facts + query, check the 1st token of the value."""
        pool = self.test_words if held_out else self.train_words
        ok, was = 0, model.training
        model.eval()
        for _ in range(n):
            store, ans = self._mem_episode(pool)
            ids = torch.tensor([self._e(store)], dtype=torch.long, device=device)
            logits, _, _, _ = model(ids)
            ok += int(int(logits[0, -1].argmax()) == self._e(" " + ans)[0])
        model.train(was)
        return ok / n

    @torch.no_grad()
    def _gen(self, model, prompt, device, max_new=20):
        model.eval()
        ids = torch.tensor([self._e(prompt)], dtype=torch.long, device=device)
        states = model.init_states(1, device)
        gen, _ = model.generate_stream(ids, max_new, states, temperature=0.7, top_k=1)
        return self.tok.decode(gen[0].tolist())

    @torch.no_grad()
    def accuracy(self, model, distance=0, device="cpu", n=64, held_out=True, n_facts=0, guided=None):
        """dict: route (name) · full (name+args) · valid (syntactically). guided=callable → constrained."""
        import re
        pool = self.test_words if held_out else self.train_words
        ok_route, ok_full, valid = 0, 0, 0
        was = model.training
        for _ in range(n):
            intent, bot, tool, args = self._episode(pool)
            prompt = f"User: {intent}\nBot:"
            out = guided(model, self.tok, prompt, device) if guided else self._gen(model, prompt, device)
            m = re.search(r"<tool>\s*(\w+)\s*(.*?)\s*</tool>", out)
            if tool is None:
                got_route = m is None
                valid += int(m is None)
                got_full = got_route
            else:
                valid += int(m is not None)
                got_route = bool(m) and m.group(1) == tool
                got_full = got_route and m.group(2).split() == args
            ok_route += int(got_route); ok_full += int(got_full)
        model.train(was)
        return {"route": ok_route / n, "full": ok_full / n, "valid": valid / n}
