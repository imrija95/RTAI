This file is to be edited only on user's explicit agreement:

What this model IS:
- naturally fractal language model

Requirements (must not break, not even during a rabbit hole):
1. The dashboard is a mandatory part of the project. For EVERY model variant it must
   truthfully show: (a) the model's real atomic shape — the actual architecture
   geometry, not decoration; (b) the key facts (params, depth, scales, tied/untied,
   experts, γ/τ ladder …); (c) where the data flows through the model; (d) the live
   process; and (e) a visualization of the model's REAL learning (real gradients /
   real loss of an actual training run — never a stand-in or a canned replay).
2. Metrics / telemetry must NOT meaningfully slow down the real training. Instrumentation
   is throttled (sampled every N steps), avoids extra forward/backward passes, and stays
   off the hot path. If a metric would cost real training throughput, it is not worth it.
3. Prove-in-small. Every capability must be demonstrated on modest hardware (a single
   laptop-class GPU / CPU) before allocating the cluster. Nothing is "real" until shown
   in small.
4. Persistent in-weights memory is the model's identity. The self-modifying fast-weight
   associative memory (W ← γW + β(v−Wk)kᵀ), O(1) state, no KV cache, that survives a
   restart — is the distinctive core. Do not trade it away to chase SOTA components.
5. Naturally fractal, one rule. Weight-tied recursion over depth plus a ladder of
   timescales is the default identity (untie is an option, not the identity). Any growth
   must PRESERVE function (mitosis / Net2Net), never shock the network (empty-add
   neurogenesis is refuted).
6. Runs on modest, laptop-class hardware (e.g. a 4GB laptop GPU such as the RTX A2000).
   No component that only works at cloud scale.
7. A usable assistant, not a benchmark toy — a "smart caveman": understands natural
   English, does what is asked, uses tools, knows facts and principles; caveman-style
   replies are fine. Utility over polish.
8. Learns during operation and remembers across a restart. The assistant acquires facts
   at inference time and recalls them after a process restart (persistence verified),
   beating a frozen baseline.
9. Honest, falsifiable evaluation. Every capability claim has a cheap check with a
   falsification criterion. Negative results are reported honestly (e.g. neurogenesis was
   refuted). No self-deception, no silent truncation of coverage.
10. Recall must generalize to unseen words / facts, not just memorize the training set.
    Data pipelines are non-destructive (never overwrite the tokenizer or existing data
    in place).
11. Prove the real design in small — never design *for* small. A small-scale run VALIDATES
    the scale-invariant target; it is not a license to reshape the target into whatever a
    tiny model can do. The only compromises allowed are the ones required to RUN or MEASURE
    on modest hardware (fewer params, shorter / single-GPU runs, smaller batches, cheaper
    eval proxies, in-process checks). NOT allowed: changing the architecture, objective, data
    strategy, or method into something that merely "works small" but abandons the target —
    no crutches, no forcing what is meant to emerge, no keeping a component only because
    emergence needs scale. A capability that provably needs scale to appear is an honest
    scale-gated result to REPORT (VIBE #9), with the design left intact — not a reason to
    swap in a small-scale substitute and call it solved. "It won't emerge at this size" means
    "report it as scale-gated," not "redesign so it does."