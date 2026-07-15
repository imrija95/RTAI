"""fractal — a self-unrolling model with a fractal core.

The model is not a stored matrix but a GENERATING RULE: a single operator (delta-rule
weight self-modification) applied over a geometric ladder of time scales and also in depth
(weight-tied recursion). Model size = how far you unroll the rule.

Built from scratch in pure PyTorch, independent of the `rtai/` package.
"""
