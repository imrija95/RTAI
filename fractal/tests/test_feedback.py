"""Predictive Event Algebra state, rating, and persistence tests."""

from __future__ import annotations

import torch

from fractal import feedback, persist
from fractal.agent import _feed
from fractal.model import Config, FractalLM
from fractal.unit import FractalState


def _model():
    torch.manual_seed(7)
    cfg = Config(vocab_size=48, n_embd=24, n_head=4, depth=2, n_scales=2,
                 chunk_size=8, event_algebra=True, eligibility_decay=0.9)
    return FractalLM(cfg).eval()


def test_eligibility_is_stream_chunking_invariant():
    model = _model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 11))
    one = model.init_states(1, "cpu")
    tokenwise = model.init_states(1, "cpu")
    with torch.no_grad():
        _, one = model.forward_stream(ids, one)
        for pos in range(ids.shape[1]):
            _, tokenwise = model.forward_stream(ids[:, pos:pos + 1], tokenwise)
    for a, b in zip(one, tokenwise):
        for wa, wb in zip(a.W, b.W):
            torch.testing.assert_close(wa, wb, atol=2e-5, rtol=2e-5)
        for ea, eb in zip(a.eligibility, b.eligibility):
            torch.testing.assert_close(ea, eb, atol=2e-5, rtol=2e-5)


def test_rating_updates_session_and_fresh_session_weights():
    model = _model()
    live = model.init_states(1, "cpu")
    evidence = feedback.message_eligibility(model, [3, 5, 8, 13], "cpu")
    before_w = [[w.clone() for w in state.W] for state in live]
    before_w0 = feedback.w0_snapshot(model)
    result = feedback.apply_rating(model, live, evidence, 5)
    assert result.credit == 1.0
    assert result.fast_update_norm > 0
    assert result.w0_update_norm > 0
    assert any(not torch.equal(old, new) for row, state in zip(before_w, live)
               for old, new in zip(row, state.W))
    assert any(not torch.equal(old, cell.W0) for row, block in zip(before_w0, model.blocks)
               for old, cell in zip(row, block.unit.cells))


def test_external_span_receives_autonomous_credit_but_keeps_trace_identity():
    model = _model()
    prior = model.init_states(1, "cpu")
    _, prior = model.forward_stream(torch.tensor([[2, 4, 6]]), prior)
    before = [state.clone() for state in prior]
    ids = [8, 10, 12, 14]
    logits, after = model.forward_stream(torch.tensor([ids]), prior)
    evidence = feedback.recent_evidence(before, after, len(ids), model.cfg.eligibility_decay)
    factor = model.cfg.eligibility_decay ** len(ids)
    for old, new, recent in zip(before, after, evidence):
        for e0, e1, er in zip(old.eligibility, new.eligibility, recent.eligibility):
            torch.testing.assert_close(e1, factor * e0 + er)
    assert 0.0 <= feedback.observed_surprise(logits, ids) <= 1.0

    plain = model.init_states(1, "cpu")
    credited = model.init_states(1, "cpu")
    _, plain = _feed(model, plain, ids, "cpu", autonomous_evidence=False)
    _, credited = _feed(model, credited, ids, "cpu", autonomous_evidence=True)
    assert any(not torch.equal(a, b) for sa, sb in zip(plain, credited)
               for a, b in zip(sa.W, sb.W))
    assert model._last_autonomous_credit["tokens"] == len(ids)


def test_disabled_event_algebra_does_not_copy_state_for_autonomous_evidence(monkeypatch):
    model = FractalLM(Config(
        vocab_size=48, n_embd=24, n_head=4, depth=2, n_scales=2, chunk_size=8,
        event_algebra=False,
    )).eval()
    states = model.init_states(1, "cpu")

    def fail_clone(_state):
        raise AssertionError("disabled Event Algebra must not clone fast-weight state")

    monkeypatch.setattr(FractalState, "clone", fail_clone)
    _feed(model, states, [2, 4, 6], "cpu", autonomous_evidence=True)


def test_eligibility_and_w0_overlay_round_trip(tmp_path):
    model = _model()
    states = feedback.message_eligibility(model, [1, 2, 3, 4], "cpu")
    state_path = tmp_path / "state.pt"
    persist.save_states(state_path, states)
    loaded = persist.load_states(state_path, "cpu")
    for a, b in zip(states, loaded):
        for ea, eb in zip(a.eligibility, b.eligibility):
            torch.testing.assert_close(ea, eb)

    overlay_path = tmp_path / "feedback-w0.pt"
    feedback.consolidate_w0(model, states, credit=1.0)
    expected = feedback.w0_snapshot(model)
    feedback.save_w0(overlay_path, model)
    restored = _model()
    assert feedback.load_w0(overlay_path, restored)
    for row, block in zip(expected, restored.blocks):
        for value, cell in zip(row, block.unit.cells):
            torch.testing.assert_close(value, cell.W0)


def test_feedback_queue_and_exactly_once_state(tmp_path):
    queue = tmp_path / "feedback.jsonl"
    event = {"event_id": "evt-1", "content": "A fact", "credit_delta": 1.0}
    feedback.append_event(queue, event)
    feedback.append_event(queue, {"event_id": "evt-2", "content": "A correction",
                                  "credit_delta": -1.0})
    assert [item["event_id"] for item in feedback.read_events(queue)] == ["evt-1", "evt-2"]

    model = _model()
    evidence = feedback.message_eligibility(model, [7, 9, 11], "cpu")
    feedback.consolidate_w0(model, evidence, 1.0)
    durable = tmp_path / "consolidation.pt"
    feedback.save_consolidation_state(durable, model, {"evt-1"})
    expected = feedback.w0_snapshot(model)
    restored = _model()
    assert feedback.load_consolidation_state(durable, restored) == {"evt-1"}
    for row, block in zip(expected, restored.blocks):
        for value, cell in zip(row, block.unit.cells):
            torch.testing.assert_close(value, cell.W0)


def test_vector_teaching_reduces_loss_using_only_w0():
    model = _model()
    slow_before = model.tok_emb.weight.detach().clone()
    w0_before = feedback.w0_snapshot(model)
    result = feedback.teach_w0(
        model, [2, 7, 11, 5], 17, "cpu", lr=0.1, steps=4, max_fraction=0.25,
        anchor_prompts=[[3, 4], [8, 9], [12, 13, 14]], anchor_weight=0.1)
    assert result.update_norm > 0
    assert result.final_loss < result.initial_loss
    assert model.cfg.event_algebra
    assert all(block.unit.event_algebra for block in model.blocks)
    torch.testing.assert_close(model.tok_emb.weight, slow_before)
    assert any(not torch.equal(old, cell.W0) for row, block in zip(w0_before, model.blocks)
               for old, cell in zip(row, block.unit.cells))
