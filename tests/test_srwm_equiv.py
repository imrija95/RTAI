"""Verify that chunk-parallel SRWM matches the recurrent implementation."""

import torch

from rtai.model import GPTConfig, RTAIModel


def test_srwm_chunk_matches_recurrent() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPTConfig(vocab_size=66, block_size=48, n_layer=3, n_head=2, n_embd=64,
                    srwm_mode="recurrent", chunk_size=16)
    model = RTAIModel(cfg).to(device).eval()
    batch, tokens = 2, 40
    x = torch.randint(0, cfg.vocab_size, (batch, tokens), device=device)
    initial = model.init_states(batch, device)

    with torch.no_grad():
        for block in model.blocks:
            block.srwm.mode = "recurrent"
        recurrent_logits, _, recurrent_w, _ = model(
            x, states=[state.clone() for state in initial])
        for block in model.blocks:
            block.srwm.mode = "chunk"
        chunk_logits, _, chunk_w, _ = model(x, states=[state.clone() for state in initial])

    logits_delta = (recurrent_logits - chunk_logits).abs().max().item()
    state_delta = max((left - right).abs().max().item()
                      for left, right in zip(recurrent_w, chunk_w))
    assert logits_delta < 1e-4
    assert state_delta < 1e-4


if __name__ == "__main__":
    test_srwm_chunk_matches_recurrent()
    print("OK - recurrent and chunk SRWM paths match")
