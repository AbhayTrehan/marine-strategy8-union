"""
Run with: python3 tests/test_tristate_logits.py

We don't have GPU/internet access to download real LVLM weights in this
environment, so this test validates the *mechanics* of
TriStateGuidanceLogits (incremental KV-cached forward passes across two
independent guidance branches, attention-mask extension, and the Eq. 20
blending formula) against a small but REAL causal self-attention model
with an actual KV cache -- not a mock. The key property under test is:
"feeding the model 1 new token + cached past_key_values produces IDENTICAL
output to feeding the model the entire sequence from scratch" -- which is
exactly what the original codebase's GuidanceLogits class also relies on,
and exactly the kind of off-by-one/cache bug that's easy to introduce
silently when hand-rolling incremental decoding.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from tristate_logits import TriStateGuidanceLogits


class _ModelOutput:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class ToyCausalLM(nn.Module):
    """A tiny, real, single-head causal self-attention LM with explicit KV
    caching and an "image" conditioning token, mimicking the call signature
    our LogitsProcessor uses against the real LLaVA model:
        model(input_ids=..., pixel_values=..., attention_mask=...,
              use_cache=True, past_key_values=...)
    On the FIRST call for a branch (past_key_values is None), pixel_values
    must be provided and is embedded as a prefix "image token". On
    subsequent calls, only the new token(s) are passed along with the
    existing past_key_values (no pixel_values needed, exactly matching how
    LLaVA-style models are used in this codebase).
    """

    def __init__(self, vocab_size=23, hidden=16, image_feat_dim=5, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, hidden)
        self.image_proj = nn.Linear(image_feat_dim, hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, vocab_size)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)

    @torch.no_grad()
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None,
                use_cache=True, past_key_values=None):
        tok_emb = self.embed(input_ids)  # (B, T_new, H)

        if past_key_values is None:
            assert pixel_values is not None, "first call for a branch must supply pixel_values"
            img_tok = self.image_proj(pixel_values).unsqueeze(1)  # (B, 1, H)
            x = torch.cat([img_tok, tok_emb], dim=1)
            past_k = past_v = None
        else:
            x = tok_emb
            past_k, past_v = past_key_values

        k_new = self.k_proj(x)
        v_new = self.v_proj(x)
        if past_k is not None:
            k_all = torch.cat([past_k, k_new], dim=1)
            v_all = torch.cat([past_v, v_new], dim=1)
        else:
            k_all, v_all = k_new, v_new

        q_new = self.q_proj(x)

        T_new = x.shape[1]
        T_total = k_all.shape[1]
        scores = torch.einsum("bth,bsh->bts", q_new, k_all) / (self.hidden ** 0.5)

        mask = torch.zeros(T_new, T_total, dtype=torch.bool)
        for t in range(T_new):
            global_t = T_total - T_new + t
            if global_t + 1 < T_total:
                mask[t, global_t + 1:] = True
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bts,bsh->bth", attn, v_all)
        logits = self.out_proj(ctx)

        return _ModelOutput(logits=logits, past_key_values=(k_all, v_all))


def _run_from_scratch(model: ToyCausalLM, full_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
    """Reference computation: one big forward pass over the WHOLE sequence
    (no cache reuse at all), returning log-softmax'd logits for the final
    position. This is the ground truth the incremental path must match."""
    out = model(input_ids=full_ids, pixel_values=pixel_values, attention_mask=torch.ones_like(full_ids), past_key_values=None)
    return F.log_softmax(out.logits[:, -1:, :], dim=-1).squeeze(1)


def test_incremental_branch_matches_from_scratch_recompute():
    torch.manual_seed(0)
    model = ToyCausalLM(vocab_size=23, hidden=16, seed=0)
    B = 2
    pixel_values = torch.randn(B, 5)

    guidance_ids = torch.randint(0, 23, (B, 4))
    guidance_mask = torch.ones(B, 4, dtype=torch.long)

    processor = TriStateGuidanceLogits(
        alpha=0.5,
        guidance_pos_ids=guidance_ids,
        guidance_neg_ids=guidance_ids.clone(),  # same prompt for this isolated branch test
        images_pos=pixel_values,
        images_neg=pixel_values,
        guidance_pos_attention_mask=guidance_mask,
        guidance_neg_attention_mask=guidance_mask.clone(),
        model=model,
    )

    # Simulate 5 decoding steps. The FIRST call to a branch "primes" it
    # (processes guidance_ids alone -- this mirrors decode step 0, where
    # HF's generate() loop has not generated anything yet either, exactly
    # like the original GuidanceLogits class's `if self.out is None`
    # branch). Each call AFTER that consumes the token generated at the
    # PREVIOUS step to extend the cache by one and predicts the next one.
    generated = torch.randint(0, 23, (B, 5))

    # step 0 (priming): new_token is irrelevant/unused here, ground truth
    # is a from-scratch pass over guidance_ids alone.
    z0 = processor._step_branch("pos", generated[:, 0:1])
    ref0 = _run_from_scratch(model, guidance_ids, pixel_values)
    assert torch.allclose(z0, ref0, atol=1e-5), (z0, ref0)

    for t in range(1, 5):
        new_token = generated[:, t - 1:t]  # token produced at the previous step
        z_pos_incremental = processor._step_branch("pos", new_token)

        full_ids = torch.cat([guidance_ids, generated[:, :t]], dim=1)
        z_pos_scratch = _run_from_scratch(model, full_ids, pixel_values)

        assert torch.allclose(z_pos_incremental, z_pos_scratch, atol=1e-5), (
            f"step {t}: incremental and from-scratch logits diverge\n"
            f"{z_pos_incremental}\nvs\n{z_pos_scratch}"
        )
    print("test_incremental_branch_matches_from_scratch_recompute OK")


def test_two_branches_are_independent():
    torch.manual_seed(1)
    model = ToyCausalLM(vocab_size=23, hidden=16, seed=1)
    B = 1
    pixel_values = torch.randn(B, 5)

    pos_ids = torch.randint(0, 23, (B, 3))
    neg_ids = torch.randint(0, 23, (B, 6))  # deliberately different length
    pos_mask = torch.ones(B, 3, dtype=torch.long)
    neg_mask = torch.ones(B, 6, dtype=torch.long)

    processor = TriStateGuidanceLogits(
        alpha=0.4,
        guidance_pos_ids=pos_ids,
        guidance_neg_ids=neg_ids,
        images_pos=pixel_values,
        images_neg=pixel_values,
        guidance_pos_attention_mask=pos_mask,
        guidance_neg_attention_mask=neg_mask,
        model=model,
    )

    generated = torch.randint(0, 23, (B, 4))

    # priming step
    z_pos = processor._step_branch("pos", generated[:, 0:1])
    z_neg = processor._step_branch("neg", generated[:, 0:1])
    ref_pos = _run_from_scratch(model, pos_ids, pixel_values)
    ref_neg = _run_from_scratch(model, neg_ids, pixel_values)
    assert torch.allclose(z_pos, ref_pos, atol=1e-5)
    assert torch.allclose(z_neg, ref_neg, atol=1e-5)
    assert not torch.allclose(z_pos, z_neg, atol=1e-3)

    for t in range(1, 4):
        new_token = generated[:, t - 1:t]
        z_pos = processor._step_branch("pos", new_token)
        z_neg = processor._step_branch("neg", new_token)

        full_pos = torch.cat([pos_ids, generated[:, :t]], dim=1)
        full_neg = torch.cat([neg_ids, generated[:, :t]], dim=1)
        ref_pos = _run_from_scratch(model, full_pos, pixel_values)
        ref_neg = _run_from_scratch(model, full_neg, pixel_values)

        assert torch.allclose(z_pos, ref_pos, atol=1e-5)
        assert torch.allclose(z_neg, ref_neg, atol=1e-5)
        # the two branches should NOT be accidentally sharing state/cache
        assert not torch.allclose(z_pos, z_neg, atol=1e-3)
    print("test_two_branches_are_independent OK")


def test_eq20_blending_formula():
    """Directly checks z_final = z_ung + alpha*(z_pos - z_neg),
    then re-normalized via log_softmax, against a manual computation from
    captured branch outputs -- exercised across TWO sequential decode
    steps through the full __call__ path (not just _step_branch), so the
    `new_token = input_ids[:, -1:]` extraction logic is covered too."""
    torch.manual_seed(2)
    model = ToyCausalLM(vocab_size=23, hidden=16, seed=2)
    B = 1
    pixel_values = torch.randn(B, 5)

    pos_ids = torch.randint(0, 23, (B, 3))
    neg_ids = torch.randint(0, 23, (B, 3))
    mask = torch.ones(B, 3, dtype=torch.long)
    alpha = 0.7

    processor = TriStateGuidanceLogits(
        alpha=alpha,
        guidance_pos_ids=pos_ids,
        guidance_neg_ids=neg_ids,
        images_pos=pixel_values,
        images_neg=pixel_values,
        guidance_pos_attention_mask=mask.clone(),
        guidance_neg_attention_mask=mask.clone(),
        model=model,
    )

    # --- decode step 0 (priming): input_ids is just the original unconditioned
    # prompt, generate() has not appended anything yet.
    main_ids_step0 = torch.randint(0, 23, (B, 5))  # stand-in c_ung prompt tokens
    raw_logits_step0 = torch.randn(B, 23) * 3.0

    z_final_0 = processor(main_ids_step0, raw_logits_step0)

    z_ung_0_expected = F.log_softmax(raw_logits_step0, dim=-1)
    z_pos_0_expected = _run_from_scratch(model, pos_ids, pixel_values)
    z_neg_0_expected = _run_from_scratch(model, neg_ids, pixel_values)
    z_final_0_expected = F.log_softmax(
        z_ung_0_expected + alpha * (z_pos_0_expected - z_neg_0_expected), dim=-1
    )
    assert torch.allclose(z_final_0, z_final_0_expected, atol=1e-5), (z_final_0, z_final_0_expected)
    assert torch.allclose(z_final_0.exp().sum(dim=-1), torch.ones(B), atol=1e-4)

    # --- decode step 1: generate() has now appended ONE new token (sampled
    # from step 0's distribution) to the running sequence.
    next_token = torch.randint(0, 23, (B, 1))
    main_ids_step1 = torch.cat([main_ids_step0, next_token], dim=1)
    raw_logits_step1 = torch.randn(B, 23) * 3.0

    z_final_1 = processor(main_ids_step1, raw_logits_step1)

    z_ung_1_expected = F.log_softmax(raw_logits_step1, dim=-1)
    full_pos_1 = torch.cat([pos_ids, next_token], dim=1)
    full_neg_1 = torch.cat([neg_ids, next_token], dim=1)
    z_pos_1_expected = _run_from_scratch(model, full_pos_1, pixel_values)
    z_neg_1_expected = _run_from_scratch(model, full_neg_1, pixel_values)
    z_final_1_expected = F.log_softmax(
        z_ung_1_expected + alpha * (z_pos_1_expected - z_neg_1_expected), dim=-1
    )
    assert torch.allclose(z_final_1, z_final_1_expected, atol=1e-5), (z_final_1, z_final_1_expected)
    assert torch.allclose(z_final_1.exp().sum(dim=-1), torch.ones(B), atol=1e-4)
    print("test_eq20_blending_formula OK")


def test_alpha_zero_recovers_unconditioned():
    torch.manual_seed(3)
    model = ToyCausalLM(vocab_size=23, hidden=16, seed=3)
    B = 1
    pixel_values = torch.randn(B, 5)
    pos_ids = torch.randint(0, 23, (B, 3))
    neg_ids = torch.randint(0, 23, (B, 3))
    mask = torch.ones(B, 3, dtype=torch.long)

    processor = TriStateGuidanceLogits(
        alpha=0.0,
        guidance_pos_ids=pos_ids,
        guidance_neg_ids=neg_ids,
        images_pos=pixel_values,
        images_neg=pixel_values,
        guidance_pos_attention_mask=mask.clone(),
        guidance_neg_attention_mask=mask.clone(),
        model=model,
    )
    main_ids = torch.randint(0, 23, (B, 4))
    raw_logits = torch.randn(B, 23) * 2.0
    z_final = processor(main_ids, raw_logits)
    expected = F.log_softmax(F.log_softmax(raw_logits, dim=-1), dim=-1)
    assert torch.allclose(z_final, expected, atol=1e-5)
    print("test_alpha_zero_recovers_unconditioned OK")


def test_invalid_alpha_raises():
    model = ToyCausalLM(vocab_size=5, hidden=4)
    pos_ids = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.long)
    for bad_alpha in [-0.1, 1.1]:
        try:
            TriStateGuidanceLogits(
                alpha=bad_alpha,
                guidance_pos_ids=pos_ids,
                guidance_neg_ids=pos_ids.clone(),
                images_pos=torch.randn(1, 5),
                images_neg=torch.randn(1, 5),
                guidance_pos_attention_mask=mask,
                guidance_neg_attention_mask=mask.clone(),
                model=model,
            )
            raise AssertionError(f"should have raised for alpha={bad_alpha}")
        except ValueError:
            pass
    print("test_invalid_alpha_raises OK")


if __name__ == "__main__":
    test_incremental_branch_matches_from_scratch_recompute()
    test_two_branches_are_independent()
    test_eq20_blending_formula()
    test_alpha_zero_recovers_unconditioned()
    test_invalid_alpha_raises()
    print("\nALL tristate_logits.py TESTS PASSED")
