"""
tristate_logits.py
===================

Implements Phase II "Tri-State Contrastive Decoding"
(Strategy8_Union_contrastive.pdf, Section 4). At each decoding step t:

    z_ung  = LogitsM(y_t | y_<t, c_ung,  I)      (Eq. 17)
    z_pos  = LogitsM(y_t | y_<t, c_pos,  I)      (Eq. 18)
    z_neg  = LogitsM(y_t | y_<t, c_neg,  I)      (Eq. 19)
    z_final = z_ung + alpha * (z_pos - z_neg)     (modified decoding)

NOTE: this is a modification of the original Eq. 20 from the spec
    (1 - alpha) * z_ung + alpha * (z_pos - z_neg)
The difference: the unguided branch is now kept at full weight (coefficient
1 instead of 1-alpha) regardless of alpha. This means alpha only controls
how strongly the positive/negative contrast is injected on top of the base
generation, rather than simultaneously downweighting the unguided branch.
The alpha range of interest shifts accordingly (0.6-0.8 is reasonable).

This is a genuinely new 3-branch decoder, NOT a drop-in extension of the
original MARINE codebase's `marine/utils/utils_guidance.py::GuidanceLogits`
(which only ever combines two branches: z_final = gamma*z_cond +
(1-gamma)*z_ung). Rather than retrofit that class, we implement this fresh
inside strategy8-union/ so the original file is left completely untouched
(per the "minimal changes to the original codebase" requirement) -- but we
deliberately mirror its overall structure and HF `LogitsProcessor`
integration pattern (one manual forward pass to "prime" each guidance
branch's KV cache, then a single incremental forward per decoding step
afterward) for consistency and because that pattern is already known to
work correctly with this codebase's LVLM loading code.

z_ung's source: exactly like the original `GuidanceLogits`, z_ung is NOT
computed inside this processor -- it is simply the `logits` argument HF's
own `generate()` loop already computed using `input_ids` (the primary
positional argument to `model.generate(...)`, which the caller must set up
to be the UNCONDITIONED prompt c_ung). This processor only has to run the
two ADDITIONAL forward passes for c_pos and c_neg.

Two correctness details that differ from (i.e. improve upon) the original
two-branch `GuidanceLogits`:
  1. We extend `attention_mask` by one column every step for both guidance
     branches, so that batched generation with differently-padded prompts
     does not silently attend across padding in the cached prefix. (The
     original class never passes `attention_mask` on incremental steps.)
  2. Logits are sliced as `out.logits[:, -1:, :]` uniformly for any batch
     size, rather than special-casing batch_size == 1.

Validated against a hand-rolled causal-attention toy model with a real KV
cache in `tests/test_tristate_logits.py`, which checks that the
incremental/cached computation this class performs is numerically
identical to a from-scratch (no-cache) recomputation of the same branch at
the same decoding step.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor


class TriStateGuidanceLogits(LogitsProcessor):
    def __init__(
        self,
        alpha: float,
        guidance_pos_ids: torch.Tensor,
        guidance_neg_ids: torch.Tensor,
        images_pos: torch.Tensor,
        images_neg: torch.Tensor,
        guidance_pos_attention_mask: torch.Tensor,
        guidance_neg_attention_mask: torch.Tensor,
        model,
        tokenizer=None,
    ):
        """
        Args:
            alpha: guidance strength alpha in Eq. 20.
            guidance_pos_ids: tokenized c_pos prompt, (B, T_pos).
            guidance_neg_ids: tokenized c_neg prompt, (B, T_neg).
            images_pos / images_neg: pixel_values used to prime the
                positive / negative branches' KV caches respectively.
                These come from the SAME source image (dataset.py computes
                one pixel_values tensor per text branch via the full
                multimodal processor, all numerically identical since
                pixel_values is a pure function of the image) -- kept as
                two separate arguments simply so this class never has to
                assume anything about how the caller obtained them.
            guidance_pos_attention_mask / guidance_neg_attention_mask:
                attention masks for the two guidance prompts, (B, T_pos)
                and (B, T_neg) respectively.
            model: the LVLM (already on the correct device).
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.guidance_pos_ids = guidance_pos_ids
        self.guidance_neg_ids = guidance_neg_ids
        self.images_pos = images_pos
        self.images_neg = images_neg
        self.guidance_pos_attention_mask = guidance_pos_attention_mask
        self.guidance_neg_attention_mask = guidance_neg_attention_mask
        self.model = model
        self.tokenizer = tokenizer

        self._pos_out = None
        self._neg_out = None
        self._pos_mask = guidance_pos_attention_mask
        self._neg_mask = guidance_neg_attention_mask

    def _step_branch(self, branch: str, new_token: torch.Tensor) -> torch.Tensor:
        """Advance one guidance branch ('pos' or 'neg') by one decoding
        step and return its log-softmax'd logits for the next token,
        shape (B, V). Maintains its own KV cache and extended attention
        mask across calls (primed on the first call, incremental after)."""
        out_attr = f"_{branch}_out"
        mask_attr = f"_{branch}_mask"
        ids_attr = f"guidance_{branch}_ids"
        images_attr = f"images_{branch}"

        out = getattr(self, out_attr)
        mask = getattr(self, mask_attr)

        if out is None:
            guidance_ids = getattr(self, ids_attr)
            images = getattr(self, images_attr)
            out = self.model(
                input_ids=guidance_ids,
                pixel_values=images,
                attention_mask=mask,
                use_cache=True,
            )
        else:
            mask = torch.cat(
                [mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)],
                dim=1,
            )
            out = self.model(
                input_ids=new_token,
                use_cache=True,
                attention_mask=mask,
                past_key_values=out.past_key_values,
            )

        setattr(self, out_attr, out)
        setattr(self, mask_attr, mask)

        step_logits = out.logits[:, -1:, :]  # (B, 1, V), uniform for any batch size
        return F.log_softmax(step_logits, dim=-1).squeeze(1)

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        z_ung = F.log_softmax(logits, dim=-1)
        new_token = input_ids[:, -1:]

        z_pos = self._step_branch("pos", new_token)
        z_neg = self._step_branch("neg", new_token)

        z_final = z_ung + self.alpha * (z_pos - z_neg)  # modified decoding: z_ung + alpha*(z_pos - z_neg)
        return F.log_softmax(z_final, dim=-1)
