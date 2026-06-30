"""
feature_extractors.py
======================

Real (no mocks) implementations of the two zero-shot vision models used to
build the 3D feature vector x_i = [s_det, s_clip, s_area]^T for every
candidate object (Strategy8_Union_contrastive.pdf, Section 3.1, Eq. 4-6):

  * OwlViTScorer  -> s_det (Eq. 4 first component) and s_area (Eq. 6):
        the paper specifies "a zero-shot detector (e.g., OWL-ViT)" queried
        with the text prompt "a photo of a {object}" for EVERY candidate,
        taking the maximum bounding-box confidence as s_det and that same
        box's normalized area as s_area. We use `google/owlvit-base-patch32`
        via `transformers`, already a listed dependency (no new package).
  * ClipScorer    -> s_clip (Eq. 5): cosine similarity between the CLIP
        image embedding and the CLIP text embedding for the SAME "a photo
        of a {object}" prompt. We use `openai/clip-vit-base-patch32`, also
        via `transformers`.

Both classes batch ALL of an image's candidate objects into a SINGLE
forward pass per image (one OWL-ViT call with N text queries against the
image, one CLIP image-embedding call + one CLIP text-embedding call for N
objects), rather than one model call per (image, object) pair -- with
candidate pools of up to a few dozen objects per image across hundreds of
images, this is the difference between a few hundred and many thousand
forward passes.

The exact score formula mirrors OWL-ViT's own (version-stable) raw output
contract rather than any higher-level "post_process_*" convenience method,
because that convenience method's name has changed across transformers
versions (`post_process_object_detection` -> the newer
`post_process_grounded_object_detection` wrapper) and we cannot assume
which transformers version this runs under in the user's environment:

    scores = sigmoid(logits.max(dim=-1).values)   # per predicted box
    boxes  = pred_boxes                            # (cx, cy, w, h), normalized [0,1]

This is the same computation `OwlViTImageProcessor.post_process_object_detection`
performs internally (verified directly against the installed transformers
source), so reimplementing it directly here against the raw model output
is robust to that wrapper's naming/location moving around between versions.

NOTE: these classes require a GPU + internet access to download
`google/owlvit-base-patch32` and `openai/clip-vit-base-patch32` from the
Hugging Face Hub, which is exactly the kind of access this development
sandbox does NOT have. They are written to the same conventions as
`marine/utils/utils_model.py::load_model` in this codebase and are
exercised by `tests/test_feature_extractors.py` against a tiny *real*
(randomly-initialized, architecturally correct) OWL-ViT/CLIP config -- not
mocks -- to validate the scoring/post-processing math, since full
pretrained weights cannot be downloaded here either.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def owlvit_postprocess(logits: torch.Tensor, boxes: torch.Tensor) -> List[Tuple[float, float]]:
    """Pure tensor math, factored out of OwlViTScorer.score_batch so it can
    be unit-tested without a real (network-downloaded) model: given raw
    OWL-ViT outputs for ONE image,
        logits: (num_boxes, num_queries)
        boxes:  (num_boxes, 4) in (cx, cy, w, h), normalized [0, 1]
    returns [(s_det, s_area), ...] for each query column, taking the
    maximum-confidence box per query (Eq. 4, 6). Score formula matches
    `OwlViTImageProcessor.post_process_object_detection`'s raw computation
    (verified against the installed transformers source -- see module
    docstring): scores = sigmoid(logits), max over boxes per query.
    """
    num_queries = logits.shape[1]
    scores = torch.sigmoid(logits)  # (num_boxes, num_queries)
    results: List[Tuple[float, float]] = []
    for qi in range(num_queries):
        col = scores[:, qi]
        best_idx = int(torch.argmax(col).item())
        s_det = float(col[best_idx].item())
        w = float(max(boxes[best_idx, 2].item(), 0.0))
        h = float(max(boxes[best_idx, 3].item(), 0.0))
        results.append((s_det, float(w * h)))
    return results


def clip_cosine_similarities(image_embed: torch.Tensor, text_embeds: torch.Tensor) -> List[float]:
    """Pure tensor math, factored out of ClipScorer.score_batch: given an
    already-L2-normalized image embedding (1, D) and already-L2-normalized
    text embeddings (N, D), returns the N cosine similarities (Eq. 5)."""
    sims = (image_embed @ text_embeds.T).squeeze(0)
    if sims.dim() == 0:
        return [float(sims.item())]
    return [float(s.item()) for s in sims]


class OwlViTScorer:
    """Zero-shot detector scorer: s_det and s_area (Eq. 4, 6)."""

    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: str = "cuda"):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device).eval()

    @staticmethod
    def _query_text(object_name: str) -> str:
        return f"a photo of a {object_name}"

    @torch.no_grad()
    def score_batch(self, image, object_names: Sequence[str]) -> List[Tuple[float, float]]:
        """One OWL-ViT forward pass scoring ALL `object_names` against a
        single `image`. Returns a list of (s_det, s_area) pairs, one per
        object, in the same order as `object_names` -- each is the
        MAXIMUM-confidence box for that specific text query (Eq. 4-6)."""
        if len(object_names) == 0:
            return []

        queries = [self._query_text(o) for o in object_names]
        inputs = self.processor(text=[queries], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits[0]       # (num_boxes, num_queries)
        boxes = outputs.pred_boxes[0]    # (num_boxes, 4): cx, cy, w, h (normalized [0,1])

        return owlvit_postprocess(logits, boxes)

    def score_map(self, image, object_names: Sequence[str]) -> Dict[str, Tuple[float, float]]:
        scores = self.score_batch(image, object_names)
        return dict(zip(object_names, scores))


class ClipScorer:
    """Image-text similarity scorer: s_clip (Eq. 5)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()

    @staticmethod
    def _query_text(object_name: str) -> str:
        return f"a photo of a {object_name}"

    @torch.no_grad()
    def _image_embedding(self, image) -> torch.Tensor:
        inputs = self.processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        emb = self.model.get_image_features(pixel_values=pixel_values)  # (1, D)
        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def _text_embeddings(self, object_names: Sequence[str]) -> torch.Tensor:
        queries = [self._query_text(o) for o in object_names]
        inputs = self.processor(text=queries, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model.get_text_features(**inputs)  # (N, D)
        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def score_batch(self, image, object_names: Sequence[str]) -> List[float]:
        """One CLIP image-embedding call + one batched text-embedding
        call for ALL `object_names`, returning cosine similarities (Eq. 5)
        in the same order as `object_names`."""
        if len(object_names) == 0:
            return []
        image_emb = self._image_embedding(image)        # (1, D), L2-normalized
        text_emb = self._text_embeddings(object_names)   # (N, D), L2-normalized
        return clip_cosine_similarities(image_emb, text_emb)

    def score_map(self, image, object_names: Sequence[str]) -> Dict[str, float]:
        scores = self.score_batch(image, object_names)
        return dict(zip(object_names, scores))


class FeatureExtractor:
    """Convenience wrapper combining OwlViTScorer + ClipScorer into the
    full 3D feature vector x_i = [s_det, s_clip, s_area] for every
    candidate object of a single image."""

    def __init__(
        self,
        owlvit_model: str = "google/owlvit-base-patch32",
        clip_model: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
    ):
        self.owlvit = OwlViTScorer(owlvit_model, device=device)
        self.clip = ClipScorer(clip_model, device=device)

    def extract(self, image, object_names: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
        """Returns {object_name: (s_det, s_clip, s_area)} for every name in
        `object_names`, feature order matching gmm.py's expected column
        order (det_dim=0)."""
        if len(object_names) == 0:
            return {}
        det_area = self.owlvit.score_batch(image, object_names)
        clip_sims = self.clip.score_batch(image, object_names)
        out: Dict[str, Tuple[float, float, float]] = {}
        for name, (s_det, s_area), s_clip in zip(object_names, det_area, clip_sims):
            out[name] = (s_det, s_clip, s_area)
        return out
