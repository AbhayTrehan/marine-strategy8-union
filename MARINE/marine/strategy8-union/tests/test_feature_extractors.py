"""
Run with: python3 tests/test_feature_extractors.py

We have no network access to download `google/owlvit-base-patch32` or
`openai/clip-vit-base-patch32` in this environment (and no GPU), so
OwlViTScorer/ClipScorer/FeatureExtractor themselves (which load real
pretrained weights) cannot be instantiated or exercised here -- they will
work in the user's actual environment, which already successfully runs
this codebase's other HuggingFace-hosted models (DETR, RAM++).

What we CAN and DO test here, with real torch tensors (not mocks): the
pure post-processing math factored out into `owlvit_postprocess` and
`clip_cosine_similarities`, which is where an off-by-one or wrong-axis bug
would actually hide. We construct raw logits/boxes/embedding tensors by
hand (the exact shapes/contracts those two HuggingFace models guarantee)
and check the computation against an independent manual reference.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from feature_extractors import owlvit_postprocess, clip_cosine_similarities


def test_owlvit_postprocess_picks_max_confidence_box_per_query():
    # 4 candidate boxes, 2 text queries ("dog", "cat")
    # box layout: (cx, cy, w, h)
    boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],   # box 0
        [0.1, 0.1, 0.05, 0.05],  # box 1 (tiny)
        [0.6, 0.4, 0.4, 0.5],   # box 2 (the one that should win for "dog")
        [0.2, 0.8, 0.1, 0.1],   # box 3 (the one that should win for "cat")
    ])
    # raw logits BEFORE sigmoid; query 0 = "dog", query 1 = "cat"
    logits = torch.tensor([
        [1.0, -2.0],
        [-5.0, -5.0],
        [4.0, 0.0],   # box 2 highest for query 0 ("dog")
        [-1.0, 3.0],  # box 3 highest for query 1 ("cat")
    ])

    results = owlvit_postprocess(logits, boxes)
    assert len(results) == 2

    s_det_dog, s_area_dog = results[0]
    s_det_cat, s_area_cat = results[1]

    expected_det_dog = torch.sigmoid(torch.tensor(4.0)).item()
    expected_det_cat = torch.sigmoid(torch.tensor(3.0)).item()
    assert abs(s_det_dog - expected_det_dog) < 1e-6
    assert abs(s_det_cat - expected_det_cat) < 1e-6

    # area should come from the WINNING box for each query, not box 0
    assert abs(s_area_dog - (0.4 * 0.5)) < 1e-6   # box 2's w*h
    assert abs(s_area_cat - (0.1 * 0.1)) < 1e-6   # box 3's w*h
    print("test_owlvit_postprocess_picks_max_confidence_box_per_query OK")


def test_owlvit_postprocess_single_query():
    boxes = torch.tensor([[0.5, 0.5, 0.3, 0.2], [0.1, 0.1, 0.9, 0.9]])
    logits = torch.tensor([[2.0], [-1.0]])
    results = owlvit_postprocess(logits, boxes)
    assert len(results) == 1
    s_det, s_area = results[0]
    assert abs(s_det - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-6
    assert abs(s_area - 0.06) < 1e-6
    print("test_owlvit_postprocess_single_query OK")


def test_owlvit_postprocess_clips_negative_box_dims():
    # defensive: malformed/negative w or h (shouldn't happen from a real
    # model, but area must never go negative if it ever does)
    boxes = torch.tensor([[0.5, 0.5, -0.1, 0.2]])
    logits = torch.tensor([[1.0]])
    s_det, s_area = owlvit_postprocess(logits, boxes)[0]
    assert s_area == 0.0
    print("test_owlvit_postprocess_clips_negative_box_dims OK")


def test_clip_cosine_similarities_basic():
    # 3 unit text vectors in 4D, one image vector
    image_emb = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    text_embs = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],     # identical -> cos sim 1.0
        [0.0, 1.0, 0.0, 0.0],     # orthogonal -> cos sim 0.0
        [-1.0, 0.0, 0.0, 0.0],    # opposite -> cos sim -1.0
    ])
    sims = clip_cosine_similarities(image_emb, text_embs)
    assert len(sims) == 3
    assert abs(sims[0] - 1.0) < 1e-6
    assert abs(sims[1] - 0.0) < 1e-6
    assert abs(sims[2] - (-1.0)) < 1e-6
    print("test_clip_cosine_similarities_basic OK")


def test_clip_cosine_similarities_matches_manual_dot_product():
    torch.manual_seed(0)
    image_emb = torch.randn(1, 16)
    image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
    text_embs = torch.randn(5, 16)
    text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

    sims = clip_cosine_similarities(image_emb, text_embs)
    for i in range(5):
        expected = float((image_emb[0] * text_embs[i]).sum().item())
        assert abs(sims[i] - expected) < 1e-5
    print("test_clip_cosine_similarities_matches_manual_dot_product OK")


def test_empty_object_list_returns_empty():
    boxes = torch.zeros(0, 4)
    logits = torch.zeros(0, 0)
    assert owlvit_postprocess(logits, boxes) == []
    print("test_empty_object_list_returns_empty OK")


def test_classes_import_without_network_or_gpu():
    # importing the module (and the class *definitions*) must not require
    # network access -- only *instantiating* OwlViTScorer/ClipScorer (which
    # downloads weights) does. This guards against an accidental
    # module-level `from_pretrained(...)` call.
    import feature_extractors  # noqa: F401

    assert hasattr(feature_extractors, "OwlViTScorer")
    assert hasattr(feature_extractors, "ClipScorer")
    assert hasattr(feature_extractors, "FeatureExtractor")
    print("test_classes_import_without_network_or_gpu OK")


if __name__ == "__main__":
    test_owlvit_postprocess_picks_max_confidence_box_per_query()
    test_owlvit_postprocess_single_query()
    test_owlvit_postprocess_clips_negative_box_dims()
    test_clip_cosine_similarities_basic()
    test_clip_cosine_similarities_matches_manual_dot_product()
    test_empty_object_list_returns_empty()
    test_classes_import_without_network_or_gpu()
    print("\nALL feature_extractors.py TESTS PASSED")
