"""Run with: python3 tests/test_v4_features.py

Validates the three v4 improvements end-to-end on realistic input:
  1. POS noun filter in VLM captions (text_objects.py)
  2. Physical entity filter for RAM++/VLM tags (synonyms.py)
  3. sqrt(s_area) + z-score normalization (fit_gmm.py FeatureScaler)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


# ---------------------------------------------------------------------------
# 1. POS noun filter
# ---------------------------------------------------------------------------
def test_pos_filter_removes_gerunds_and_verbs():
    from text_objects import extract_candidate_nouns
    # The exact screenshot caption
    caption = "A cat is sitting on a bed in a room with a television and a poster on the wall."
    nouns = extract_candidate_nouns(caption)
    assert "sitting" not in nouns, f"'sitting' (VBG) should be filtered: {nouns}"
    for expected in ["cat", "bed", "room", "television", "poster", "wall"]:
        assert expected in nouns, f"'{expected}' should be kept: {nouns}"
    print("test_pos_filter_removes_gerunds_and_verbs OK")


def test_pos_filter_keeps_double_word_compounds():
    from text_objects import extract_candidate_nouns
    caption = "There are several teddy bears and a traffic light near a cell phone."
    nouns = extract_candidate_nouns(caption)
    assert "teddy bear" in nouns, nouns
    assert "traffic light" in nouns, nouns
    assert "cell phone" in nouns, nouns
    assert "teddy" not in nouns and "bear" not in nouns
    print("test_pos_filter_keeps_double_word_compounds OK")


# ---------------------------------------------------------------------------
# 2. Physical entity filter
# ---------------------------------------------------------------------------
def test_physical_entity_filter_removes_noise_from_ram():
    from synonyms import UnionCanonicalizer, build_raw_mentions
    uc = UnionCanonicalizer()
    # Exact RAM++ tags from the screenshot
    ram = ["bed", "bedcover", "blanket", "bookshelf", "cat", "comfort",
           "floor", "lay", "relax", "sleep", "tabby", "television"]
    detr = ["remote", "chair", "bed", "tv", "cat"]
    vlm = ["cat", "bed", "room", "television", "poster", "wall"]

    raws = build_raw_mentions(ram, detr, vlm)
    cands = uc.canonicalize_pool(raws, filter_non_objects=True)
    canonicals = {c.canonical for c in cands}

    # should be gone
    for noise in ["comfort", "relax", "sleep", "lay"]:
        assert noise not in canonicals, f"'{noise}' should be filtered: {canonicals}"
    # should be kept (note: "television" maps to COCO canonical "tv")
    for obj in ["bed", "cat", "blanket", "tv", "floor", "room", "poster", "wall", "bookshelf"]:
        assert obj in canonicals, f"'{obj}' should be kept: {canonicals}"
    # tabby correctly merged into cat
    assert "tabby" not in canonicals, "tabby should merge into cat"
    assert "cat" in canonicals
    print("test_physical_entity_filter_removes_noise_from_ram OK")


def test_physical_entity_filter_off_by_default_keeps_detr():
    """filter_non_objects=False must not drop anything (DETR default)."""
    from synonyms import UnionCanonicalizer, build_raw_mentions, RawMention
    uc = UnionCanonicalizer()
    # DETR items should always be kept
    raws = [RawMention("relax", "detr"), RawMention("sleep", "detr"), RawMention("bed", "detr")]
    cands_filtered = uc.canonicalize_pool(raws, filter_non_objects=True)
    cands_unfiltered = uc.canonicalize_pool(raws, filter_non_objects=False)
    # DETR sources bypass the filter even when filter_non_objects=True
    can_filtered = {c.canonical for c in cands_filtered}
    can_unfiltered = {c.canonical for c in cands_unfiltered}
    # "bed" should be in both
    assert "bed" in can_filtered
    # DETR "relax" and "sleep" should be kept even with filter=True (DETR exemption)
    # (they're from a detection model, not RAM)
    assert len(can_filtered) == len(can_unfiltered)
    print("test_physical_entity_filter_off_by_default_keeps_detr OK")


# ---------------------------------------------------------------------------
# 3. FeatureScaler: sqrt(s_area) + z-score
# ---------------------------------------------------------------------------
def test_scaler_sqrt_applied_to_area_before_stats():
    from fit_gmm import FeatureScaler
    X_raw = np.array([[0.6, 0.25, 0.09], [0.1, 0.15, 0.25]])
    sc = FeatureScaler.fit(X_raw, use_area=True)
    # sqrt(0.09)=0.3, sqrt(0.25)=0.5; mean of area dim = 0.4, std = 0.1
    assert abs(sc.mean[2] - 0.4) < 1e-6, sc.mean
    assert abs(sc.std[2] - 0.1) < 1e-6, sc.std
    print("test_scaler_sqrt_applied_to_area_before_stats OK")


def test_scaler_zero_area_maps_to_negative():
    """A hallucinated object (s_area=0) should get a negative normalized area,
    well separated from a real object with s_area > 0."""
    from fit_gmm import FeatureScaler
    rng = np.random.RandomState(0)
    X_real = np.column_stack([rng.normal(0.6, 0.1, 100), rng.normal(0.25, 0.02, 100),
                               rng.uniform(0.05, 0.3, 100)])
    X_hall = np.column_stack([rng.normal(0.05, 0.02, 100), rng.normal(0.15, 0.02, 100),
                               np.zeros(100)])
    X_all = np.vstack([X_real, X_hall])
    sc = FeatureScaler.fit(X_all, use_area=True)

    # Hallucinated candidates have s_area=0, real ones have s_area>0
    X_real_norm = sc.transform(X_real)
    X_hall_norm = sc.transform(X_hall)
    # After normalization, real objects should have positive area dim, hallucinated negative
    assert X_real_norm[:, 2].mean() > 0, "real objects should have positive normalized area"
    assert X_hall_norm[:, 2].mean() < 0, "hallucinated objects should have negative normalized area"
    print("test_scaler_zero_area_maps_to_negative OK")


def test_scaler_clip_scores_are_better_separated_after_normalization():
    """The key motivation for z-score: s_clip spans a narrow range (~0.15-0.30),
    so in raw space it barely contributes to cluster separation. After z-scoring
    it should have comparable variance to s_det."""
    from fit_gmm import FeatureScaler
    rng = np.random.RandomState(1)
    X_real = np.column_stack([rng.normal(0.65, 0.08, 200),  # s_det: wide range
                               rng.normal(0.27, 0.02, 200),  # s_clip: narrow range
                               rng.uniform(0.05, 0.3, 200)])  # s_area
    X_hall = np.column_stack([rng.normal(0.05, 0.02, 200),
                               rng.normal(0.19, 0.02, 200),
                               np.zeros(200)])
    X_all = np.vstack([X_real, X_hall])
    sc = FeatureScaler.fit(X_all, use_area=True)
    X_norm = sc.transform(X_all)

    # After normalization: all dims should have std ≈ 1
    stds = X_norm.std(axis=0)
    for i, std in enumerate(stds):
        assert 0.8 < std < 1.2, f"dim {i} std after normalization should be ~1, got {std:.4f}"
    print("test_scaler_clip_scores_are_better_separated_after_normalization OK")


def test_full_pipeline_no_noise_objects_reach_gmm():
    """End-to-end: build a candidate pool from RAM/DETR/VLM, verify that noise
    words (relax, sleep, sitting) are not in the candidates fed to the GMM."""
    from synonyms import UnionCanonicalizer, build_raw_mentions
    from fit_gmm import pool_raw_features
    import json
    import tempfile, os

    uc = UnionCanonicalizer()
    ram = ["bed", "comfort", "relax", "sleep", "cat", "television"]
    detr = ["bed", "cat"]
    vlm = ["cat", "sitting", "bed"]  # 'sitting' filtered by POS in text_objects; simulate here

    raws = build_raw_mentions(ram, detr, vlm)
    cands = uc.canonicalize_pool(raws, filter_non_objects=True)
    names = [c.canonical for c in cands]

    # noise should not be present
    for noise in ["comfort", "relax", "sleep", "sitting"]:
        assert noise not in names, f"'{noise}' should not reach GMM: {names}"
    # real objects should be present
    for obj in ["bed", "cat", "tv"]:  # "television" maps to canonical "tv"
        assert obj in names, f"'{obj}' should be present: {names}"
    print("test_full_pipeline_no_noise_objects_reach_gmm OK")


if __name__ == "__main__":
    test_pos_filter_removes_gerunds_and_verbs()
    test_pos_filter_keeps_double_word_compounds()
    test_physical_entity_filter_removes_noise_from_ram()
    test_physical_entity_filter_off_by_default_keeps_detr()
    test_scaler_sqrt_applied_to_area_before_stats()
    test_scaler_zero_area_maps_to_negative()
    test_scaler_clip_scores_are_better_separated_after_normalization()
    test_full_pipeline_no_noise_objects_reach_gmm()
    print("\nALL v4 FEATURE TESTS PASSED")
