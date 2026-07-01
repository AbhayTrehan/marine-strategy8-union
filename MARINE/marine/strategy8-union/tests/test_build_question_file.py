"""Run with: python3 tests/test_build_question_file.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from build_question_file import build_question_file, classify_image_candidates
from fit_gmm import FeatureScaler
from gmm import GlobalGMM


def _make_fitted_gmm():
    rng = np.random.RandomState(0)
    # Use 2D (no area) for speed; scaler is fitted on same data
    X_raw = np.clip(np.vstack([
        rng.normal([0.7, 0.3], 0.02, size=(200, 2)),
        rng.normal([0.05, 0.05], 0.02, size=(200, 2)),
    ]), 0, 1)
    scaler = FeatureScaler.fit(X_raw, use_area=False)
    X_norm = scaler.transform(X_raw)
    gmm = GlobalGMM(learning_rate=1.0, max_iters=200, init_strategy="kmeans", random_state=0)
    gmm.fit(X_norm)
    return gmm, scaler


def test_classify_image_candidates_splits_by_tau():
    gmm, scaler = _make_fitted_gmm()
    candidates = [
        {"canonical": "dog", "s_det": 0.75, "s_clip": 0.32, "s_area": 0.11},
        {"canonical": "fork", "s_det": 0.04, "s_clip": 0.03, "s_area": 0.01},
    ]
    o_pos, o_neg, gammas = classify_image_candidates(candidates, gmm, scaler, tau=0.5)
    assert o_pos == ["dog"]
    assert o_neg == ["fork"]
    assert len(gammas) == 2
    assert gammas[0] > 0.9
    assert gammas[1] < 0.1
    print("test_classify_image_candidates_splits_by_tau OK")


def test_classify_image_candidates_empty():
    gmm, scaler = _make_fitted_gmm()
    o_pos, o_neg, gammas = classify_image_candidates([], gmm, scaler, tau=0.5)
    assert o_pos == [] and o_neg == [] and gammas == []
    print("test_classify_image_candidates_empty OK")


def test_tau_threshold_moves_objects_between_sets():
    gmm, scaler = _make_fitted_gmm()
    candidates = [{"canonical": "borderline", "s_det": 0.35, "s_clip": 0.15, "s_area": 0.05}]
    _, _, gammas = classify_image_candidates(candidates, gmm, scaler, tau=0.5)
    gamma = gammas[0]
    o_pos_low, o_neg_low, _ = classify_image_candidates(candidates, gmm, scaler, tau=max(0.0, gamma - 0.2))
    o_pos_high, o_neg_high, _ = classify_image_candidates(candidates, gmm, scaler, tau=min(1.0, gamma + 0.2))
    assert o_pos_low == ["borderline"]
    assert o_pos_high == []
    print("test_tau_threshold_moves_objects_between_sets OK")


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def _write_jsonl(path, items):
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def test_build_question_file_chair_format():
    d = tempfile.mkdtemp()
    qpath = os.path.join(d, "chair.json")
    _write_json(qpath, [
        {"id": 1, "image": "img0.jpg", "conversations": [{"from": "human", "value": "Generate a short caption of the image."}, {"from": "gpt", "value": ""}]},
        {"id": 2, "image": "img1.jpg", "conversations": [{"from": "human", "value": "Generate a short caption of the image."}, {"from": "gpt", "value": ""}]},
    ])
    cache = {
        "img0.jpg": {"image": "img0.jpg", "candidates": [
            {"canonical": "dog", "s_det": 0.8, "s_clip": 0.3, "s_area": 0.1},
            {"canonical": "fork", "s_det": 0.02, "s_clip": 0.02, "s_area": 0.01},
        ]},
        "img1.jpg": {"image": "img1.jpg", "candidates": []},
    }
    gmm, scaler = _make_fitted_gmm()
    questions, per_image = build_question_file(qpath, cache, gmm, scaler, tau=0.5)
    assert len(questions) == 2
    q0 = next(q for q in questions if q["id"] == 1)
    convs = {c["from"]: c["value"] for c in q0["conversations"]}
    assert "dog" in convs["guidance_pos"]
    assert "fork" in convs["guidance_neg"]
    assert convs["human"] == "Generate a short caption of the image."

    q1 = next(q for q in questions if q["id"] == 2)
    convs1 = {c["from"]: c["value"] for c in q1["conversations"]}
    # no candidates at all -> both branches fall back to the plain query
    assert convs1["guidance_pos"] == convs1["human"]
    assert convs1["guidance_neg"] == convs1["human"]

    assert per_image["img0.jpg"]["o_pos"] == ["dog"]
    assert per_image["img0.jpg"]["o_neg"] == ["fork"]
    print("test_build_question_file_chair_format OK")


def test_build_question_file_pope_format_and_per_image_caching():
    d = tempfile.mkdtemp()
    qpath = os.path.join(d, "pope.json")
    # POPE-style flat jsonl, 3 questions sharing ONE image (like the real
    # 6-questions-per-image structure)
    _write_jsonl(qpath, [
        {"question_id": 1, "image": "img0.jpg", "text": "Is there a dog in the image?", "label": "yes"},
        {"question_id": 2, "image": "img0.jpg", "text": "Is there a fork in the image?", "label": "no"},
        {"question_id": 3, "image": "img0.jpg", "text": "Is there a cat in the image?", "label": "no"},
    ])
    cache = {
        "img0.jpg": {"image": "img0.jpg", "candidates": [
            {"canonical": "dog", "s_det": 0.8, "s_clip": 0.3, "s_area": 0.1},
        ]},
    }
    gmm, scaler = _make_fitted_gmm()
    questions, per_image = build_question_file(qpath, cache, gmm, scaler, tau=0.5)
    assert len(questions) == 3
    assert len(per_image) == 1  # classification computed once, shared across all 3 questions
    for q in questions:
        convs = {c["from"]: c["value"] for c in q["conversations"]}
        assert "dog" in convs["guidance_pos"]
    print("test_build_question_file_pope_format_and_per_image_caching OK")


def test_build_question_file_image_filter():
    d = tempfile.mkdtemp()
    qpath = os.path.join(d, "chair.json")
    _write_json(qpath, [
        {"id": 1, "image": "img0.jpg", "conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": ""}]},
        {"id": 2, "image": "img1.jpg", "conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": ""}]},
    ])
    cache = {
        "img0.jpg": {"image": "img0.jpg", "candidates": []},
        "img1.jpg": {"image": "img1.jpg", "candidates": []},
    }
    gmm, scaler = _make_fitted_gmm()
    questions, per_image = build_question_file(qpath, cache, gmm, scaler, tau=0.5, image_filter=["img0.jpg"])
    assert len(questions) == 1
    assert questions[0]["image"] == "img0.jpg"
    print("test_build_question_file_image_filter OK")


def test_build_question_file_missing_image_in_cache_treated_as_no_candidates():
    d = tempfile.mkdtemp()
    qpath = os.path.join(d, "chair.json")
    _write_json(qpath, [
        {"id": 1, "image": "unknown.jpg", "conversations": [{"from": "human", "value": "Generate a short caption of the image."}, {"from": "gpt", "value": ""}]},
    ])
    gmm, scaler = _make_fitted_gmm()
    questions, per_image = build_question_file(qpath, {}, gmm, scaler, tau=0.5)
    assert len(questions) == 1
    convs = {c["from"]: c["value"] for c in questions[0]["conversations"]}
    assert convs["guidance_pos"] == convs["human"]
    print("test_build_question_file_missing_image_in_cache_treated_as_no_candidates OK")


if __name__ == "__main__":
    test_classify_image_candidates_splits_by_tau()
    test_classify_image_candidates_empty()
    test_tau_threshold_moves_objects_between_sets()
    test_build_question_file_chair_format()
    test_build_question_file_pope_format_and_per_image_caching()
    test_build_question_file_image_filter()
    test_build_question_file_missing_image_in_cache_treated_as_no_candidates()
    print("\nALL build_question_file.py TESTS PASSED")
