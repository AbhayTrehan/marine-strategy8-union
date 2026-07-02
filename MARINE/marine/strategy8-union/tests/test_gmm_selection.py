"""Run with: python3 tests/test_gmm_selection.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from gmm import GlobalGMM
from gmm_selection import GMMSelectionResult, compute_gmm_quality, select_best_gmm_preset
from fit_gmm import FeatureScaler, pool_raw_features
from hyperparam_grid import DEFAULT_GMM_PRESETS


def _fake_cache_well_separated(n_images=80, seed=0):
    rng = np.random.RandomState(seed)
    cache = {}
    for i in range(n_images):
        n_cand = rng.randint(2, 6)
        cands = []
        for j in range(n_cand):
            is_pos = rng.rand() < 0.5
            if is_pos:
                s_det, s_clip, s_area = np.clip(rng.normal([0.75, 0.30, 0.10], 0.03), 0, 1)
            else:
                s_det, s_clip, s_area = np.clip(rng.normal([0.05, 0.04, 0.01], 0.02), 0, 1)
            cands.append({"canonical": f"obj{j}", "s_det": float(s_det), "s_clip": float(s_clip), "s_area": float(s_area)})
        cache[f"img{i}.jpg"] = {"image": f"img{i}.jpg", "candidates": cands}
    return cache


def _fake_cache_overlapping(n_images=80, seed=1):
    """Both 'clusters' drawn from nearly the same distribution -- a GMM
    fit on this should have much worse intrinsic separation."""
    rng = np.random.RandomState(seed)
    cache = {}
    for i in range(n_images):
        n_cand = rng.randint(2, 6)
        cands = []
        for j in range(n_cand):
            s_det, s_clip, s_area = np.clip(rng.normal([0.4, 0.15, 0.05], 0.15), 0, 1)
            cands.append({"canonical": f"obj{j}", "s_det": float(s_det), "s_clip": float(s_clip), "s_area": float(s_area)})
        cache[f"img{i}.jpg"] = {"image": f"img{i}.jpg", "candidates": cands}
    return cache


def test_well_separated_data_scores_higher_silhouette_than_overlapping():
    images = [f"img{i}.jpg" for i in range(80)]

    cache_sep = _fake_cache_well_separated()
    from fit_gmm import fit_global_gmm, pool_features
    gmm_sep, sc_sep = fit_global_gmm(cache_sep, images, DEFAULT_GMM_PRESETS[0])
    q_sep = compute_gmm_quality(gmm_sep, sc_sep.transform(pool_raw_features(cache_sep, images)))

    cache_overlap = _fake_cache_overlapping()
    gmm_overlap, sc_ov = fit_global_gmm(cache_overlap, images, DEFAULT_GMM_PRESETS[0])
    q_overlap = compute_gmm_quality(gmm_overlap, sc_ov.transform(pool_raw_features(cache_overlap, images)))

    assert q_sep["silhouette"] > q_overlap["silhouette"], (q_sep, q_overlap)
    assert q_sep["mean_separation"] > q_overlap["mean_separation"]
    print("test_well_separated_data_scores_higher_silhouette_than_overlapping OK")


def test_select_best_gmm_preset_picks_highest_silhouette():
    images = [f"img{i}.jpg" for i in range(80)]
    cache = _fake_cache_well_separated()

    result = select_best_gmm_preset(cache, images, DEFAULT_GMM_PRESETS)
    assert result.chosen_preset_name in {p["name"] for p in DEFAULT_GMM_PRESETS}
    # the chosen preset's silhouette should be >= every other considered preset's
    chosen_sil = result.quality_by_preset[result.chosen_preset_name]["silhouette"]
    for name, q in result.quality_by_preset.items():
        assert chosen_sil >= q["silhouette"] - 1e-9, (name, q, chosen_sil)
    print("test_select_best_gmm_preset_picks_highest_silhouette OK")


def test_no_lvlm_generation_involved():
    """Sanity: select_best_gmm_preset only needs the cache + image list +
    presets + use_area flag -- nothing resembling a model/tokenizer/processor argument."""
    import inspect
    sig = inspect.signature(select_best_gmm_preset)
    params = set(sig.parameters)
    assert params == {"candidate_pool_cache", "fitting_images", "candidate_presets", "use_area"}
    print("test_no_lvlm_generation_involved OK")


def test_selection_result_serialization_roundtrip(tmp_path="/tmp/_test_gmm_selection.json"):
    images = [f"img{i}.jpg" for i in range(80)]
    cache = _fake_cache_well_separated()
    result = select_best_gmm_preset(cache, images, DEFAULT_GMM_PRESETS)
    result.save(tmp_path)
    loaded = GMMSelectionResult.load(tmp_path)
    assert loaded.chosen_preset_name == result.chosen_preset_name
    assert loaded.quality_by_preset.keys() == result.quality_by_preset.keys()
    assert isinstance(loaded.chosen_scaler, FeatureScaler)

    gmm = GlobalGMM.from_params(loaded.chosen_gmm_params)
    X_raw = np.array([[0.7, 0.3, 0.1]])   # 3D [s_det, s_clip, s_area] matching default use_area=True
    X_norm = loaded.chosen_scaler.transform(X_raw)
    g1 = gmm.responsibility_positive(X_norm)
    assert g1.shape == (1,)
    os.remove(tmp_path)
    print("test_selection_result_serialization_roundtrip OK")


def test_degenerate_collapsed_fit_does_not_crash_silhouette():
    """All points identical -> GMM may collapse both components onto the
    same point; silhouette_score would raise on a single-label clustering,
    compute_gmm_quality must handle that gracefully (worst-case score)."""
    cache = {
        "img0.jpg": {
            "image": "img0.jpg",
            "candidates": [{"canonical": "x", "s_det": 0.5, "s_clip": 0.2, "s_area": 0.1} for _ in range(20)],
        }
    }
    images = ["img0.jpg"]
    from fit_gmm import fit_global_gmm, pool_features
    from fit_gmm import pool_raw_features, FeatureScaler
    gmm, sc = fit_global_gmm(cache, images, DEFAULT_GMM_PRESETS[0])
    X_raw = pool_raw_features(cache, images)
    X_norm = sc.transform(X_raw)
    q = compute_gmm_quality(gmm, X_norm)
    assert q["silhouette"] == -1.0 or -1.0 <= q["silhouette"] <= 1.0
    print("test_degenerate_collapsed_fit_does_not_crash_silhouette OK")


if __name__ == "__main__":
    test_well_separated_data_scores_higher_silhouette_than_overlapping()
    test_select_best_gmm_preset_picks_highest_silhouette()
    test_no_lvlm_generation_involved()
    test_selection_result_serialization_roundtrip()
    test_degenerate_collapsed_fit_does_not_crash_silhouette()
    print("\nALL gmm_selection.py TESTS PASSED")
