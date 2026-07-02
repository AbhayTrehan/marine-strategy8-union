"""Run with: python3 tests/test_fit_gmm.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fit_gmm import fit_global_gmm, pool_raw_features, pool_and_normalize, FeatureScaler
from hyperparam_grid import DEFAULT_GMM_PRESETS


def _fake_cache(seed=0):
    rng = np.random.RandomState(seed)
    cache = {}
    for i in range(60):
        n_cand = rng.randint(1, 6)
        cands = []
        for j in range(n_cand):
            is_pos = rng.rand() < 0.5
            if is_pos:
                s_det = float(np.clip(rng.normal(0.7, 0.05), 0, 1))
                s_clip = float(np.clip(rng.normal(0.3, 0.03), 0, 1))
                s_area = float(np.clip(rng.normal(0.1, 0.02), 0, 1))
            else:
                s_det = float(np.clip(rng.normal(0.1, 0.05), 0, 1))
                s_clip = float(np.clip(rng.normal(0.05, 0.02), 0, 1))
                s_area = float(np.clip(rng.normal(0.02, 0.01), 0, 1))
            cands.append({"canonical": f"obj{j}", "s_det": s_det, "s_clip": s_clip, "s_area": s_area})
        cache[f"img{i}.jpg"] = {"image": f"img{i}.jpg", "candidates": cands}
    return cache


def test_pool_features_stacks_correctly():
    cache = _fake_cache()
    images = list(cache.keys())[:10]
    X = pool_raw_features(cache, images, use_area=False)   # raw 2D
    expected_n = sum(len(cache[img]["candidates"]) for img in images)
    assert X.shape == (expected_n, 2), X.shape

    X3 = pool_raw_features(cache, images, use_area=True)   # raw 3D
    assert X3.shape == (expected_n, 3), X3.shape

    # pool_and_normalize should return (X_norm, scaler)
    X_norm, sc = pool_and_normalize(cache, images, use_area=True)
    assert X_norm.shape == (expected_n, 3)
    assert isinstance(sc, FeatureScaler)
    print("test_pool_features_stacks_correctly OK")


def test_pool_features_handles_missing_images():
    cache = _fake_cache()
    X = pool_raw_features(cache, ["img0.jpg", "nonexistent.jpg"], use_area=False)
    assert X.shape[0] == len(cache["img0.jpg"]["candidates"])
    assert X.shape[1] == 2  # default: 2D
    print("test_pool_features_handles_missing_images OK")


def test_pool_features_empty():
    cache = {}
    X = pool_raw_features(cache, ["a.jpg"], use_area=False)
    assert X.shape == (0, 2)
    print("test_pool_features_empty OK")


def test_fit_global_gmm_with_each_preset():
    cache = _fake_cache()
    images = list(cache.keys())
    for preset in DEFAULT_GMM_PRESETS:
        gmm, sc = fit_global_gmm(cache, images, preset)
        assert gmm.params is not None
        assert isinstance(sc, FeatureScaler)
        assert gmm.params.n_fit_points == pool_raw_features(cache, images).shape[0]
        # the positive cluster should indeed have higher mean s_det
        pos_mean = gmm.params.means[gmm.params.pos_idx][0]
        neg_mean = gmm.params.means[1 - gmm.params.pos_idx][0]
        assert pos_mean > neg_mean
    print("test_fit_global_gmm_with_each_preset OK")


def test_fit_global_gmm_raises_on_too_few_points():
    cache = {"a.jpg": {"image": "a.jpg", "candidates": [{"canonical": "x", "s_det": 0.5, "s_clip": 0.2, "s_area": 0.1}]}}
    try:
        fit_global_gmm(cache, ["a.jpg"], DEFAULT_GMM_PRESETS[0])
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("test_fit_global_gmm_raises_on_too_few_points OK")


if __name__ == "__main__":
    test_pool_features_stacks_correctly()
    test_pool_features_handles_missing_images()
    test_pool_features_empty()
    test_fit_global_gmm_with_each_preset()
    test_fit_global_gmm_raises_on_too_few_points()
    print("\nALL fit_gmm.py TESTS PASSED")
