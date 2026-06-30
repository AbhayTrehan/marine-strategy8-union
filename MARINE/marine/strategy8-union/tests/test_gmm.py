"""Run with: python3 tests/test_gmm.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from gmm import GlobalGMM, GMMParams, _multivariate_gaussian_logpdf


def _synthetic_two_clusters(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    mean_pos = np.array([0.7, 0.3, 0.1])
    mean_neg = np.array([0.1, 0.05, 0.02])
    cov = np.eye(3) * 0.01
    X_pos = rng.multivariate_normal(mean_pos, cov, size=n // 2)
    X_neg = rng.multivariate_normal(mean_neg, cov, size=n // 2)
    X = np.vstack([X_pos, X_neg])
    perm = rng.permutation(len(X))
    return X[perm], mean_pos, mean_neg


def test_logpdf_matches_scipy_style_reference():
    # reference via numpy's own (slow, explicit) formula for a single point
    rng = np.random.RandomState(0)
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.8]])
    X = rng.multivariate_normal(mean, cov, size=5)
    ours = _multivariate_gaussian_logpdf(X, mean, cov)

    inv = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    d = 3
    ref = np.zeros(5)
    for i in range(5):
        diff = X[i] - mean
        maha2 = diff @ inv @ diff
        ref[i] = -0.5 * (d * np.log(2 * np.pi) + logdet + maha2)
    assert np.allclose(ours, ref, atol=1e-8), (ours, ref)
    print("test_logpdf_matches_scipy_style_reference OK")


def test_matches_sklearn_em():
    from sklearn.mixture import GaussianMixture

    X, mean_pos, mean_neg = _synthetic_two_clusters(n=2000, seed=0)

    ours = GlobalGMM(learning_rate=1.0, max_iters=200, tol=1e-10, init_strategy="kmeans", random_state=0)
    ours.fit(X)

    sk = GaussianMixture(n_components=2, covariance_type="full", max_iter=200, tol=1e-10, random_state=0)
    sk.fit(X)

    our_ll = ours.params.log_likelihood
    sk_ll = sk.score(X) * len(X)
    assert abs(our_ll - sk_ll) < 1.0, (our_ll, sk_ll)

    # means should match sklearn's solution up to component ordering
    our_means_sorted = ours.params.means[np.argsort(ours.params.means[:, 0])]
    sk_means_sorted = sk.means_[np.argsort(sk.means_[:, 0])]
    assert np.allclose(our_means_sorted, sk_means_sorted, atol=1e-3), (our_means_sorted, sk_means_sorted)
    print("test_matches_sklearn_em OK")


def test_positive_cluster_identification_eq14():
    X, mean_pos, mean_neg = _synthetic_two_clusters(seed=1)
    gmm = GlobalGMM(learning_rate=1.0, max_iters=200, init_strategy="kmeans", random_state=0).fit(X)
    pos_mean = gmm.params.means[gmm.params.pos_idx]
    neg_mean = gmm.params.means[1 - gmm.params.pos_idx]
    assert pos_mean[0] > neg_mean[0]  # higher mean detection confidence (dim 0 = s_det)
    assert pos_mean[0] > 0.5 and neg_mean[0] < 0.3
    print("test_positive_cluster_identification_eq14 OK")


def test_damping_converges_to_similar_optimum_slower():
    X, _, _ = _synthetic_two_clusters(seed=2)
    full = GlobalGMM(learning_rate=1.0, max_iters=500, tol=1e-10, init_strategy="kmeans", random_state=0).fit(X)
    damped = GlobalGMM(learning_rate=0.3, max_iters=500, tol=1e-10, init_strategy="kmeans", random_state=0).fit(X)
    assert damped.params.n_iter >= full.params.n_iter
    full_sorted = full.params.means[np.argsort(full.params.means[:, 0])]
    damped_sorted = damped.params.means[np.argsort(damped.params.means[:, 0])]
    assert np.allclose(full_sorted, damped_sorted, atol=0.02)
    print("test_damping_converges_to_similar_optimum_slower OK")


def test_all_init_strategies_agree():
    X, _, _ = _synthetic_two_clusters(seed=3)
    results = []
    for strat, kwargs in [
        ("kmeans", {}),
        ("quantile", {}),
        ("fixed_prior", {
            "init_means": np.array([[0.6, 0.25, 0.08], [0.05, 0.1, 0.02]]),
            "init_covariances": np.array([np.eye(3) * 0.02, np.eye(3) * 0.02]),
        }),
    ]:
        gmm = GlobalGMM(learning_rate=1.0, max_iters=300, tol=1e-10, init_strategy=strat, **kwargs)
        gmm.fit(X)
        m = gmm.params.means[gmm.params.pos_idx]
        results.append(m)
    for r in results[1:]:
        assert np.allclose(results[0], r, atol=0.03), (results[0], r)
    print("test_all_init_strategies_agree OK")


def test_frozen_apply_to_new_unseen_data():
    X, mean_pos, mean_neg = _synthetic_two_clusters(seed=4)
    gmm = GlobalGMM(learning_rate=1.0, max_iters=200, init_strategy="kmeans", random_state=0).fit(X)

    rng = np.random.RandomState(99)
    cov = np.eye(3) * 0.01
    X_new_pos = rng.multivariate_normal(mean_pos, cov, size=20)
    X_new_neg = rng.multivariate_normal(mean_neg, cov, size=20)

    gamma_pos = gmm.responsibility_positive(X_new_pos)
    gamma_neg = gmm.responsibility_positive(X_new_neg)
    assert (gamma_pos > 0.9).mean() > 0.9, gamma_pos
    assert (gamma_neg < 0.1).mean() > 0.9, gamma_neg
    print("test_frozen_apply_to_new_unseen_data OK")


def test_serialization_roundtrip(tmp_path="/tmp/_test_gmm_params.json"):
    X, _, _ = _synthetic_two_clusters(seed=5)
    gmm = GlobalGMM(learning_rate=1.0, max_iters=200, init_strategy="kmeans", random_state=0).fit(X)
    gmm.params.save(tmp_path)
    loaded = GMMParams.load(tmp_path)
    gmm2 = GlobalGMM.from_params(loaded)

    X_probe = X[:50]
    g1 = gmm.responsibility_positive(X_probe)
    g2 = gmm2.responsibility_positive(X_probe)
    assert np.allclose(g1, g2)
    os.remove(tmp_path)
    print("test_serialization_roundtrip OK")


def test_rejects_too_few_points():
    try:
        GlobalGMM().fit(np.random.rand(2, 3))
        raise AssertionError("should have raised ValueError")
    except ValueError:
        pass
    print("test_rejects_too_few_points OK")


def test_rejects_invalid_learning_rate():
    for bad_lr in [0.0, -0.1, 1.1]:
        try:
            GlobalGMM(learning_rate=bad_lr)
            raise AssertionError(f"should have raised for lr={bad_lr}")
        except ValueError:
            pass
    print("test_rejects_invalid_learning_rate OK")


def test_degenerate_imbalanced_data_does_not_crash():
    # 95% of points tightly clustered near zero, 5% spread out -- checks
    # reg_covar keeps the tight cluster's covariance invertible.
    rng = np.random.RandomState(6)
    X_tight = rng.normal(loc=[0.02, 0.01, 0.005], scale=1e-4, size=(950, 3))
    X_spread = rng.uniform(low=[0.3, 0.1, 0.02], high=[0.9, 0.5, 0.3], size=(50, 3))
    X = np.clip(np.vstack([X_tight, X_spread]), 0, None)
    gmm = GlobalGMM(learning_rate=1.0, max_iters=200, reg_covar=1e-6, init_strategy="kmeans", random_state=0)
    gmm.fit(X)  # should not raise
    assert np.isfinite(gmm.params.log_likelihood)
    print("test_degenerate_imbalanced_data_does_not_crash OK")


if __name__ == "__main__":
    test_logpdf_matches_scipy_style_reference()
    test_matches_sklearn_em()
    test_positive_cluster_identification_eq14()
    test_damping_converges_to_similar_optimum_slower()
    test_all_init_strategies_agree()
    test_frozen_apply_to_new_unseen_data()
    test_serialization_roundtrip()
    test_rejects_too_few_points()
    test_rejects_invalid_learning_rate()
    test_degenerate_imbalanced_data_does_not_crash()
    print("\nALL gmm.py TESTS PASSED")
