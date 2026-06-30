"""
gmm.py
======

Implements the Phase I "offline sorter" from Strategy8_Union_contrastive.pdf,
Section 3.2: a 2-component multivariate Gaussian Mixture Model over the 3D
feature vectors x_i = [s_det, s_clip, s_area]^T (Eq. 4), fit via
Expectation-Maximization (Eq. 7-14).

Fitting scope -- IMPORTANT design decision (confirmed with the user): the
paper's algorithm box literally fits one GMM per image. In practice that is
fragile (a single image may propose anywhere from 0 to a handful of
candidate objects -- nowhere near enough points to reliably fit a 3D, full-
covariance, 2-component mixture) and gives every image its own ungrounded
notion of "high" vs. "low" evidence. Per explicit instruction, we instead
fit ONE GLOBAL GMM on the pooled candidate feature vectors from a "fitting"
set of images (the tuning split), then FREEZE its parameters (pi, mu, Sigma)
and apply only the E-step (Eq. 8) to classify candidates in any other image
(tuning, held-out test, or the full dataset) -- mirroring the "fit-on-train,
apply-frozen-to-test" pattern used by other strategies in this codebase.

Tunable hyperparameters (exposed for the grid search in hyperparam_grid.py):
  - init_strategy: how (pi, mu, Sigma) are initialized before EM starts.
      'kmeans'        -- 2-means hard clustering on standardized features,
                         then closed-form mu/Sigma/pi from that assignment.
      'quantile'      -- split on the detection-confidence dimension at a
                         given percentile.
      'fixed_prior'   -- caller-supplied numeric mu_pos/mu_neg/Sigma.
  - learning_rate (lr): damping factor on the M-step update. The standard
      EM M-step (Eq. 9-12) computes closed-form new parameters; we then set
          theta_new = (1 - lr) * theta_old + lr * theta_closed_form
      so lr=1.0 reproduces textbook EM exactly, and lr<1.0 damps/smooths
      the trajectory (useful when the pooled feature set is noisy).
  - max_iters, tol: standard EM stopping criteria on log-likelihood (Eq. 13).
  - reg_covar: a small value added to the diagonal of every covariance
      matrix at every M-step, standard EM regularization to keep Sigma_k
      invertible (relevant since one of the two clusters can become very
      tight if, e.g., most "hallucinated" candidates have near-identical
      near-zero scores).

Correctness: `tests/test_gmm.py` checks this implementation against
`sklearn.mixture.GaussianMixture` (lr=1.0 case) on synthetic 2-cluster data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


_LOG_2PI = np.log(2.0 * np.pi)


def _safe_cov(cov: np.ndarray, reg_covar: float) -> np.ndarray:
    d = cov.shape[-1]
    return cov + reg_covar * np.eye(d)


def _solve_triangular_lower(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Forward substitution for L @ X = B with L lower-triangular.
    (numpy has no public solve_triangular outside scipy; this avoids adding
    a scipy dependency just for that one call.)"""
    n = L.shape[0]
    X = np.zeros_like(B, dtype=float)
    for i in range(n):
        X[i] = (B[i] - L[i, :i] @ X[:i]) / L[i, i]
    return X


def _multivariate_gaussian_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """log N(x | mean, cov) for each row of X (N, D). Numerically stable via
    Cholesky decomposition of the covariance matrix."""
    d = X.shape[1]
    diff = X - mean[None, :]
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # extremely defensive fallback: covariance not PD even after
        # regularization (should not happen given reg_covar > 0, but guard
        # anyway so a single bad fit can't crash a whole hyperparameter run)
        cov = cov + 1e-4 * np.eye(d)
        L = np.linalg.cholesky(cov)
    z = _solve_triangular_lower(L, diff.T)
    maha2 = np.sum(z ** 2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * _LOG_2PI + log_det + maha2)


@dataclass
class GMMParams:
    weights: np.ndarray   # (2,)
    means: np.ndarray     # (2, D)
    covariances: np.ndarray  # (2, D, D)
    pos_idx: int           # which component (0 or 1) is the Positive/Real cluster
    n_iter: int = 0
    converged: bool = False
    log_likelihood: float = float("nan")
    log_likelihood_history: List[float] = field(default_factory=list)
    n_fit_points: int = 0
    feature_names: Tuple[str, str, str] = ("s_det", "s_clip", "s_area")

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "covariances": self.covariances.tolist(),
            "pos_idx": self.pos_idx,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "log_likelihood": self.log_likelihood,
            "log_likelihood_history": self.log_likelihood_history,
            "n_fit_points": self.n_fit_points,
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GMMParams":
        return cls(
            weights=np.array(d["weights"], dtype=float),
            means=np.array(d["means"], dtype=float),
            covariances=np.array(d["covariances"], dtype=float),
            pos_idx=int(d["pos_idx"]),
            n_iter=int(d.get("n_iter", 0)),
            converged=bool(d.get("converged", False)),
            log_likelihood=float(d.get("log_likelihood", float("nan"))),
            log_likelihood_history=list(d.get("log_likelihood_history", [])),
            n_fit_points=int(d.get("n_fit_points", 0)),
            feature_names=tuple(d.get("feature_names", ["s_det", "s_clip", "s_area"])),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GMMParams":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _init_kmeans(X: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd

    km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
    labels = km.fit_predict(Xs)

    means = np.zeros((2, X.shape[1]))
    covs = np.zeros((2, X.shape[1], X.shape[1]))
    weights = np.zeros(2)
    for k in range(2):
        Xk = X[labels == k]
        if len(Xk) < 2:
            Xk = X  # degenerate cluster from kmeans init; fall back to global stats
        means[k] = Xk.mean(axis=0)
        covs[k] = np.cov(Xk.T) if Xk.shape[0] > 1 else np.eye(X.shape[1]) * 1e-2
        weights[k] = max(int((labels == k).sum()), 1) / len(X)
    weights = weights / weights.sum()
    return weights, means, covs


def _init_quantile(X: np.ndarray, det_dim: int = 0, percentile: float = 50.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresh = np.percentile(X[:, det_dim], percentile)
    high_mask = X[:, det_dim] >= thresh
    low_mask = ~high_mask
    if high_mask.sum() < 2:
        high_mask = X[:, det_dim] >= np.median(X[:, det_dim])
        low_mask = ~high_mask
    if low_mask.sum() < 2:
        low_mask = ~high_mask
        high_mask = ~low_mask

    Xhigh, Xlow = X[high_mask], X[low_mask]
    means = np.stack([Xlow.mean(axis=0), Xhigh.mean(axis=0)])
    covs = np.stack([
        np.cov(Xlow.T) if Xlow.shape[0] > 1 else np.eye(X.shape[1]) * 1e-2,
        np.cov(Xhigh.T) if Xhigh.shape[0] > 1 else np.eye(X.shape[1]) * 1e-2,
    ])
    weights = np.array([low_mask.mean(), high_mask.mean()])
    weights = weights / weights.sum()
    return weights, means, covs


class GlobalGMM:
    """2-component multivariate GMM with a configurable, damped EM fitter.

    Usage:
        gmm = GlobalGMM(learning_rate=1.0, max_iters=100, tol=1e-6,
                         init_strategy='kmeans')
        gmm.fit(X_pooled)                       # X_pooled: (N, 3) from many images
        gamma_pos = gmm.responsibility_positive(X_new)   # apply frozen params
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iters: int = 100,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        init_strategy: str = "kmeans",
        random_state: int = 0,
        init_means: Optional[np.ndarray] = None,
        init_covariances: Optional[np.ndarray] = None,
        init_weights: Optional[np.ndarray] = None,
        det_dim: int = 0,
    ):
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if init_strategy not in ("kmeans", "quantile", "fixed_prior"):
            raise ValueError(f"unknown init_strategy: {init_strategy}")
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.init_strategy = init_strategy
        self.random_state = random_state
        self.init_means = None if init_means is None else np.asarray(init_means, dtype=float)
        self.init_covariances = None if init_covariances is None else np.asarray(init_covariances, dtype=float)
        self.init_weights = None if init_weights is None else np.asarray(init_weights, dtype=float)
        self.det_dim = det_dim

        self.params: Optional[GMMParams] = None

    # ---- initialization -------------------------------------------------
    def _initialize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.init_strategy == "fixed_prior":
            if self.init_means is None or self.init_covariances is None:
                raise ValueError("init_strategy='fixed_prior' requires init_means and init_covariances")
            weights = self.init_weights if self.init_weights is not None else np.array([0.5, 0.5])
            return weights.copy(), self.init_means.copy(), self.init_covariances.copy()
        elif self.init_strategy == "kmeans":
            return _init_kmeans(X, self.random_state)
        else:  # quantile
            return _init_quantile(X, det_dim=self.det_dim)

    # ---- E-step -----------------------------------------------------------
    def _e_step(self, X: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Eq. 8: responsibilities gamma_ik, plus the data log-likelihood
        (Eq. 13) under the current parameters."""
        K = weights.shape[0]
        log_resp = np.zeros((X.shape[0], K))
        for k in range(K):
            cov_k = _safe_cov(covs[k], self.reg_covar)
            log_resp[:, k] = np.log(max(weights[k], 1e-300)) + _multivariate_gaussian_logpdf(X, means[k], cov_k)
        max_log = np.max(log_resp, axis=1, keepdims=True)
        log_norm = max_log + np.log(np.sum(np.exp(log_resp - max_log), axis=1, keepdims=True))
        log_resp_normed = log_resp - log_norm
        resp = np.exp(log_resp_normed)
        log_likelihood = float(np.sum(log_norm))
        return resp, log_likelihood

    # ---- M-step (closed form) ---------------------------------------------
    def _m_step_closed_form(self, X: np.ndarray, resp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Eq. 9-12."""
        N, D = X.shape
        K = resp.shape[1]
        Nk = resp.sum(axis=0)
        Nk_safe = np.clip(Nk, 1e-12, None)

        means = (resp.T @ X) / Nk_safe[:, None]

        covs = np.zeros((K, D, D))
        for k in range(K):
            diff = X - means[k][None, :]
            weighted = diff * resp[:, k][:, None]
            covs[k] = (weighted.T @ diff) / Nk_safe[k]

        pis = Nk / N
        return pis, means, covs

    # ---- fit ----------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "GlobalGMM":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (N, D), got shape {X.shape}")
        N, D = X.shape
        if N < 4:
            raise ValueError(
                f"GlobalGMM.fit needs a reasonably sized pooled feature set "
                f"(got N={N}); pool more fitting images."
            )

        weights, means, covs = self._initialize(X)
        for k in range(covs.shape[0]):
            covs[k] = _safe_cov(covs[k], self.reg_covar)

        ll_history: List[float] = []
        prev_ll = -np.inf
        converged = False
        n_iter = 0

        for it in range(1, self.max_iters + 1):
            n_iter = it
            resp, ll = self._e_step(X, weights, means, covs)
            ll_history.append(ll)

            pis_cf, means_cf, covs_cf = self._m_step_closed_form(X, resp)

            lr = self.learning_rate
            weights = (1 - lr) * weights + lr * pis_cf
            weights = weights / weights.sum()
            means = (1 - lr) * means + lr * means_cf
            covs = (1 - lr) * covs + lr * covs_cf
            for k in range(covs.shape[0]):
                covs[k] = _safe_cov(covs[k], self.reg_covar)

            if it > 1 and abs(ll - prev_ll) < self.tol * max(1.0, abs(prev_ll)):
                converged = True
                prev_ll = ll
                break
            prev_ll = ll

        # one final E-step to report the log-likelihood under the parameters
        # we are about to freeze
        _, final_ll = self._e_step(X, weights, means, covs)
        ll_history.append(final_ll)

        pos_idx = int(np.argmax(means[:, self.det_dim]))  # Eq. 14

        self.params = GMMParams(
            weights=weights,
            means=means,
            covariances=covs,
            pos_idx=pos_idx,
            n_iter=n_iter,
            converged=converged,
            log_likelihood=final_ll,
            log_likelihood_history=ll_history,
            n_fit_points=N,
        )
        return self

    # ---- apply frozen params to new data -----------------------------------
    def responsibility_positive(self, X: np.ndarray) -> np.ndarray:
        """Eq. 8 (E-step only) applied with FROZEN parameters: returns
        gamma_i = responsibility of the Positive cluster for each row of X.
        This is what should be used for any image that was NOT part of the
        pool this GMM was fit on (held-out test images, the full-500 run,
        etc.) as well as, for simplicity, the fitting images themselves."""
        if self.params is None:
            raise RuntimeError("GlobalGMM has not been fit yet")
        X = np.asarray(X, dtype=float)
        resp, _ = self._e_step(X, self.params.weights, self.params.means, self.params.covariances)
        return resp[:, self.params.pos_idx]

    @classmethod
    def from_params(cls, params: GMMParams, **kwargs) -> "GlobalGMM":
        obj = cls(**kwargs)
        obj.params = params
        return obj
