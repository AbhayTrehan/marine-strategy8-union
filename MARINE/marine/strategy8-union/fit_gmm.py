"""
fit_gmm.py
==========

Step B of the Strategy 8-U pipeline: pool feature vectors from the tuning
images, fit one global 2-component GMM, and return it alongside a fitted
FeatureScaler.

Feature vector (default, use_area=True): x_i = [s_det, s_clip, s_area]
  Before feeding to the GMM, two transforms are applied:
    1. sqrt(s_area) -- reduces the dominance of large objects and spreads
       the near-zero region so that small-but-real objects (s_area≈0.005)
       are no longer indistinguishable from hallucinated objects (s_area=0).
    2. z-score normalize ALL dimensions -- brings s_clip's narrow 0.15-0.30
       range to the same influence as s_det's wider 0-0.9+ range. Stats are
       fitted ONCE on the tuning pool and saved alongside the GMM so that
       exactly the same transform is applied at inference time.

Feature vector (use_area=False, via --no_area_feature):
    x_i = [s_det, s_clip] -- 2D, same z-score normalization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from gmm import GlobalGMM, GMMParams


# ---------------------------------------------------------------------------
# FeatureScaler: sqrt(area) transform + z-score normalization
# ---------------------------------------------------------------------------
@dataclass
class FeatureScaler:
    """Encapsulates the sqrt(s_area) + z-score normalization fitted on the
    tuning pool. Stored alongside the GMM params so inference applies the
    EXACT same transform the GMM was trained on."""
    mean: np.ndarray   # (D,)
    std: np.ndarray    # (D,)  -- protected from near-zero division
    use_area: bool

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Apply sqrt(area) then z-score to a raw (N, D) feature matrix.
        Safe for single-row arrays (i.e. one candidate at a time)."""
        X = np.array(X_raw, dtype=float)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if self.use_area and X.shape[1] >= 3:
            X = X.copy()
            X[:, 2] = np.sqrt(np.maximum(X[:, 2], 0.0))
        return (X - self.mean) / self.std

    @classmethod
    def fit(cls, X_raw: np.ndarray, use_area: bool = True) -> "FeatureScaler":
        """Fit from a raw (N, D) feature matrix (applies sqrt to area dim
        first, then computes mean/std so they reflect the final distribution
        the GMM will actually see)."""
        X = np.array(X_raw, dtype=float)
        if use_area and X.shape[1] >= 3:
            X = X.copy()
            X[:, 2] = np.sqrt(np.maximum(X[:, 2], 0.0))
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0   # constant dimension: keep original scale
        return cls(mean=mean, std=std, use_area=use_area)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "use_area": self.use_area,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        return cls(
            mean=np.array(d["mean"], dtype=float),
            std=np.array(d["std"], dtype=float),
            use_area=bool(d["use_area"]),
        )


# ---------------------------------------------------------------------------
# Raw feature pooling (no transforms -- scaler not yet applied)
# ---------------------------------------------------------------------------
def pool_raw_features(
    candidate_pool_cache: Dict[str, dict],
    fitting_images: Sequence[str],
    use_area: bool = True,
) -> np.ndarray:
    """Returns an (N, D) array of RAW (untransformed) feature vectors for
    every candidate from every image in `fitting_images`.
    D=3 ([s_det, s_clip, s_area]) when use_area=True, D=2 otherwise."""
    dims = ["s_det", "s_clip", "s_area"] if use_area else ["s_det", "s_clip"]
    D = len(dims)
    rows: List[List[float]] = []
    for img in fitting_images:
        rec = candidate_pool_cache.get(img)
        if rec is None:
            continue
        for c in rec["candidates"]:
            rows.append([c[d] for d in dims])
    if not rows:
        return np.zeros((0, D))
    return np.array(rows, dtype=float)


def pool_and_normalize(
    candidate_pool_cache: Dict[str, dict],
    fitting_images: Sequence[str],
    use_area: bool = True,
    scaler: Optional[FeatureScaler] = None,
):
    """Pools raw features and applies (or fits) the scaler.

    Returns (X_normalized, scaler):
      - If scaler is None, fits a new one from the pooled data (training mode).
      - If scaler is provided, applies it as-is (inference mode, e.g. when
        evaluating the same scaler on a different image subset).
    """
    X_raw = pool_raw_features(candidate_pool_cache, fitting_images, use_area=use_area)
    if scaler is None:
        scaler = FeatureScaler.fit(X_raw, use_area=use_area)
    X_norm = scaler.transform(X_raw)
    return X_norm, scaler


# legacy alias so existing callers of pool_features still work
def pool_features(
    candidate_pool_cache: Dict[str, dict],
    fitting_images: Sequence[str],
    use_area: bool = True,
    scaler: Optional[FeatureScaler] = None,
):
    return pool_and_normalize(candidate_pool_cache, fitting_images, use_area=use_area, scaler=scaler)


# ---------------------------------------------------------------------------
# GMM fitting
# ---------------------------------------------------------------------------
def fit_global_gmm(
    candidate_pool_cache: Dict[str, dict],
    fitting_images: Sequence[str],
    gmm_preset: dict,
    use_area: bool = True,
    scaler: Optional[FeatureScaler] = None,
):
    """Pools, normalizes, and fits one GlobalGMM per gmm_preset.

    Returns (gmm, scaler). If scaler is None a new one is fitted from the
    pooled data; otherwise the provided scaler is reused (use this when
    all presets in a selection step should share the same normalization)."""
    X_norm, scaler = pool_and_normalize(
        candidate_pool_cache, fitting_images, use_area=use_area, scaler=scaler
    )

    if X_norm.shape[0] < 4:
        raise ValueError(
            f"Only {X_norm.shape[0]} candidate feature vectors pooled from "
            f"{len(list(fitting_images))} fitting images -- pool more images."
        )

    init_means = gmm_preset.get("init_means")
    init_covariances = gmm_preset.get("init_covariances")
    init_weights = gmm_preset.get("init_weights")

    gmm = GlobalGMM(
        learning_rate=gmm_preset.get("learning_rate", 1.0),
        max_iters=gmm_preset.get("max_iters", 100),
        tol=gmm_preset.get("tol", 1e-6),
        init_strategy=gmm_preset.get("init_strategy", "kmeans"),
        init_means=np.array(init_means) if init_means is not None else None,
        init_covariances=np.array(init_covariances) if init_covariances is not None else None,
        init_weights=np.array(init_weights) if init_weights is not None else None,
    )
    gmm.fit(X_norm)
    return gmm, scaler
