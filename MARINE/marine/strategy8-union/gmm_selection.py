"""
gmm_selection.py
=================

Decouples GMM-fit hyperparameter selection from the (tau, alpha) grid search.
Every candidate preset is fit on the SAME pooled features (with the SAME
scaler, fitted once) and scored with intrinsic label-free cluster-quality
metrics (silhouette score / cluster separation) -- no LVLM generation,
no GPU, just numpy/sklearn over the already-cached features.

The FeatureScaler (sqrt(s_area) + z-score) is fitted ONCE from the pooled
tuning data and stored in GMMSelectionResult alongside the best GMM params,
so downstream code (build_question_file.py, run_pipeline.py) can apply the
exact same transform at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import silhouette_score

from fit_gmm import FeatureScaler, fit_global_gmm, pool_raw_features
from gmm import GlobalGMM, GMMParams


def compute_gmm_quality(gmm: GlobalGMM, X_norm: np.ndarray) -> Dict[str, float]:
    """Intrinsic (label-free) fit-quality metrics for an already-fit
    GlobalGMM, evaluated on NORMALIZED feature matrix X_norm."""
    gamma_pos = gmm.responsibility_positive(X_norm)
    hard_labels = (gamma_pos >= 0.5).astype(int)

    n_pos = int(hard_labels.sum())
    n_neg = int(len(hard_labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        silhouette = -1.0
    else:
        silhouette = float(silhouette_score(X_norm, hard_labels))

    pos_idx = gmm.params.pos_idx
    neg_idx = 1 - pos_idx
    mean_separation = float(np.linalg.norm(gmm.params.means[pos_idx] - gmm.params.means[neg_idx]))

    return {
        "silhouette": silhouette,
        "mean_separation": mean_separation,
        "log_likelihood": float(gmm.params.log_likelihood),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "converged": bool(gmm.params.converged),
        "n_iter": int(gmm.params.n_iter),
    }


@dataclass
class GMMSelectionResult:
    chosen_preset: Dict
    chosen_gmm_params: GMMParams
    chosen_scaler: FeatureScaler
    quality_by_preset: Dict[str, Dict[str, float]]
    n_fit_points: int

    @property
    def chosen_preset_name(self) -> str:
        return self.chosen_preset["name"]

    @property
    def chosen_quality(self) -> Dict[str, float]:
        return self.quality_by_preset[self.chosen_preset_name]

    def to_dict(self) -> dict:
        return {
            "chosen_preset": self.chosen_preset,
            "chosen_gmm_params": self.chosen_gmm_params.to_dict(),
            "chosen_scaler": self.chosen_scaler.to_dict(),
            "quality_by_preset": self.quality_by_preset,
            "n_fit_points": self.n_fit_points,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GMMSelectionResult":
        return cls(
            chosen_preset=d["chosen_preset"],
            chosen_gmm_params=GMMParams.from_dict(d["chosen_gmm_params"]),
            chosen_scaler=FeatureScaler.from_dict(d["chosen_scaler"]),
            quality_by_preset=d["quality_by_preset"],
            n_fit_points=d["n_fit_points"],
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GMMSelectionResult":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def select_best_gmm_preset(
    candidate_pool_cache: Dict[str, dict],
    fitting_images: Sequence[str],
    candidate_presets: Sequence[Dict],
    use_area: bool = True,
) -> GMMSelectionResult:
    """Fits every preset on the SAME pooled tuning-image features with the
    SAME scaler (fitted once), picks the one with the best silhouette score.
    No LVLM generation -- pure numpy/sklearn."""

    # Fit the scaler ONCE from the pooled raw features, then share it
    # across all presets so they're evaluated on identical normalized data.
    X_raw = pool_raw_features(candidate_pool_cache, fitting_images, use_area=use_area)
    if X_raw.shape[0] < 4:
        raise ValueError(
            f"Only {X_raw.shape[0]} candidate feature vectors pooled from "
            f"{len(list(fitting_images))} fitting images -- need more images."
        )
    shared_scaler = FeatureScaler.fit(X_raw, use_area=use_area)
    X_norm = shared_scaler.transform(X_raw)

    quality_by_preset: Dict[str, Dict[str, float]] = {}
    gmm_by_preset: Dict[str, GlobalGMM] = {}

    for preset in candidate_presets:
        # Reuse the shared scaler (no re-fitting for each preset)
        gmm, _ = fit_global_gmm(
            candidate_pool_cache, fitting_images, preset,
            use_area=use_area, scaler=shared_scaler,
        )
        quality = compute_gmm_quality(gmm, X_norm)
        quality_by_preset[preset["name"]] = quality
        gmm_by_preset[preset["name"]] = gmm

    best_name = max(
        quality_by_preset,
        key=lambda name: (quality_by_preset[name]["silhouette"], quality_by_preset[name]["mean_separation"]),
    )
    chosen_preset = next(p for p in candidate_presets if p["name"] == best_name)
    chosen_gmm = gmm_by_preset[best_name]

    return GMMSelectionResult(
        chosen_preset=chosen_preset,
        chosen_gmm_params=chosen_gmm.params,
        chosen_scaler=shared_scaler,
        quality_by_preset=quality_by_preset,
        n_fit_points=X_raw.shape[0],
    )
