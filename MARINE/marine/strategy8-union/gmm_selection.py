"""
gmm_selection.py
=================

Decouples GMM-fit hyperparameter selection (init_strategy / means /
covariances, and optionally learning_rate if --tune_learning_rate) from
the (tau, alpha) grid search, per explicit instruction: running LVLM
generation for every candidate GMM configuration is far too slow, so
instead every candidate preset is fit on the SAME pooled tuning-image
features and scored with intrinsic, label-free cluster-quality metrics --
no generation, no GPU, just numpy/sklearn over the already-cached
features from candidate_pool.py.

Metrics used (all computed on the same pooled feature matrix X, using the
GMM's own hard cluster assignment argmax_k responsibility(x_i, k)):

  * silhouette score (sklearn.metrics.silhouette_score) -- the standard
    measure of how well-separated and internally-cohesive the two
    clusters are; ranges roughly [-1, 1], higher is better. This is the
    PRIMARY selection criterion: a well-separated positive/hallucinated
    split is exactly the functional property the offline sorter needs.
  * mean_separation -- the Euclidean distance between the two cluster
    means in the (already comparably-scaled, all roughly [0, 1]) raw
    feature space. A simple, directly-interpretable secondary signal.
  * log_likelihood -- the fitted model's own EM objective (Eq. 13), for
    reference/diagnostics. Not used as the primary criterion on its own,
    since a higher-likelihood fit can still have poorly-separated/
    low-confidence clusters (e.g. one huge diffuse component).

The winning preset is the one with the highest silhouette score; ties
broken by mean_separation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import silhouette_score

from fit_gmm import fit_global_gmm, pool_features
from gmm import GlobalGMM, GMMParams


def compute_gmm_quality(gmm: GlobalGMM, X: np.ndarray) -> Dict[str, float]:
    """Intrinsic (label-free) fit-quality metrics for an already-fit
    GlobalGMM, evaluated on feature matrix X (typically the same pool it
    was fit on)."""
    gamma_pos = gmm.responsibility_positive(X)
    hard_labels = (gamma_pos >= 0.5).astype(int)

    n_pos = int(hard_labels.sum())
    n_neg = int(len(hard_labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        # degenerate: the fit collapsed everything into one cluster --
        # silhouette is undefined (sklearn would raise), report worst case
        silhouette = -1.0
    else:
        silhouette = float(silhouette_score(X, hard_labels))

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
            "quality_by_preset": self.quality_by_preset,
            "n_fit_points": self.n_fit_points,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GMMSelectionResult":
        return cls(
            chosen_preset=d["chosen_preset"],
            chosen_gmm_params=GMMParams.from_dict(d["chosen_gmm_params"]),
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
    use_area: bool = False,
) -> GMMSelectionResult:
    """Fits every preset in `candidate_presets` on the SAME pooled
    tuning-image features and picks the one with the best intrinsic
    cluster quality (silhouette score, ties broken by mean separation).
    No LVLM generation is involved -- this is pure numpy/sklearn.
    use_area controls feature dimensionality and must match what will be
    passed to fit_global_gmm/classify_image_candidates (default: off)."""
    X = pool_features(candidate_pool_cache, fitting_images, use_area=use_area)
    if X.shape[0] < 4:
        raise ValueError(
            f"Only {X.shape[0]} candidate feature vectors pooled from "
            f"{len(fitting_images)} fitting images -- need more images."
        )

    quality_by_preset: Dict[str, Dict[str, float]] = {}
    gmm_by_preset: Dict[str, GlobalGMM] = {}

    for preset in candidate_presets:
        gmm = fit_global_gmm(candidate_pool_cache, fitting_images, preset, use_area=use_area)
        quality = compute_gmm_quality(gmm, X)
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
        quality_by_preset=quality_by_preset,
        n_fit_points=X.shape[0],
    )
