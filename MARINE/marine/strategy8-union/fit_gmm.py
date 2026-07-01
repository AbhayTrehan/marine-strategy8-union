"""
fit_gmm.py
==========

Step B of the Strategy 8-U pipeline: pool the feature vectors of every
candidate object across a set of "fitting" images (the tuning split from
splits.py) and fit ONE global 2-component GMM (gmm.py) on the pooled set,
per Eq. 7-14 -- this is the "fit-on-train" half of the global fit/freeze/
apply design confirmed with the user. Pure numpy over candidate_pool.py's
cache; no LVLM/vision-model calls, so this (and applying the frozen result
in build_question_file.py) is the CHEAP part of a hyperparameter trial.

Feature dimensions: by default only [s_det, s_clip] are used (2D). s_area
is excluded by default because it conflates "small object" with "hallucinated
object" -- a tiny but genuinely present object (a spoon in the background,
a distant bird) has near-zero s_area and gets pulled into the negative cluster
even when s_det and s_clip are reasonable. Pass use_area=True to restore the
original 3D [s_det, s_clip, s_area] behaviour; s_area is always stored in
the pool cache regardless, so this can be changed without rebuilding the cache.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence

import numpy as np

from gmm import GlobalGMM, GMMParams

FEATURE_DIMS_NO_AREA = ["s_det", "s_clip"]
FEATURE_DIMS_WITH_AREA = ["s_det", "s_clip", "s_area"]


def pool_features(
    candidate_pool_cache: Dict[str, dict],
    fitting_images: Sequence[str],
    use_area: bool = False,
) -> np.ndarray:
    """Stacks x_i for every candidate object of every image in
    `fitting_images` into one (N, D) array.
    use_area=False (default): D=2, x_i = [s_det, s_clip]
    use_area=True:            D=3, x_i = [s_det, s_clip, s_area]
    """
    dims = FEATURE_DIMS_WITH_AREA if use_area else FEATURE_DIMS_NO_AREA
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


def fit_global_gmm(
    candidate_pool_cache: Dict[str, dict],
    fitting_images: Sequence[str],
    gmm_preset: dict,
    use_area: bool = False,
) -> GlobalGMM:
    """`gmm_preset` is one of hyperparam_grid.py's preset dicts: must
    contain learning_rate, max_iters, tol, init_strategy, and (for
    init_strategy == 'fixed_prior') init_means / init_covariances.
    use_area controls whether s_area is included as a feature (default off,
    see module docstring)."""
    X = pool_features(candidate_pool_cache, fitting_images, use_area=use_area)
    if X.shape[0] < 4:
        raise ValueError(
            f"Only {X.shape[0]} candidate feature vectors pooled from "
            f"{len(fitting_images)} fitting images -- need more images or "
            f"a larger fitting set to fit a stable global GMM."
        )

    kwargs = dict(
        learning_rate=gmm_preset["learning_rate"],
        max_iters=gmm_preset["max_iters"],
        tol=gmm_preset["tol"],
        init_strategy=gmm_preset["init_strategy"],
    )
    if gmm_preset["init_strategy"] == "fixed_prior":
        kwargs["init_means"] = np.array(gmm_preset["init_means"], dtype=float)
        kwargs["init_covariances"] = np.array(gmm_preset["init_covariances"], dtype=float)
        if "init_weights" in gmm_preset:
            kwargs["init_weights"] = np.array(gmm_preset["init_weights"], dtype=float)

    gmm = GlobalGMM(**kwargs)
    gmm.fit(X)
    return gmm


def main():
    from candidate_pool import load_candidate_pool_cache

    parser = argparse.ArgumentParser(description="Strategy8-U Step B: fit the global GMM")
    parser.add_argument("--candidate_pool_cache", type=str, required=True)
    parser.add_argument("--fitting_images_file", type=str, required=True,
                        help="JSON file: list of image filenames to pool features from")
    parser.add_argument("--gmm_preset_file", type=str, required=True,
                        help="JSON file containing one GMM preset dict (see hyperparam_grid.py)")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    cache = load_candidate_pool_cache(args.candidate_pool_cache)
    with open(args.fitting_images_file) as f:
        fitting_images = json.load(f)
    with open(args.gmm_preset_file) as f:
        gmm_preset = json.load(f)

    gmm = fit_global_gmm(cache, fitting_images, gmm_preset)
    gmm.params.save(args.output_file)
    print(f"[Strategy8-U][Step B] Fit global GMM on {gmm.params.n_fit_points} candidates "
          f"from {len(fitting_images)} images ({gmm.params.n_iter} EM iterations, "
          f"converged={gmm.params.converged}). Saved to {args.output_file}")


if __name__ == "__main__":
    main()
