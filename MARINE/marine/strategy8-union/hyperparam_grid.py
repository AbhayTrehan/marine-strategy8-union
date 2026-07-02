"""
hyperparam_grid.py
===================

Defines the (bounded) hyperparameter search space for Strategy 8-U and the
F1 criterion used to pick a winner, per the user's spec:

    "Precision = 1 - CHAIRi, Recall is computed anyways as a metric ...
     choose the one that gives the best f1 score ... Be smart in creating
     the grid as it can't run forever."

Two families of hyperparameters are involved, and they are deliberately
decoupled (see gmm_selection.py and run_pipeline.py's
run_hyperparameter_search):

  * GMM-fit hyperparameters (init_strategy [+ init_means/init_covariances
    for 'fixed_prior'], and learning_rate/max_iters/tol if
    --tune_learning_rate is set) -- these are selected ONCE via an
    intrinsic fit-quality metric (silhouette score / cluster separation)
    computed directly on the pooled tuning-image features, with NO LVLM
    generation involved at all (gmm_selection.py). This is what makes it
    safe to consider several GMM presets without each one multiplying the
    cost of the expensive grid below.
  * Decoding hyperparameters (tau: the Eq. 15/16 responsibility threshold
    that splits O_pos/O_neg, and alpha: the Eq. 20 guidance strength) --
    changing EITHER one changes the text actually fed to the LVLM, so each
    distinct (tau, alpha) combination requires a fresh, real generation
    run over the tuning images. This is the EXPENSIVE part, and the only
    dimension build_grid() below actually searches over (gmm_presets is
    accepted for generality/testability but run_pipeline.py always passes
    a single, already-chosen preset).

DEFAULT_TAUS / DEFAULT_ALPHAS are intentionally a small, curated set
within the ranges that tend to matter (see the comments next to them)
rather than a dense sweep; if the cross product still exceeds
`max_trials`, build_grid() samples without replacement with a fixed seed
(reproducible), optionally forcing one specific combination to be
evaluated first via `preferred_first`.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# GMM-fit presets, split into two groups:
#
#   BASE_GMM_PRESETS    -- learning_rate=1.0, i.e. standard textbook EM
#                          (no M-step damping at all). This is what gets
#                          searched by default.
#   DAMPED_GMM_PRESETS  -- learning_rate<1.0 variants. Only included in the
#                          grid if you explicitly ask for it (see
#                          select_gmm_presets / run_pipeline.py's
#                          --tune_learning_rate flag) -- most of the time
#                          you do NOT need to tune this, since lr=1.0 (no
#                          damping) converges fine on a reasonably-sized
#                          pooled feature set; damping mainly helps if the
#                          tuning-image pool is small/noisy.
#
# 'fixed_prior' inits are expressed as *relative* score levels (not raw
# numbers pulled out of thin air): s_det/s_clip are roughly-calibrated
# confidence/similarity scores in [0, 1] for a correctly-grounded object
# (OWL-ViT confidence is typically well above 0.5 for an object that is
# genuinely visible and clearly named; CLIP image-text cosine similarity
# for a correct, simple "a photo of a X" prompt is typically ~0.25-0.35),
# vs. much lower values for an absent/ungrounded object. s_area is small
# for most individual objects in a scene. These are only an EM *starting
# point*; the user-facing means/covariances are exactly what gets searched
# under 'fixed_prior' below.
# ---------------------------------------------------------------------------
def _make_base_presets(use_area: bool = True) -> List[Dict]:
    """Returns the base (lr=1.0) GMM presets, with fixed_prior means/
    covariances in NORMALIZED space (after sqrt(area) + z-score), sized
    for the chosen feature dimensionality:
    use_area=True  (default): 3D [s_det, s_clip, sqrt(s_area)], all z-scored
    use_area=False:            2D [s_det, s_clip], z-scored

    In normalized space (roughly N(0,1) per dimension), the positive cluster
    (real objects) tends to lie above the mean (high detection confidence,
    reasonable clip similarity, some bounding-box area), and the negative
    cluster below it. The initializations below reflect this structure and
    serve as sensible starting points for the EM."""
    if use_area:
        fp_means = [[1.0, 0.8, 0.5], [-1.0, -0.8, -0.5]]
        fp_covs = [
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
        ]
    else:
        fp_means = [[1.0, 0.8], [-1.0, -0.8]]
        fp_covs = [
            [[0.5, 0.0], [0.0, 0.5]],
            [[0.5, 0.0], [0.0, 0.5]],
        ]
    return [
        {
            "name": "standard_kmeans",
            "learning_rate": 1.0,
            "max_iters": 100,
            "tol": 1e-6,
            "init_strategy": "kmeans",
        },
        {
            "name": "quantile_init",
            "learning_rate": 1.0,
            "max_iters": 100,
            "tol": 1e-6,
            "init_strategy": "quantile",
        },
        {
            "name": "fixed_prior",
            "learning_rate": 1.0,
            "max_iters": 100,
            "tol": 1e-6,
            "init_strategy": "fixed_prior",
            "init_means": fp_means,
            "init_covariances": fp_covs,
        },
    ]


BASE_GMM_PRESETS: List[Dict] = _make_base_presets(use_area=True)

DAMPED_GMM_PRESETS: List[Dict] = [
    {
        "name": "damped_kmeans_lr0.5",
        "learning_rate": 0.5,
        "max_iters": 200,
        "tol": 1e-6,
        "init_strategy": "kmeans",
    },
    {
        "name": "damped_kmeans_lr0.3",
        "learning_rate": 0.3,
        "max_iters": 300,
        "tol": 1e-6,
        "init_strategy": "kmeans",
    },
]


def select_gmm_presets(tune_learning_rate: bool = False, use_area: bool = True) -> List[Dict]:
    """tune_learning_rate=False (default): only lr=1.0 presets (standard,
    undamped EM) are searched -- the M-step damping dimension is fixed off.
    tune_learning_rate=True: the damped variants are ADDED to the grid too.
    use_area controls whether fixed_prior's init_means/covariances are sized
    for 2D [s_det, s_clip] (default) or 3D [s_det, s_clip, s_area]."""
    base = _make_base_presets(use_area=use_area)
    if tune_learning_rate:
        return base + DAMPED_GMM_PRESETS
    return base


DEFAULT_GMM_PRESETS: List[Dict] = select_gmm_presets(tune_learning_rate=False)

# Per explicit guidance: alpha in [0.5, 0.8] (higher guidance strength
# tends to help more, per the original MARINE paper's own ablation), tau
# in [0.2, 0.5] (lower decision threshold than the textbook 0.5 default,
# since this pipeline's candidate pool also includes open-vocabulary VLM/
# RAM++ mentions that may sit at moderate-but-real responsibility values).
DEFAULT_TAUS: List[float] = [0.2, 0.3, 0.4, 0.5]
DEFAULT_ALPHAS: List[float] = [0.5, 0.6, 0.7, 0.8]


@dataclass
class TrialConfig:
    trial_id: str
    gmm_preset: Dict
    tau: float
    alpha: float

    def to_dict(self) -> dict:
        return {"trial_id": self.trial_id, "gmm_preset": self.gmm_preset, "tau": self.tau, "alpha": self.alpha}

    @classmethod
    def from_dict(cls, d: dict) -> "TrialConfig":
        return cls(trial_id=d["trial_id"], gmm_preset=d["gmm_preset"], tau=d["tau"], alpha=d["alpha"])


def build_grid(
    gmm_presets: Optional[Sequence[Dict]] = None,
    taus: Optional[Sequence[float]] = None,
    alphas: Optional[Sequence[float]] = None,
    max_trials: Optional[int] = 12,
    seed: int = 0,
    preferred_first: Optional[Dict[str, float]] = None,
) -> List[TrialConfig]:
    """Builds the (possibly down-sampled) list of TrialConfig to actually
    evaluate. If the full cross product of presets x taus x alphas is
    larger than max_trials, sample max_trials of them without replacement
    (seeded, so the grid is reproducible) rather than only ever trying the
    first N in iteration order.

    preferred_first: optional {"tau": ..., "alpha": ...} (and optionally
    "gmm_preset_name") identifying one specific combination that should be
    evaluated FIRST, regardless of sampling order -- and, if max_trials
    capping would otherwise drop it, it is guaranteed a slot rather than
    left to chance. If no matching combination exists in the (gmm_presets x
    taus x alphas) space, this is a no-op (grid is built normally).
    """
    gmm_presets = list(gmm_presets) if gmm_presets is not None else DEFAULT_GMM_PRESETS
    taus = list(taus) if taus is not None else DEFAULT_TAUS
    alphas = list(alphas) if alphas is not None else DEFAULT_ALPHAS

    full = list(itertools.product(gmm_presets, taus, alphas))

    preferred_entry = None
    if preferred_first is not None:
        want_tau = preferred_first.get("tau")
        want_alpha = preferred_first.get("alpha")
        want_preset_name = preferred_first.get("gmm_preset_name")
        for entry in full:
            preset, tau, alpha = entry
            if want_tau is not None and abs(tau - want_tau) > 1e-9:
                continue
            if want_alpha is not None and abs(alpha - want_alpha) > 1e-9:
                continue
            if want_preset_name is not None and preset["name"] != want_preset_name:
                continue
            preferred_entry = entry
            break

    if max_trials is not None and len(full) > max_trials:
        rng = random.Random(seed)
        if preferred_entry is not None:
            remaining = [e for e in full if e != preferred_entry]
            sampled_rest = rng.sample(remaining, max_trials - 1)
            full = [preferred_entry] + sampled_rest
        else:
            full = rng.sample(full, max_trials)
    elif preferred_entry is not None:
        full = [preferred_entry] + [e for e in full if e != preferred_entry]

    trials: List[TrialConfig] = []
    for preset, tau, alpha in full:
        trial_id = f"{preset['name']}__tau{tau}__alpha{alpha}"
        trials.append(TrialConfig(trial_id=trial_id, gmm_preset=preset, tau=tau, alpha=alpha))
    return trials


def chair_f1(chair_i: float, recall: float) -> float:
    """F1 from CHAIR metrics, per the user's explicit instruction:
    Precision := 1 - CHAIRi (fraction of mentioned-and-grounded words that
    are NOT hallucinated), Recall := CHAIR's own Recall metric (fraction of
    ground-truth objects actually mentioned). Standard harmonic mean;
    returns 0.0 in the degenerate P=R=0 case instead of raising."""
    precision = 1.0 - chair_i
    if precision < 0.0:
        precision = 0.0
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


@dataclass
class TrialResult:
    trial: TrialConfig
    chair_s: float
    chair_i: float
    recall: float
    f1: float
    n_images: int
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trial": self.trial.to_dict(),
            "chair_s": self.chair_s,
            "chair_i": self.chair_i,
            "recall": self.recall,
            "f1": self.f1,
            "n_images": self.n_images,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrialResult":
        return cls(
            trial=TrialConfig.from_dict(d["trial"]),
            chair_s=d["chair_s"],
            chair_i=d["chair_i"],
            recall=d["recall"],
            f1=d["f1"],
            n_images=d["n_images"],
            extra=d.get("extra", {}),
        )


def pick_best(results: Sequence[TrialResult]) -> TrialResult:
    if not results:
        raise ValueError("no trial results to pick from")
    return max(results, key=lambda r: r.f1)
