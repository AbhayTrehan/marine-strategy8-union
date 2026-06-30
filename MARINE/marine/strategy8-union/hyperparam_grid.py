"""
hyperparam_grid.py
===================

Defines the (bounded) hyperparameter search space for Strategy 8-U and the
F1 criterion used to pick a winner, per the user's spec:

    "Precision = 1 - CHAIRi, Recall is computed anyways as a metric ...
     choose the one that gives the best f1 score ... Be smart in creating
     the grid as it can't run forever."

Two families of hyperparameters are involved:

  * GMM-fit hyperparameters (learning_rate, max_iters, tol, init_strategy
    [+ init_means/init_covariances for 'fixed_prior']) -- these only affect
    the *global* GMM fit (gmm.py), which is pure numpy over cached feature
    vectors. Trying many of these is CHEAP (no GPU, no LVLM calls).
  * Decoding hyperparameters (tau: the Eq. 15/16 responsibility threshold
    that splits O_pos/O_neg, and alpha: the Eq. 20 guidance strength) --
    changing EITHER one changes the text actually fed to the LVLM, so each
    distinct (GMM-fit, tau, alpha) combination requires a fresh, real
    generation run over the tuning images. This is the EXPENSIVE part.

Because of that asymmetry, we deliberately keep this a small, curated
search rather than a dense cross product: a handful of named GMM-fit
"presets" (each a sensible, internally-consistent combination of
learning_rate/max_iters/tol/init_strategy) crossed with a handful of tau
and alpha values. If the full cross product exceeds `max_trials`, we
sample without replacement with a fixed seed for reproducibility, rather
than silently truncating the grid -- so a smaller `max_trials` still
explores the *whole* hyperparameter space, just more sparsely.
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
BASE_GMM_PRESETS: List[Dict] = [
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
        "init_means": [[0.6, 0.28, 0.08], [0.05, 0.12, 0.02]],
        "init_covariances": [
            [[0.05, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.02]],
            [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.01]],
        ],
    },
]

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


def select_gmm_presets(tune_learning_rate: bool = False) -> List[Dict]:
    """tune_learning_rate=False (default): only lr=1.0 presets (standard,
    undamped EM) are searched -- the M-step damping dimension is fixed
    off. tune_learning_rate=True: the damped variants are ADDED to the
    grid as well, so the search also explores lr<1.0."""
    if tune_learning_rate:
        return BASE_GMM_PRESETS + DAMPED_GMM_PRESETS
    return list(BASE_GMM_PRESETS)


DEFAULT_GMM_PRESETS: List[Dict] = select_gmm_presets(tune_learning_rate=False)

DEFAULT_TAUS: List[float] = [0.4, 0.5, 0.6]
DEFAULT_ALPHAS: List[float] = [0.3, 0.5, 0.7]


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
) -> List[TrialConfig]:
    """Builds the (possibly down-sampled) list of TrialConfig to actually
    evaluate. If the full cross product of presets x taus x alphas is
    larger than max_trials, sample max_trials of them without replacement
    (seeded, so the grid is reproducible) rather than only ever trying the
    first N in iteration order."""
    gmm_presets = list(gmm_presets) if gmm_presets is not None else DEFAULT_GMM_PRESETS
    taus = list(taus) if taus is not None else DEFAULT_TAUS
    alphas = list(alphas) if alphas is not None else DEFAULT_ALPHAS

    full = list(itertools.product(gmm_presets, taus, alphas))
    if max_trials is not None and len(full) > max_trials:
        rng = random.Random(seed)
        full = rng.sample(full, max_trials)

    trials: List[TrialConfig] = []
    for i, (preset, tau, alpha) in enumerate(full):
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
